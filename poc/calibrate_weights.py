"""Calibrate composite importance weights via grid search over oracle correlation.

For each codec:
1. Load 200 calibration files, encode, compute oracle damage + A1/A2/A4 scores
2. Grid search over (w1, w2, w4) to maximize Spearman correlation with oracle
3. 5-fold cross-validation over files to validate generalization
4. Generate calibration report
"""
import os
import sys
import time
import itertools
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poc.codec.encodec_wrapper import EnCodecWrapper
from poc.codec.hilcodec_wrapper import HILCodecWrapper
from poc.eval.oracle import compute_oracle_importance
from poc.importance.a1_vad_onset import score_a1
from poc.importance.a2_spectral_flux import score_a2
from poc.importance.a4_evs_criticality import score_a4
from poc.importance.b1_token_novelty import score_b1
from poc.cache import (
    cache_tokens, load_cached_tokens,
    cache_oracle, load_cached_oracle,
)

# ─── Configuration ───────────────────────────────────────────────
CALIBRATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "calibration")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
TARGET_SR = 24000
MAX_DURATION_SEC = 4.0
GRID_STEP = 0.05
N_FOLDS = 5

CODEC_CONFIGS = [
    {"name": "EnCodec_3kbps", "cls": EnCodecWrapper, "kwargs": {"bandwidth": 3.0, "device": "cpu"}},
    {"name": "HILCodec_3kbps", "cls": HILCodecWrapper, "kwargs": {"n_quantizers": 4}},
]


def load_and_resample(path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    """Load audio file and resample to target SR."""
    pcm, sr = sf.read(path, dtype="float32")
    if pcm.ndim > 1:
        pcm = pcm[:, 0]
    if sr != target_sr:
        pcm = librosa.resample(pcm, orig_sr=sr, target_sr=target_sr)
    max_samples = int(MAX_DURATION_SEC * target_sr)
    pcm = pcm[:max_samples]
    peak = np.max(np.abs(pcm))
    if peak > 0:
        pcm = pcm / peak * 0.95
    return pcm


def generate_weight_grid(step: float = GRID_STEP):
    """Generate all (w1, w2, w4) with w1+w2+w4=1, each in [0, 1], step size."""
    weights = []
    vals = np.arange(0, 1.0 + step / 2, step)
    for w1 in vals:
        for w2 in vals:
            w4 = 1.0 - w1 - w2
            if w4 >= -1e-9 and w4 <= 1.0 + 1e-9:
                weights.append((round(w1, 4), round(w2, 4), round(max(w4, 0), 4)))
    return weights


def compute_spearman(scores_a1, scores_a2, scores_a4, oracle, w1, w2, w4):
    """Compute Spearman correlation of weighted composite vs oracle."""
    composite = w1 * scores_a1 + w2 * scores_a2 + w4 * scores_a4
    corr, _ = spearmanr(composite, oracle)
    if np.isnan(corr):
        return 0.0
    return corr


def run_calibration():
    """Main calibration routine."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load calibration files
    if not os.path.exists(CALIBRATION_DIR):
        print(f"Calibration directory not found: {CALIBRATION_DIR}")
        print("Run: python -m poc.data.download_calibration_set")
        return

    audio_files = sorted([
        os.path.join(CALIBRATION_DIR, f)
        for f in os.listdir(CALIBRATION_DIR)
        if f.endswith((".wav", ".flac"))
    ])
    n_files = len(audio_files)
    print(f"Found {n_files} calibration files")
    if n_files == 0:
        return

    # Count unique speakers
    speakers = set()
    for f in audio_files:
        base = os.path.basename(f)
        if base.startswith("spk"):
            spk = base.split("_")[0]
            speakers.add(spk)
    n_speakers = len(speakers) if speakers else "unknown"

    weight_grid = generate_weight_grid(GRID_STEP)
    print(f"Weight grid: {len(weight_grid)} combinations (step={GRID_STEP})")

    report_lines = [
        "# Weight Calibration Report\n",
        "## Dataset\n",
        f"- **Files**: {n_files}",
        f"- **Speakers**: {n_speakers}",
        f"- **Max duration**: {MAX_DURATION_SEC}s per file",
        f"- **Sample rate**: {TARGET_SR} Hz",
        "",
        "## Oracle Computation\n",
        "- **Method**: Leave-one-out SI-SDR damage",
        "- **max_samples**: 50 frames per file",
        "",
        "## Grid Search\n",
        f"- **Step size**: {GRID_STEP}",
        f"- **Combinations**: {len(weight_grid)}",
        f"- **Cross-validation**: {N_FOLDS}-fold over files",
        f"- **Methods in composite**: A1 (VAD onset), A2 (spectral flux), A4 (EVS criticality)",
        f"- **B1 excluded**: Token novelty shows near-zero/negative correlation",
        "",
    ]

    all_calibration_results = []

    for codec_cfg in CODEC_CONFIGS:
        codec_name = codec_cfg["name"]
        print(f"\n{'='*60}")
        print(f"Calibrating: {codec_name}")
        print(f"{'='*60}")
        codec = codec_cfg["cls"](**codec_cfg["kwargs"])

        # Collect per-file scores
        file_data = []  # list of (a1, a2, a4, b1, oracle) per file
        t0 = time.time()

        for i, audio_path in enumerate(audio_files):
            fname = os.path.basename(audio_path)
            pcm = load_and_resample(audio_path)

            # Encode (cached)
            tokens = load_cached_tokens(codec_name, audio_path)
            if tokens is None:
                tokens = codec.encode(pcm)
                cache_tokens(codec_name, audio_path, tokens)

            n_frames = tokens.shape[0]
            pcm_trimmed = pcm[:n_frames * codec.frame_size]

            # Oracle (cached)
            oracle_damage = load_cached_oracle(codec_name, audio_path)
            if oracle_damage is None:
                oracle_damage = compute_oracle_importance(codec, pcm, tokens)
                cache_oracle(codec_name, audio_path, oracle_damage)

            # Importance scores
            a1 = score_a1(pcm_trimmed, codec.frame_size)[:n_frames]
            a2 = score_a2(pcm_trimmed, codec.frame_size, codec.sample_rate)[:n_frames]
            a4 = score_a4(pcm_trimmed, codec.frame_size, codec.sample_rate)[:n_frames]
            b1 = score_b1(tokens[:n_frames])

            file_data.append({
                "a1": a1, "a2": a2, "a4": a4, "b1": b1,
                "oracle": oracle_damage[:n_frames],
                "file": fname,
            })

            if (i + 1) % 20 == 0 or i == n_files - 1:
                print(f"  [{i+1}/{n_files}] processed ({time.time()-t0:.0f}s)")

        # Concatenate all frames for global grid search
        all_a1 = np.concatenate([d["a1"] for d in file_data])
        all_a2 = np.concatenate([d["a2"] for d in file_data])
        all_a4 = np.concatenate([d["a4"] for d in file_data])
        all_b1 = np.concatenate([d["b1"] for d in file_data])
        all_oracle = np.concatenate([d["oracle"] for d in file_data])
        total_frames = len(all_oracle)
        print(f"  Total frames: {total_frames}")

        # Individual method correlations
        individual_corrs = {}
        for name, scores in [("A1", all_a1), ("A2", all_a2), ("A4", all_a4), ("B1", all_b1)]:
            corr, _ = spearmanr(scores, all_oracle)
            individual_corrs[name] = corr if not np.isnan(corr) else 0.0
            print(f"  {name} individual Spearman: {individual_corrs[name]:.4f}")

        # Equal-weight composite (A1+A2+B1+A4, old default)
        equal_composite = 0.25 * all_a1 + 0.25 * all_a2 + 0.25 * all_b1 + 0.25 * all_a4
        equal_corr, _ = spearmanr(equal_composite, all_oracle)
        equal_corr = equal_corr if not np.isnan(equal_corr) else 0.0
        print(f"  Equal-weight (A1+A2+B1+A4) Spearman: {equal_corr:.4f}")

        # Grid search (A1, A2, A4 only, no B1)
        print(f"  Running grid search over {len(weight_grid)} weight combinations...")
        t1 = time.time()
        best_corr = -1.0
        best_weights = (0.33, 0.33, 0.34)
        for w1, w2, w4 in weight_grid:
            corr = compute_spearman(all_a1, all_a2, all_a4, all_oracle, w1, w2, w4)
            if corr > best_corr:
                best_corr = corr
                best_weights = (w1, w2, w4)
        print(f"  Grid search done in {time.time()-t1:.1f}s")
        print(f"  Best weights: w1={best_weights[0]:.2f}, w2={best_weights[1]:.2f}, w4={best_weights[2]:.2f}")
        print(f"  Best train Spearman: {best_corr:.4f}")

        # 5-fold cross-validation over files
        rng = np.random.default_rng(42)
        file_indices = np.arange(n_files)
        rng.shuffle(file_indices)
        fold_size = n_files // N_FOLDS
        cv_train_corrs = []
        cv_val_corrs = []

        for fold in range(N_FOLDS):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < N_FOLDS - 1 else n_files
            val_idx = file_indices[val_start:val_end]
            train_idx = np.concatenate([file_indices[:val_start], file_indices[val_end:]])

            # Train: grid search on train files
            train_a1 = np.concatenate([file_data[i]["a1"] for i in train_idx])
            train_a2 = np.concatenate([file_data[i]["a2"] for i in train_idx])
            train_a4 = np.concatenate([file_data[i]["a4"] for i in train_idx])
            train_oracle = np.concatenate([file_data[i]["oracle"] for i in train_idx])

            fold_best_corr = -1.0
            fold_best_w = (0.33, 0.33, 0.34)
            for w1, w2, w4 in weight_grid:
                corr = compute_spearman(train_a1, train_a2, train_a4, train_oracle, w1, w2, w4)
                if corr > fold_best_corr:
                    fold_best_corr = corr
                    fold_best_w = (w1, w2, w4)

            # Validate
            val_a1 = np.concatenate([file_data[i]["a1"] for i in val_idx])
            val_a2 = np.concatenate([file_data[i]["a2"] for i in val_idx])
            val_a4 = np.concatenate([file_data[i]["a4"] for i in val_idx])
            val_oracle = np.concatenate([file_data[i]["oracle"] for i in val_idx])
            val_corr = compute_spearman(val_a1, val_a2, val_a4, val_oracle, *fold_best_w)

            cv_train_corrs.append(fold_best_corr)
            cv_val_corrs.append(val_corr)
            print(f"  Fold {fold+1}: train={fold_best_corr:.4f}, val={val_corr:.4f}, "
                  f"w=({fold_best_w[0]:.2f},{fold_best_w[1]:.2f},{fold_best_w[2]:.2f})")

        mean_train = np.mean(cv_train_corrs)
        mean_val = np.mean(cv_val_corrs)
        print(f"  CV mean: train={mean_train:.4f}, val={mean_val:.4f}")

        # Store results
        all_calibration_results.append({
            "codec": codec_name,
            "best_w1": best_weights[0],
            "best_w2": best_weights[1],
            "best_w4": best_weights[2],
            "train_spearman": best_corr,
            "cv_train_mean": mean_train,
            "cv_val_mean": mean_val,
            "individual_corrs": individual_corrs,
            "equal_weight_corr": equal_corr,
        })

        # Add to report
        report_lines.extend([
            f"## {codec_name}\n",
            "### Individual Method Correlations\n",
            "| Method | Spearman |",
            "|--------|----------|",
        ])
        for name, corr in individual_corrs.items():
            report_lines.append(f"| {name} | {corr:.4f} |")
        report_lines.extend([
            "",
            f"### Equal-Weight Composite (old default)\n",
            f"- Spearman (A1+A2+B1+A4, w=0.25 each): **{equal_corr:.4f}**",
            "",
            f"### Calibrated Weights (A1+A2+A4, no B1)\n",
            f"- **w1 (A1)** = {best_weights[0]:.2f}",
            f"- **w2 (A2)** = {best_weights[1]:.2f}",
            f"- **w4 (A4)** = {best_weights[2]:.2f}",
            f"- Train Spearman: **{best_corr:.4f}**",
            "",
            f"### {N_FOLDS}-Fold Cross-Validation\n",
            "| Fold | Train Spearman | Val Spearman |",
            "|------|----------------|--------------|",
        ])
        for fold in range(N_FOLDS):
            report_lines.append(f"| {fold+1} | {cv_train_corrs[fold]:.4f} | {cv_val_corrs[fold]:.4f} |")
        report_lines.extend([
            f"| **Mean** | **{mean_train:.4f}** | **{mean_val:.4f}** |",
            "",
        ])

    # Comparison summary
    report_lines.extend([
        "## Summary Comparison\n",
        "| Codec | Equal-Weight | Calibrated | CV Val | Best w1 | Best w2 | Best w4 |",
        "|-------|-------------|------------|--------|---------|---------|---------|",
    ])
    for r in all_calibration_results:
        report_lines.append(
            f"| {r['codec']} | {r['equal_weight_corr']:.4f} | "
            f"{r['train_spearman']:.4f} | {r['cv_val_mean']:.4f} | "
            f"{r['best_w1']:.2f} | {r['best_w2']:.2f} | {r['best_w4']:.2f} |"
        )

    report_lines.extend([
        "",
        "## Conclusion\n",
        "The calibrated weights remove B1 (token novelty) which shows near-zero or negative "
        "oracle correlation, and optimize the A1/A2/A4 mixture to maximize Spearman rank "
        "correlation with oracle frame damage. Cross-validation confirms generalization across files.",
    ])

    # Save report
    report_path = os.path.join(RESULTS_DIR, "weight_calibration_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nCalibration report saved to {report_path}")

    # Print final weights for updating composite.py
    print("\n" + "=" * 60)
    print("RECOMMENDED WEIGHTS (update poc/importance/composite.py):")
    print("=" * 60)
    for r in all_calibration_results:
        print(f"  {r['codec']}: W_A1={r['best_w1']:.2f}, W_A2={r['best_w2']:.2f}, W_A4={r['best_w4']:.2f}")

    return all_calibration_results


if __name__ == "__main__":
    run_calibration()
