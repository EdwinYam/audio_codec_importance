"""CLI entry point: run the full experiment matrix and save results."""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import soundfile as sf
import librosa

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poc.codec.encodec_wrapper import EnCodecWrapper
from poc.pipeline import run_single_experiment
from poc.eval.oracle import compute_oracle_importance
from poc.eval.diagnostics import oracle_spearman, precision_at_k
from poc.importance.composite import score_individual


# ─── Configuration ───────────────────────────────────────────────
NETWORK_TYPES = ["random_loss"]
PLRS = [0.01, 0.03, 0.05]
PROTECTION_METHODS = ["none", "random", "heuristic", "importance_aware"]
BUDGET_FRAC = 0.10  # 10% redundancy budget
SEEDS = [42, 123]  # 2 seeds for averaging
TARGET_SR = 24000
MAX_DURATION_SEC = 4.0  # Truncate long files for speed


def load_audio_files(data_dir: str):
    """Load .wav/.flac files from data directory."""
    files = []
    for f in sorted(os.listdir(data_dir)):
        if f.endswith((".wav", ".flac")):
            files.append(os.path.join(data_dir, f))
    return files


def load_and_resample(path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    """Load audio file and resample to target SR."""
    pcm, sr = sf.read(path, dtype="float32")
    if pcm.ndim > 1:
        pcm = pcm[:, 0]  # mono
    if sr != target_sr:
        pcm = librosa.resample(pcm, orig_sr=sr, target_sr=target_sr)
    # Truncate
    max_samples = int(MAX_DURATION_SEC * target_sr)
    pcm = pcm[:max_samples]
    # Normalize
    peak = np.max(np.abs(pcm))
    if peak > 0:
        pcm = pcm / peak * 0.95
    return pcm


def run_full_matrix(data_dir: str, results_dir: str):
    """Run full experiment matrix."""
    os.makedirs(results_dir, exist_ok=True)

    print("Loading EnCodec 24kHz model...")
    codec = EnCodecWrapper(bandwidth=3.0, device="cpu")
    print(f"Codec loaded: SR={codec.sample_rate}, frame_size={codec.frame_size}, "
          f"codebooks={codec.n_codebooks}")

    audio_files = load_audio_files(data_dir)
    if not audio_files:
        print(f"No audio files found in {data_dir}")
        return

    print(f"Found {len(audio_files)} audio files")

    all_results = []
    oracle_diagnostics = []

    for audio_idx, audio_path in enumerate(audio_files):
        fname = os.path.basename(audio_path)
        print(f"\n[{audio_idx+1}/{len(audio_files)}] Processing: {fname}")

        pcm = load_and_resample(audio_path)
        print(f"  Audio: {len(pcm)} samples, {len(pcm)/TARGET_SR:.2f}s")

        # Encode once
        tokens = codec.encode(pcm)
        n_frames = tokens.shape[0]
        print(f"  Encoded: {n_frames} frames, {tokens.shape[1]} codebooks")

        # Compute oracle importance (expensive but only once per file)
        print("  Computing oracle importance (leave-one-out)...")
        t0 = time.time()
        oracle_damage = compute_oracle_importance(codec, pcm, tokens)
        print(f"  Oracle computed in {time.time()-t0:.1f}s")

        # Compute all importance method scores
        scores = score_individual(pcm, tokens, codec.frame_size, codec.sample_rate)

        # Diagnostic: compare each method to oracle
        for method_name, method_scores in scores.items():
            sp = oracle_spearman(method_scores, oracle_damage)
            pk = precision_at_k(method_scores, oracle_damage, k_frac=0.2)
            oracle_diagnostics.append({
                "file": fname,
                "method": method_name,
                "spearman_corr": sp,
                "precision_at_20pct": pk,
            })
            print(f"  {method_name}: Spearman={sp:.3f}, P@20%={pk:.3f}")

        # Run experiment matrix
        for network in NETWORK_TYPES:
            for plr in PLRS:
                for method in PROTECTION_METHODS:
                    for seed in SEEDS:
                        result = run_single_experiment(
                            pcm, codec, network, plr, method,
                            budget_frac=BUDGET_FRAC, seed=seed,
                        )
                        result["file"] = fname
                        result["seed"] = seed
                        all_results.append(result)

    # Save results
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(results_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(f"Total experiments: {len(all_results)}")

    # Save oracle diagnostics
    df_diag = pd.DataFrame(oracle_diagnostics)
    diag_path = os.path.join(results_dir, "oracle_diagnostics.csv")
    df_diag.to_csv(diag_path, index=False)
    print(f"Oracle diagnostics saved to {diag_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY (averaged over files and seeds)")
    print("=" * 60)
    summary = df.groupby(["network_type", "target_plr", "protection_method"]).agg({
        "PESQ": "mean",
        "STOI": "mean",
        "ESTOI": "mean",
        "SI-SDR": "mean",
        "post_repair_loss_rate": "mean",
        "concealment_rate": "mean",
    }).round(4)
    print(summary.to_string())

    return df, df_diag


def generate_synthetic_data(data_dir: str):
    """Generate simple synthetic test signals if no real data available."""
    os.makedirs(data_dir, exist_ok=True)
    sr = TARGET_SR

    # 1. Onset-heavy: silence-burst-silence-burst pattern
    t = np.linspace(0, 3.0, int(3.0 * sr), endpoint=False)
    onset_heavy = np.zeros_like(t)
    # Several speech-like bursts
    for start, dur, freq in [(0.2, 0.3, 200), (0.8, 0.2, 300),
                              (1.3, 0.4, 150), (2.0, 0.25, 250),
                              (2.5, 0.3, 180)]:
        idx_start = int(start * sr)
        idx_end = int((start + dur) * sr)
        n = idx_end - idx_start
        envelope = np.hanning(n)
        signal = envelope * np.sin(2 * np.pi * freq * np.linspace(0, dur, n))
        # Add harmonics
        signal += 0.5 * envelope * np.sin(2 * np.pi * freq * 2 * np.linspace(0, dur, n))
        signal += 0.3 * envelope * np.sin(2 * np.pi * freq * 3 * np.linspace(0, dur, n))
        onset_heavy[idx_start:idx_end] = signal
    onset_heavy *= 0.9 / (np.max(np.abs(onset_heavy)) + 1e-8)
    sf.write(os.path.join(data_dir, "synthetic_onset_heavy.wav"), onset_heavy, sr)

    # 2. General: continuous voiced-like signal with slow modulation
    t2 = np.linspace(0, 3.0, int(3.0 * sr), endpoint=False)
    f0 = 150 + 30 * np.sin(2 * np.pi * 0.5 * t2)  # slowly varying pitch
    phase = np.cumsum(2 * np.pi * f0 / sr)
    general = np.sin(phase) + 0.4 * np.sin(2 * phase) + 0.2 * np.sin(3 * phase)
    # Add amplitude modulation
    general *= 0.5 + 0.3 * np.sin(2 * np.pi * 2.0 * t2)
    general *= 0.9 / (np.max(np.abs(general)) + 1e-8)
    sf.write(os.path.join(data_dir, "synthetic_general.wav"), general, sr)

    # 3. Mixed: combination with noise bursts
    mixed = np.zeros(int(3.0 * sr))
    # Voiced segment
    t3 = np.linspace(0, 1.5, int(1.5 * sr), endpoint=False)
    voiced = np.sin(2 * np.pi * 180 * t3) * np.hanning(len(t3))
    mixed[:len(t3)] = voiced
    # Noise burst (unvoiced)
    noise_len = int(0.5 * sr)
    noise = np.random.default_rng(42).normal(0, 0.3, noise_len) * np.hanning(noise_len)
    mixed[int(1.6 * sr):int(1.6 * sr) + noise_len] = noise
    # Another voiced
    t4 = np.linspace(0, 0.8, int(0.8 * sr), endpoint=False)
    mixed[int(2.2 * sr):int(2.2 * sr) + len(t4)] = np.sin(2 * np.pi * 220 * t4) * np.hanning(len(t4))
    mixed *= 0.9 / (np.max(np.abs(mixed)) + 1e-8)
    sf.write(os.path.join(data_dir, "synthetic_mixed.wav"), mixed, sr)

    print(f"Generated 3 synthetic audio files in {data_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Importance-aware frame protection PoC")
    parser.add_argument("--data-dir", default="poc/data/audio",
                        help="Directory with audio files")
    parser.add_argument("--results-dir", default="results",
                        help="Output directory for results")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic test data if no audio files exist")
    args = parser.parse_args()

    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.data_dir,
    )
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.results_dir,
    )

    # Generate synthetic data if directory is empty
    if args.synthetic or not os.path.exists(data_dir) or not os.listdir(data_dir):
        print("No audio data found. Generating synthetic test signals...")
        generate_synthetic_data(data_dir)

    run_full_matrix(data_dir, results_dir)
