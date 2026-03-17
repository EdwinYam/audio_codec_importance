"""CLI entry point: run the streaming duplication experiment matrix."""
import os
import sys
import time
import numpy as np
import pandas as pd
import soundfile as sf
import librosa

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from poc.codec.hilcodec_wrapper import HILCodecWrapper
from poc.importance.composite import score_composite
from poc.cache import cache_tokens, load_cached_tokens, cache_importance, load_cached_importance
from poc.streaming.pipeline_streaming import run_streaming_trial

# ─── Configuration ───────────────────────────────────────────────
TARGET_SR = 24000
MAX_DURATION_SEC = 4.0
BASE_BITRATE_KBPS = 3.0

PLRS = [0.0, 0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.40]
SEEDS = [42, 123]
DUPLICATION_DELAYS = [1, 2]
THRESHOLD_METHODS = [
    "no_duplicate",
    "fixed_0.3",
    "fixed_0.5",
    "fixed_0.7",
    "adaptive_mean_std",
    "adaptive_quantile_70",
    "all_duplicate",
]


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
        pcm = pcm[:, 0]
    if sr != target_sr:
        pcm = librosa.resample(pcm, orig_sr=sr, target_sr=target_sr)
    max_samples = int(MAX_DURATION_SEC * target_sr)
    pcm = pcm[:max_samples]
    peak = np.max(np.abs(pcm))
    if peak > 0:
        pcm = pcm / peak * 0.95
    return pcm


def run_streaming_matrix(data_dir: str, results_dir: str):
    """Run the full streaming experiment matrix."""
    os.makedirs(results_dir, exist_ok=True)

    audio_files = load_audio_files(data_dir)
    if not audio_files:
        print(f"No audio files found in {data_dir}")
        return

    print(f"Found {len(audio_files)} audio files")
    n_total = (len(audio_files) * len(PLRS) * len(THRESHOLD_METHODS)
               * len(DUPLICATION_DELAYS) * len(SEEDS))
    print(f"Total trials: {n_total}")

    codec_name = "HILCodec_3kbps"
    print(f"\nLoading codec: {codec_name}")
    codec = HILCodecWrapper(n_quantizers=4)
    print(f"  SR={codec.sample_rate}, frame_size={codec.frame_size}, "
          f"codebooks={codec.n_codebooks}")

    all_results = []
    timing = {"encode": 0.0, "importance": 0.0, "trials": 0.0}
    n_done = 0

    for audio_idx, audio_path in enumerate(audio_files):
        fname = os.path.basename(audio_path)
        print(f"\n[{audio_idx+1}/{len(audio_files)}] Processing: {fname}")

        pcm = load_and_resample(audio_path)
        print(f"  Audio: {len(pcm)} samples, {len(pcm)/TARGET_SR:.2f}s")

        # Encode (with caching)
        t0 = time.time()
        cached = load_cached_tokens(codec_name, audio_path)
        tokens = codec.encode(pcm)
        if cached is not None and cached.shape == tokens.shape:
            tokens = cached
            print(f"  Tokens loaded from cache")
        else:
            cache_tokens(codec_name, audio_path, tokens)
        n_frames = tokens.shape[0]
        t_enc = time.time() - t0
        timing["encode"] += t_enc
        print(f"  Encoded: {n_frames} frames ({t_enc:.1f}s)")

        # Importance scores (with caching)
        t0 = time.time()
        cached_imp = load_cached_importance(codec_name, audio_path)
        if cached_imp is not None and "composite" in cached_imp:
            importance_scores = cached_imp["composite"]
            print(f"  Importance scores loaded from cache")
        else:
            importance_scores = score_composite(
                pcm, tokens, codec.frame_size, codec.sample_rate
            )
        t_imp = time.time() - t0
        timing["importance"] += t_imp

        # Run trial matrix
        t0 = time.time()
        file_trials = 0
        for plr in PLRS:
            for method in THRESHOLD_METHODS:
                for delay in DUPLICATION_DELAYS:
                    for seed in SEEDS:
                        result = run_streaming_trial(
                            pcm, codec, tokens, importance_scores,
                            threshold_method=method,
                            plr=plr,
                            duplication_delay=delay,
                            seed=seed,
                            base_bitrate_kbps=BASE_BITRATE_KBPS,
                        )
                        result["codec"] = codec_name
                        result["file"] = fname
                        result["seed"] = seed
                        all_results.append(result)
                        file_trials += 1
                        n_done += 1

        t_trials = time.time() - t0
        timing["trials"] += t_trials
        print(f"  {file_trials} trials in {t_trials:.1f}s "
              f"({t_trials/max(file_trials,1)*1000:.0f}ms each)")
        print(f"  Progress: {n_done}/{n_total} ({100*n_done/n_total:.0f}%)")

    # Timing summary
    print(f"\n{'='*60}")
    print("TIMING SUMMARY")
    print(f"{'='*60}")
    print(f"  Encode:     {timing['encode']:.1f}s")
    print(f"  Importance: {timing['importance']:.1f}s")
    print(f"  Trials:     {timing['trials']:.1f}s")
    print(f"  Total:      {sum(timing.values()):.1f}s")

    # Save results
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(results_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(f"Total trials: {len(all_results)}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY (averaged over files and seeds)")
    print(f"{'='*60}")
    agg_cols = {
        "PESQ_WB": "mean", "PESQ_NB": "mean", "STOI": "mean",
        "ESTOI": "mean", "SI-SDR": "mean",
        "post_recovery_loss_rate": "mean", "duplication_rate": "mean",
        "recovery_rate": "mean", "total_bitrate_kbps": "mean",
    }
    avail_agg = {k: v for k, v in agg_cols.items() if k in df.columns}
    summary = df.groupby(
        ["target_plr", "threshold_method", "duplication_delay"]
    ).agg(avail_agg).round(4)
    print(summary.to_string())

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Streaming-compatible importance-aware duplication experiment (v3)"
    )
    parser.add_argument("--data-dir", default="poc/data/audio",
                        help="Directory with audio files")
    parser.add_argument("--results-dir", default="results_hilcodec_v3",
                        help="Output directory for results")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, args.data_dir)
    results_dir = os.path.join(project_root, args.results_dir)

    run_streaming_matrix(data_dir, results_dir)
