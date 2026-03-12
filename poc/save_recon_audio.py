"""Save reconstructed audio files for HILCodec under random loss.

Generates WAV files for importance_selective and none protection methods,
across all concealment methods and target PLRs, for all audio samples.

Output structure:
  recon_audio/
    hilcodec/
      <protection_method>/
        <concealment>/
          plr_<XX>/
            <sample_name>.wav
"""
import os
import sys
import time
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from poc.codec.hilcodec_wrapper import HILCodecWrapper
from poc.pipeline import run_single_experiment
from poc.importance.composite import score_composite
from poc.cache import load_cached_tokens, cache_tokens
from poc.run_experiment import load_and_resample, load_audio_files, TARGET_SR, BUDGET_FRAC

# ─── Configuration ──────────────────────────────────────────────
PLRS = [0.0, 0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.40]
PROTECTION_METHODS = ["none", "importance_selective"]
CONCEALMENTS = ["zero_fill", "neighbor_copy"]
SEEDS = [42, 123]
NETWORK = "random_loss"


def save_recon_audio(data_dir: str, output_dir: str):
    """Generate and save all reconstructed audio files."""
    os.makedirs(output_dir, exist_ok=True)

    audio_files = load_audio_files(data_dir)
    if not audio_files:
        print(f"No audio files found in {data_dir}")
        return

    print(f"Found {len(audio_files)} audio files")
    print(f"Methods: {PROTECTION_METHODS}")
    print(f"Concealments: {CONCEALMENTS}")
    print(f"PLRs: {PLRS}")
    print(f"Seeds: {SEEDS}")

    codec_name = "HILCodec_3kbps"
    codec = HILCodecWrapper(n_quantizers=4)
    print(f"Loaded {codec_name}: SR={codec.sample_rate}, frame_size={codec.frame_size}")

    # Save originals
    orig_dir = os.path.join(output_dir, "hilcodec", "original")
    os.makedirs(orig_dir, exist_ok=True)

    total_files = (
        len(audio_files) * len(PROTECTION_METHODS) * len(CONCEALMENTS)
        * len(PLRS) * len(SEEDS)
    )
    count = 0

    for audio_idx, audio_path in enumerate(audio_files):
        fname = os.path.splitext(os.path.basename(audio_path))[0]
        print(f"\n[{audio_idx+1}/{len(audio_files)}] Processing: {fname}")

        pcm = load_and_resample(audio_path)
        print(f"  Audio: {len(pcm)} samples, {len(pcm)/TARGET_SR:.2f}s")

        # Save original
        sf.write(os.path.join(orig_dir, f"{fname}.wav"), pcm, TARGET_SR)

        # Encode
        cached_tokens = load_cached_tokens(codec_name, audio_path)
        tokens = codec.encode(pcm)
        if cached_tokens is not None and cached_tokens.shape == tokens.shape:
            tokens = cached_tokens
        else:
            cache_tokens(codec_name, audio_path, tokens)
        n_frames = tokens.shape[0]
        print(f"  Encoded: {n_frames} frames")

        # Compute importance scores
        importance_scores = score_composite(
            pcm, tokens, codec.frame_size, codec.sample_rate
        )

        # Save clean reconstruction (no loss)
        clean_dir = os.path.join(output_dir, "hilcodec", "clean_recon")
        os.makedirs(clean_dir, exist_ok=True)
        pcm_clean = codec.decode(tokens)
        min_len = min(len(pcm), len(pcm_clean))
        sf.write(os.path.join(clean_dir, f"{fname}.wav"), pcm_clean[:min_len], TARGET_SR)

        # Run all conditions
        for method in PROTECTION_METHODS:
            for conc in CONCEALMENTS:
                for plr in PLRS:
                    for seed in SEEDS:
                        # Build output path
                        plr_str = f"plr_{int(plr*100):02d}"
                        out_dir = os.path.join(
                            output_dir, "hilcodec", method, conc, plr_str
                        )
                        os.makedirs(out_dir, exist_ok=True)
                        out_path = os.path.join(out_dir, f"{fname}_seed{seed}.wav")

                        # Run experiment to get degraded audio
                        rng = np.random.default_rng(seed)
                        result = run_single_experiment(
                            pcm, codec, NETWORK, plr, method,
                            budget_frac=BUDGET_FRAC, seed=seed,
                            tokens=tokens,
                            importance_scores=importance_scores,
                            concealment=conc,
                        )

                        # Reconstruct audio (need to re-run decode since
                        # run_single_experiment doesn't return audio)
                        pcm_degraded = _reconstruct(
                            pcm, codec, tokens, importance_scores,
                            NETWORK, plr, method, conc, seed
                        )
                        min_len = min(len(pcm), len(pcm_degraded))
                        sf.write(out_path, pcm_degraded[:min_len], TARGET_SR)

                        count += 1
                        if count % 50 == 0:
                            print(f"  Saved {count}/{total_files} files...")

    print(f"\nDone! Saved {count} reconstructed audio files to {output_dir}/hilcodec/")
    print(f"\nFolder structure:")
    print(f"  {output_dir}/hilcodec/original/          - original audio")
    print(f"  {output_dir}/hilcodec/clean_recon/       - clean codec reconstruction")
    print(f"  {output_dir}/hilcodec/<method>/<concealment>/plr_XX/ - degraded audio")


def _reconstruct(pcm, codec, tokens, importance_scores,
                 network_type, plr, protection_method, concealment, seed):
    """Reconstruct degraded audio (mirrors pipeline logic)."""
    from poc.network.random_loss import apply_random_loss
    from poc.protection.no_protection import select_no_protection

    rng = np.random.default_rng(seed)
    n_frames = tokens.shape[0]
    budget = max(1, int(n_frames * BUDGET_FRAC))

    if protection_method == "importance_selective":
        important_indices = np.argsort(importance_scores)[::-1][:budget]
        important_mask = np.zeros(n_frames, dtype=bool)
        important_mask[important_indices] = True
        n_nonimportant = n_frames - budget
        adjusted_plr = min(plr * n_frames / max(n_nonimportant, 1), 1.0)
        raw_mask = np.ones(n_frames, dtype=bool)
        if n_nonimportant > 0:
            nonimportant_loss = apply_random_loss(n_nonimportant, adjusted_plr, rng)
            raw_mask[~important_mask] = nonimportant_loss
        final_mask = raw_mask
    else:  # "none"
        protected = select_no_protection(n_frames, budget)
        raw_mask = apply_random_loss(n_frames, plr, rng)
        from poc.pipeline import apply_protection
        final_mask = apply_protection(raw_mask, protected)

    pcm_degraded = codec.decode_with_mask(tokens, final_mask, concealment=concealment)
    return pcm_degraded


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Save reconstructed HILCodec audio under random loss"
    )
    parser.add_argument("--data-dir", default="poc/data/audio",
                        help="Directory with audio files")
    parser.add_argument("--output-dir", default="results/recon_audio",
                        help="Output directory for reconstructed audio")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, args.data_dir)
    output_dir = os.path.join(project_root, args.output_dir)

    save_recon_audio(data_dir, output_dir)
