"""Oracle baseline: compute ground-truth frame importance via leave-one-out."""
import numpy as np
from poc.eval.audio_quality import compute_si_sdr


def compute_oracle_importance(
    codec, pcm: np.ndarray, tokens: np.ndarray, max_samples: int = 50
) -> np.ndarray:
    """For each frame, measure SI-SDR drop when that single frame is lost.

    To speed up: if n_frames > max_samples, randomly sample frames and
    interpolate the rest with a local-energy proxy.

    Returns array of per-frame damage scores (higher = more important).
    """
    n_frames = tokens.shape[0]
    full_mask = np.ones(n_frames, dtype=bool)
    pcm_full = codec.decode_with_mask(tokens, full_mask)
    min_len = min(len(pcm), len(pcm_full))
    baseline_sdr = compute_si_sdr(pcm[:min_len], pcm_full[:min_len])

    if n_frames <= max_samples:
        sample_indices = np.arange(n_frames)
    else:
        rng = np.random.default_rng(0)
        sample_indices = np.sort(rng.choice(n_frames, size=max_samples, replace=False))

    sampled_damage = np.zeros(len(sample_indices))
    for idx, i in enumerate(sample_indices):
        mask = full_mask.copy()
        mask[i] = False
        pcm_degraded = codec.decode_with_mask(tokens, mask)
        min_len_i = min(len(pcm), len(pcm_degraded))
        sdr_i = compute_si_sdr(pcm[:min_len_i], pcm_degraded[:min_len_i])
        sampled_damage[idx] = max(baseline_sdr - sdr_i, 0.0)

    # Interpolate for non-sampled frames
    if n_frames <= max_samples:
        return sampled_damage

    damage = np.interp(np.arange(n_frames), sample_indices, sampled_damage)
    return damage
