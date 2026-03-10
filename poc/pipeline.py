"""Main experiment pipeline: encode → impair → protect → decode → eval."""
import numpy as np
from poc.codec.encodec_wrapper import EnCodecWrapper
from poc.importance.composite import score_composite
from poc.importance.a1_vad_onset import score_a1
from poc.network.random_loss import apply_random_loss
from poc.network.burst_loss import apply_burst_loss
from poc.network.jitter_discard import apply_jitter_discard
from poc.protection.no_protection import select_no_protection
from poc.protection.random_protection import select_random_protection
from poc.protection.heuristic_protection import select_heuristic_protection
from poc.protection.importance_aware import select_importance_protection
from poc.eval.audio_quality import compute_all_quality
from poc.eval.diagnostics import concealment_stats


def apply_protection(mask: np.ndarray, protected_indices: np.ndarray) -> np.ndarray:
    """Apply protection: protected frames cannot be lost (simulates redundancy)."""
    repaired = mask.copy()
    if len(protected_indices) > 0:
        repaired[protected_indices] = True
    return repaired


def run_single_experiment(
    pcm: np.ndarray,
    codec: EnCodecWrapper,
    network_type: str,
    plr: float,
    protection_method: str,
    budget_frac: float = 0.1,
    seed: int = 42,
) -> dict:
    """Run one experiment condition.

    Args:
        pcm: clean PCM audio, float32 [-1, 1]
        codec: initialized codec wrapper
        network_type: "random_loss", "burst_loss", "jitter_discard"
        plr: packet loss rate (0-1)
        protection_method: "none", "random", "heuristic", "importance_aware"
        budget_frac: fraction of frames to protect (e.g. 0.1 = 10%)
        seed: random seed

    Returns:
        dict with all metrics
    """
    rng = np.random.default_rng(seed)

    # 1. Encode
    tokens = codec.encode(pcm)
    n_frames = tokens.shape[0]
    budget = max(1, int(n_frames * budget_frac))

    # 2. Compute importance scores (needed for importance_aware method)
    importance_scores = score_composite(
        pcm, tokens, codec.frame_size, codec.sample_rate
    )

    # 3. Select protected frames based on method
    if protection_method == "none":
        protected = select_no_protection(n_frames, budget)
    elif protection_method == "random":
        protected = select_random_protection(n_frames, budget, rng)
    elif protection_method == "heuristic":
        protected = select_heuristic_protection(
            pcm, codec.frame_size, n_frames, budget
        )
    elif protection_method == "importance_aware":
        protected = select_importance_protection(importance_scores, budget)
    else:
        raise ValueError(f"Unknown protection method: {protection_method}")

    # 4. Apply network impairment
    if network_type == "random_loss":
        raw_mask = apply_random_loss(n_frames, plr, rng)
    elif network_type == "burst_loss":
        raw_mask = apply_burst_loss(n_frames, plr, burst_ratio=2.0, rng=rng)
    elif network_type == "jitter_discard":
        # Map PLR to jitter std: higher PLR -> higher jitter
        jitter_std = plr * 200  # e.g. 5% -> 10ms std
        raw_mask = apply_jitter_discard(
            n_frames, jitter_ms_std=jitter_std, buffer_depth_ms=40.0, rng=rng
        )
    else:
        raise ValueError(f"Unknown network type: {network_type}")

    # 5. Apply protection (repair lost frames that were protected)
    final_mask = apply_protection(raw_mask, protected)

    # 6. Decode with loss
    pcm_degraded = codec.decode_with_mask(tokens, final_mask)

    # 7. Also decode clean (no loss) as codec baseline
    pcm_clean_decoded = codec.decode(tokens)

    # 8. Evaluate
    min_len = min(len(pcm), len(pcm_degraded), len(pcm_clean_decoded))
    pcm_ref = pcm[:min_len]
    pcm_deg = pcm_degraded[:min_len]

    quality = compute_all_quality(pcm_ref, pcm_deg, codec.sample_rate)
    concealment = concealment_stats(final_mask)

    # Effective loss rate after protection
    raw_loss_rate = 1.0 - raw_mask.mean()
    final_loss_rate = 1.0 - final_mask.mean()

    return {
        "network_type": network_type,
        "target_plr": plr,
        "protection_method": protection_method,
        "budget_frac": budget_frac,
        "n_frames": n_frames,
        "n_protected": len(protected),
        "raw_loss_rate": float(raw_loss_rate),
        "post_repair_loss_rate": float(final_loss_rate),
        **quality,
        **concealment,
    }
