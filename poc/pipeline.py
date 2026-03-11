"""Main experiment pipeline: encode → impair → protect → decode → eval."""
import numpy as np
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
    codec,
    network_type: str,
    plr: float,
    protection_method: str,
    budget_frac: float = 0.1,
    seed: int = 42,
    tokens: np.ndarray = None,
    importance_scores: np.ndarray = None,
    concealment: str = "zero_fill",
) -> dict:
    """Run one experiment condition.

    Args:
        pcm: clean PCM audio, float32 [-1, 1]
        codec: initialized codec wrapper (CodecInterface)
        network_type: "random_loss", "burst_loss", "jitter_discard"
        plr: packet loss rate (0-1)
        protection_method: "none", "random", "heuristic", "importance_aware",
                          "importance_selective"
        budget_frac: fraction of frames to protect (e.g. 0.1 = 10%)
        seed: random seed
        tokens: pre-encoded tokens to avoid re-encoding (optional)
        importance_scores: pre-computed importance scores (optional)

    Returns:
        dict with all metrics
    """
    rng = np.random.default_rng(seed)

    # 1. Encode (use cached tokens if provided)
    if tokens is None:
        tokens = codec.encode(pcm)
    n_frames = tokens.shape[0]
    budget = max(1, int(n_frames * budget_frac))

    # 2. Compute importance scores (use cached if provided)
    if importance_scores is None:
        importance_scores = score_composite(
            pcm, tokens, codec.frame_size, codec.sample_rate
        )

    # 3. Select protected frames based on method
    if protection_method == "importance_selective":
        # Special mode: loss only applies to non-important frames.
        # Important frames (top-K by importance) are completely immune.
        # Loss is concentrated on non-important frames so the overall
        # effective loss count stays comparable to uniform PLR.
        important_indices = np.argsort(importance_scores)[::-1][:budget]
        important_mask = np.zeros(n_frames, dtype=bool)
        important_mask[important_indices] = True
        n_nonimportant = n_frames - budget
        # Adjust PLR so total expected lost frames ≈ plr * n_frames
        adjusted_plr = min(plr * n_frames / max(n_nonimportant, 1), 1.0)
        # Generate loss only for non-important frames
        raw_mask = np.ones(n_frames, dtype=bool)  # all received
        if n_nonimportant > 0:
            nonimportant_loss = apply_random_loss(n_nonimportant, adjusted_plr, rng)
            raw_mask[~important_mask] = nonimportant_loss
        final_mask = raw_mask  # no further repair needed
        protected = np.sort(important_indices)
    else:
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
            jitter_std = plr * 200
            raw_mask = apply_jitter_discard(
                n_frames, jitter_ms_std=jitter_std, buffer_depth_ms=40.0, rng=rng
            )
        else:
            raise ValueError(f"Unknown network type: {network_type}")

        # 5. Apply protection (repair lost frames that were protected)
        final_mask = apply_protection(raw_mask, protected)

    # 6. Decode with loss (apply concealment method)
    pcm_degraded = codec.decode_with_mask(tokens, final_mask, concealment=concealment)

    # 7. Evaluate
    min_len = min(len(pcm), len(pcm_degraded))
    pcm_ref = pcm[:min_len]
    pcm_deg = pcm_degraded[:min_len]

    quality = compute_all_quality(pcm_ref, pcm_deg, codec.sample_rate)
    conc_stats = concealment_stats(final_mask)

    # Effective loss rate after protection
    raw_loss_rate = 1.0 - raw_mask.mean()
    final_loss_rate = 1.0 - final_mask.mean()

    return {
        "network_type": network_type,
        "target_plr": plr,
        "protection_method": protection_method,
        "concealment": concealment,
        "budget_frac": budget_frac,
        "n_frames": n_frames,
        "n_protected": len(protected),
        "raw_loss_rate": float(raw_loss_rate),
        "post_repair_loss_rate": float(final_loss_rate),
        **quality,
        **conc_stats,
    }
