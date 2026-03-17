"""Core streaming simulation: causal, frame-by-frame duplication with packet loss."""
import numpy as np
from collections import deque
from poc.streaming.threshold_methods import should_duplicate, parse_method_config
from poc.network.random_loss import apply_random_loss
from poc.codec.concealment import apply_concealment
from poc.eval.audio_quality import compute_all_quality
from poc.eval.diagnostics import concealment_stats


def simulate_streaming(
    tokens: np.ndarray,
    importance_scores: np.ndarray,
    threshold_method: str,
    plr: float,
    duplication_delay: int = 1,
    seed: int = 42,
) -> dict:
    """Simulate streaming transmission with importance-aware duplication.

    Args:
        tokens: (n_frames, n_codebooks) encoded token indices
        importance_scores: (n_frames,) per-frame importance scores
        threshold_method: name of threshold strategy
        plr: packet loss rate for Bernoulli loss model
        duplication_delay: D - duplicate of frame t rides with slot t+D
        seed: random seed

    Returns:
        dict with reconstructed_tokens, receive_mask, and recovery stats
    """
    n_frames, n_codebooks = tokens.shape
    rng = np.random.default_rng(seed)
    config = parse_method_config(threshold_method)

    # --- Sender side: decide which frames to duplicate ---
    duplicated = np.zeros(n_frames, dtype=bool)
    history = deque(maxlen=200)  # sliding window for adaptive methods

    for t in range(n_frames):
        score = importance_scores[t]
        thresh = config.get("threshold", 0.5)

        if config["method"].startswith("adaptive_quantile"):
            dup = should_duplicate(score, config["method"], history,
                                   threshold=thresh)
        else:
            dup = should_duplicate(score, config["method"], history,
                                   threshold=thresh)
        duplicated[t] = dup
        history.append(score)

    # --- Network: apply Bernoulli loss to each slot ---
    slot_received = apply_random_loss(n_frames, plr, rng)

    # --- Receiver side: reconstruct ---
    reconstructed = tokens.copy()
    frame_available = np.ones(n_frames, dtype=bool)  # tracks final availability

    # Track recovery stats
    n_recovered = 0
    n_concealed = 0
    last_received_frame = None

    for t in range(n_frames):
        if slot_received[t]:
            # Primary frame t arrived
            frame_available[t] = True
            last_received_frame = t
        else:
            # Slot t lost. Check if frame t was duplicated and its duplicate arrived.
            dup_slot = t + duplication_delay
            if duplicated[t] and dup_slot < n_frames and slot_received[dup_slot]:
                # Recover frame t from its duplicate piggybacked on slot t+D
                frame_available[t] = True
                n_recovered += 1
            else:
                # Frame t is unrecoverable - will be concealed
                frame_available[t] = False
                n_concealed += 1

    # Apply neighbor_copy concealment for unrecoverable frames
    reconstructed = apply_concealment(tokens, frame_available, method="neighbor_copy")

    # --- Compute stats ---
    n_lost_slots = int(np.sum(~slot_received))
    raw_loss_rate = 1.0 - slot_received.mean()
    n_duplicated = int(np.sum(duplicated))
    duplication_rate = n_duplicated / n_frames if n_frames > 0 else 0.0
    n_lost_frames_final = int(np.sum(~frame_available))
    post_recovery_loss_rate = n_lost_frames_final / n_frames if n_frames > 0 else 0.0
    recovery_rate = n_recovered / max(n_lost_slots, 1) if n_lost_slots > 0 else 0.0

    return {
        "reconstructed_tokens": reconstructed,
        "receive_mask": frame_available,
        "slot_received": slot_received,
        "duplicated": duplicated,
        "n_frames": n_frames,
        "n_duplicated": n_duplicated,
        "duplication_rate": duplication_rate,
        "raw_loss_rate": float(raw_loss_rate),
        "n_recovered": n_recovered,
        "recovery_rate": recovery_rate,
        "n_concealed": n_concealed,
        "n_lost_frames": n_lost_frames_final,
        "post_recovery_loss_rate": post_recovery_loss_rate,
    }


def run_streaming_trial(
    pcm: np.ndarray,
    codec,
    tokens: np.ndarray,
    importance_scores: np.ndarray,
    threshold_method: str,
    plr: float,
    duplication_delay: int = 1,
    seed: int = 42,
    base_bitrate_kbps: float = 3.0,
) -> dict:
    """Run one full streaming trial: simulate → decode → evaluate.

    Args:
        pcm: original PCM audio (float32)
        codec: initialized codec wrapper
        tokens: pre-encoded tokens (n_frames, n_codebooks)
        importance_scores: pre-computed importance scores (n_frames,)
        threshold_method: threshold strategy name
        plr: packet loss rate
        duplication_delay: D value
        seed: random seed
        base_bitrate_kbps: base codec bitrate

    Returns:
        dict with all metrics for one trial
    """
    # Run streaming simulation
    sim = simulate_streaming(
        tokens, importance_scores, threshold_method, plr,
        duplication_delay=duplication_delay, seed=seed,
    )

    # Decode reconstructed tokens
    pcm_degraded = codec.decode(sim["reconstructed_tokens"])

    # Evaluate quality
    min_len = min(len(pcm), len(pcm_degraded))
    quality = compute_all_quality(pcm[:min_len], pcm_degraded[:min_len], codec.sample_rate)

    # Concealment stats from final mask
    conc_stats = concealment_stats(sim["receive_mask"])

    # Bandwidth overhead
    total_bitrate_kbps = base_bitrate_kbps * (1.0 + sim["duplication_rate"])
    bandwidth_overhead_pct = sim["duplication_rate"] * 100.0

    # Determine threshold value for reporting
    config = parse_method_config(threshold_method)
    if "threshold" in config:
        threshold_value = str(config["threshold"])
    elif threshold_method in ("no_duplicate", "all_duplicate"):
        threshold_value = threshold_method
    else:
        threshold_value = "dynamic"

    return {
        "network_type": "random_loss",
        "target_plr": plr,
        "threshold_method": threshold_method,
        "threshold_value": threshold_value,
        "duplication_delay": duplication_delay,
        "n_frames": sim["n_frames"],
        "n_duplicated": sim["n_duplicated"],
        "duplication_rate": sim["duplication_rate"],
        "base_bitrate_kbps": base_bitrate_kbps,
        "total_bitrate_kbps": total_bitrate_kbps,
        "bandwidth_overhead_pct": bandwidth_overhead_pct,
        "raw_loss_rate": sim["raw_loss_rate"],
        "n_recovered": sim["n_recovered"],
        "recovery_rate": sim["recovery_rate"],
        "n_concealed": sim["n_concealed"],
        "post_recovery_loss_rate": sim["post_recovery_loss_rate"],
        **quality,
        **conc_stats,
    }
