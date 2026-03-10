"""Jitter-based late-arrival discard model."""
import numpy as np


def apply_jitter_discard(
    n_frames: int,
    jitter_ms_std: float = 10.0,
    buffer_depth_ms: float = 40.0,
    frame_duration_ms: float = 13.3,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Simulate late-arrival discard due to jitter exceeding buffer depth.

    Each frame gets a random delay; if delay > buffer_depth, frame is discarded.

    Returns mask: True = frame arrived on time, False = discarded (late).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Random jitter (half-normal: always positive delay variation)
    jitter = np.abs(rng.normal(0, jitter_ms_std, n_frames))
    mask = jitter <= buffer_depth_ms
    return mask
