"""Random (Bernoulli) packet loss model."""
import numpy as np


def apply_random_loss(n_frames: int, plr: float, rng: np.random.Generator = None) -> np.ndarray:
    """Generate a frame mask with independent random loss.

    Returns mask: True = frame received, False = lost.
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.random(n_frames) >= plr
