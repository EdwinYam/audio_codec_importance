"""Baseline 2: Random protection — randomly select frames to protect."""
import numpy as np


def select_random_protection(
    n_frames: int, budget: int, rng: np.random.Generator = None
) -> np.ndarray:
    """Randomly select `budget` frames to protect.

    Returns sorted array of protected frame indices.
    """
    if rng is None:
        rng = np.random.default_rng()
    budget = min(budget, n_frames)
    return np.sort(rng.choice(n_frames, size=budget, replace=False))
