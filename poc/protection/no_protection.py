"""Baseline 1: No protection — no frames are protected."""
import numpy as np


def select_no_protection(n_frames: int, budget: int) -> np.ndarray:
    """Returns empty set of protected frame indices."""
    return np.array([], dtype=int)
