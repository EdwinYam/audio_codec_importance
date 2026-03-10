"""Proposed: Importance-aware protection — protect top-K by composite importance."""
import numpy as np


def select_importance_protection(
    importance_scores: np.ndarray, budget: int
) -> np.ndarray:
    """Select top `budget` frames by importance score.

    Returns sorted array of protected frame indices.
    """
    n_frames = len(importance_scores)
    budget = min(budget, n_frames)
    top_indices = np.argsort(importance_scores)[::-1][:budget]
    return np.sort(top_indices)
