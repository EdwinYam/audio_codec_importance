"""B1: Token change-rate / novelty importance scorer."""
import numpy as np


def score_b1(tokens: np.ndarray, tau: float = 0.5) -> np.ndarray:
    """Compute B1 importance based on token change-rate between consecutive frames.

    Args:
        tokens: shape (n_frames, n_codebooks), integer token indices
        tau: normalization threshold

    Returns:
        importance in [0, 1] for each frame
    """
    n_frames, n_codebooks = tokens.shape
    importance = np.zeros(n_frames)

    for i in range(1, n_frames):
        changed = np.sum(tokens[i] != tokens[i - 1])
        change_rate = changed / n_codebooks
        importance[i] = np.clip(change_rate / tau, 0.0, 1.0)

    # First frame: max importance (no history)
    importance[0] = 1.0
    return importance
