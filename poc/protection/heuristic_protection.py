"""Baseline 3: Simple heuristic — protect frames with highest energy / VAD onset."""
import numpy as np
from poc.importance.a1_vad_onset import score_a1


def select_heuristic_protection(
    pcm: np.ndarray, frame_size: int, n_frames: int, budget: int
) -> np.ndarray:
    """Select top `budget` frames by A1 (VAD onset) importance.

    Returns sorted array of protected frame indices.
    """
    scores = score_a1(pcm, frame_size)[:n_frames]
    budget = min(budget, n_frames)
    top_indices = np.argsort(scores)[::-1][:budget]
    return np.sort(top_indices)
