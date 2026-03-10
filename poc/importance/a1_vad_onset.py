"""A1: VAD + talkspurt onset boost importance scorer."""
import numpy as np


def compute_vad_energy(pcm: np.ndarray, frame_size: int, threshold_db: float = -40.0) -> np.ndarray:
    """Simple energy-based VAD. Returns boolean array (n_frames,)."""
    n_frames = len(pcm) // frame_size
    vad = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        frame = pcm[i * frame_size : (i + 1) * frame_size]
        energy = np.mean(frame ** 2)
        energy_db = 10 * np.log10(energy + 1e-10)
        vad[i] = energy_db > threshold_db
    return vad


def score_a1(pcm: np.ndarray, frame_size: int, onset_boost_frames: int = 3,
             vad_threshold_db: float = -40.0) -> np.ndarray:
    """Compute A1 importance: boost first M frames after silence->speech onset.

    Returns importance in [0, 1] for each frame.
    """
    vad = compute_vad_energy(pcm, frame_size, vad_threshold_db)
    n_frames = len(vad)
    importance = np.full(n_frames, 0.2)

    frames_since_onset = float("inf")
    for i in range(n_frames):
        if vad[i] and (i == 0 or not vad[i - 1]):
            frames_since_onset = 0
        if vad[i] and frames_since_onset < onset_boost_frames:
            importance[i] = 1.0
            frames_since_onset += 1
        elif not vad[i]:
            importance[i] = 0.1
            frames_since_onset = float("inf")

    return importance
