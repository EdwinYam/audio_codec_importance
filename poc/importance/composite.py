"""Composite importance scorer: weighted combination of A1 + A2 + B1 + A4."""
import numpy as np

from .a1_vad_onset import score_a1
from .b1_token_novelty import score_b1
from .a2_spectral_flux import score_a2
from .a4_evs_criticality import score_a4


def score_composite(
    pcm: np.ndarray,
    tokens: np.ndarray,
    frame_size: int,
    sr: int,
    loss_mask: np.ndarray = None,
    w1: float = 0.25,
    w2: float = 0.25,
    w3: float = 0.25,
    w4: float = 0.25,
) -> np.ndarray:
    """Compute composite importance: I[n] = w1*A1 + w2*A2 + w3*B1 + w4*A4.

    Returns importance in [0, 1] for each frame.
    """
    n_frames = min(len(pcm) // frame_size, tokens.shape[0])
    pcm_trimmed = pcm[: n_frames * frame_size]

    i_a1 = score_a1(pcm_trimmed, frame_size)[:n_frames]
    i_a2 = score_a2(pcm_trimmed, frame_size, sr)[:n_frames]
    i_b1 = score_b1(tokens[:n_frames])
    i_a4 = score_a4(pcm_trimmed, frame_size, sr, loss_mask)[:n_frames]

    composite = w1 * i_a1 + w2 * i_a2 + w3 * i_b1 + w4 * i_a4
    return np.clip(composite, 0.0, 1.0)


def score_individual(
    pcm: np.ndarray,
    tokens: np.ndarray,
    frame_size: int,
    sr: int,
    loss_mask: np.ndarray = None,
) -> dict:
    """Return all individual scores plus composite, for diagnostics."""
    n_frames = min(len(pcm) // frame_size, tokens.shape[0])
    pcm_trimmed = pcm[: n_frames * frame_size]

    return {
        "A1": score_a1(pcm_trimmed, frame_size)[:n_frames],
        "A2": score_a2(pcm_trimmed, frame_size, sr)[:n_frames],
        "B1": score_b1(tokens[:n_frames]),
        "A4": score_a4(pcm_trimmed, frame_size, sr, loss_mask)[:n_frames],
        "composite": score_composite(pcm_trimmed, tokens[:n_frames], frame_size, sr, loss_mask),
    }
