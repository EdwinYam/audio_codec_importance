"""A2: Transient / spectral flux importance scorer."""
import numpy as np
from scipy.signal import stft


def compute_spectral_flux(pcm: np.ndarray, frame_size: int, sr: int) -> np.ndarray:
    """Compute spectral flux per frame."""
    n_frames = len(pcm) // frame_size
    nfft = min(frame_size * 2, 512)
    flux = np.zeros(n_frames)

    prev_spec = None
    for i in range(n_frames):
        frame = pcm[i * frame_size : (i + 1) * frame_size]
        spec = np.abs(np.fft.rfft(frame * np.hanning(len(frame)), n=nfft))
        if prev_spec is not None:
            flux[i] = np.sum((spec - prev_spec) ** 2)
        prev_spec = spec

    return flux


def compute_energy(pcm: np.ndarray, frame_size: int) -> np.ndarray:
    """Compute short-time energy per frame."""
    n_frames = len(pcm) // frame_size
    energy = np.zeros(n_frames)
    for i in range(n_frames):
        frame = pcm[i * frame_size : (i + 1) * frame_size]
        energy[i] = np.mean(frame ** 2)
    return energy


def estimate_voicing(pcm: np.ndarray, frame_size: int, sr: int) -> np.ndarray:
    """Simple voicing estimate via autocorrelation peak ratio."""
    n_frames = len(pcm) // frame_size
    voicing = np.zeros(n_frames)
    min_lag = sr // 500  # max F0 = 500 Hz
    max_lag = sr // 60   # min F0 = 60 Hz

    for i in range(n_frames):
        frame = pcm[i * frame_size : (i + 1) * frame_size]
        if np.max(np.abs(frame)) < 1e-6:
            continue
        frame = frame - np.mean(frame)
        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(frame) - 1 :]  # positive lags
        if corr[0] == 0:
            continue
        corr = corr / corr[0]
        search_range = corr[min_lag : min(max_lag, len(corr))]
        if len(search_range) > 0:
            voicing[i] = np.max(search_range)

    return voicing


def _normalize(x: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    rng = x.max() - x.min()
    if rng < 1e-10:
        return np.zeros_like(x)
    return (x - x.min()) / rng


def score_a2(pcm: np.ndarray, frame_size: int, sr: int,
             a: float = 0.5, b: float = 0.3, c: float = 0.2) -> np.ndarray:
    """Compute A2 importance from spectral flux, energy delta, and voicing.

    I_A2[n] = clamp(a * norm(SF[n]) + b * norm(|E[n]-E[n-1]|) + c * (1 - V[n]), 0, 1)
    """
    sf = compute_spectral_flux(pcm, frame_size, sr)
    energy = compute_energy(pcm, frame_size)
    voicing = estimate_voicing(pcm, frame_size, sr)

    energy_delta = np.zeros_like(energy)
    energy_delta[1:] = np.abs(energy[1:] - energy[:-1])

    importance = (
        a * _normalize(sf)
        + b * _normalize(energy_delta)
        + c * (1.0 - voicing)
    )
    return np.clip(importance, 0.0, 1.0)
