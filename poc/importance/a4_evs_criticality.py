"""A4: EVS-style frame criticality scorer."""
import numpy as np
from .a2_spectral_flux import compute_energy, estimate_voicing, compute_spectral_flux, _normalize
from .a1_vad_onset import compute_vad_energy


def classify_frames(pcm: np.ndarray, frame_size: int, sr: int):
    """Classify each frame into: transient, unstable_voiced, unvoiced, generic, stable_voiced.
    Returns class scores in [0, 1]."""
    n_frames = len(pcm) // frame_size
    voicing = estimate_voicing(pcm, frame_size, sr)
    energy = compute_energy(pcm, frame_size)
    sf = compute_spectral_flux(pcm, frame_size, sr)
    sf_norm = _normalize(sf)
    energy_delta = np.zeros_like(energy)
    energy_delta[1:] = np.abs(energy[1:] - energy[:-1])
    ed_norm = _normalize(energy_delta)

    # Frame class prior scores
    scores = np.zeros(n_frames)
    for i in range(n_frames):
        if sf_norm[i] > 0.7 and ed_norm[i] > 0.5:
            scores[i] = 1.0  # transient
        elif voicing[i] > 0.5 and sf_norm[i] > 0.3:
            scores[i] = 0.8  # unstable voiced
        elif voicing[i] < 0.3:
            scores[i] = 0.7  # unvoiced
        elif voicing[i] > 0.7 and sf_norm[i] < 0.2:
            scores[i] = 0.4  # stable voiced
        else:
            scores[i] = 0.5  # generic
    return scores


def score_transition(pcm: np.ndarray, frame_size: int, sr: int) -> np.ndarray:
    """Detect voiced/unvoiced and silence/speech transitions."""
    vad = compute_vad_energy(pcm, frame_size).astype(float)
    voicing = estimate_voicing(pcm, frame_size, sr)
    n_frames = len(vad)
    transition = np.zeros(n_frames)
    for i in range(1, n_frames):
        vad_change = abs(vad[i] - vad[i - 1])
        voicing_change = abs(voicing[i] - voicing[i - 1])
        transition[i] = min(vad_change + voicing_change, 1.0)
    return transition


def score_pitch_instability(pcm: np.ndarray, frame_size: int, sr: int) -> np.ndarray:
    """Estimate pitch instability from autocorrelation peak location changes."""
    n_frames = len(pcm) // frame_size
    min_lag = sr // 500
    max_lag = sr // 60
    pitch_lags = np.zeros(n_frames)

    for i in range(n_frames):
        frame = pcm[i * frame_size : (i + 1) * frame_size]
        frame = frame - np.mean(frame)
        if np.max(np.abs(frame)) < 1e-6:
            continue
        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(frame) - 1 :]
        search = corr[min_lag : min(max_lag, len(corr))]
        if len(search) > 0:
            pitch_lags[i] = np.argmax(search) + min_lag

    instability = np.zeros(n_frames)
    for i in range(1, n_frames):
        if pitch_lags[i] > 0 and pitch_lags[i - 1] > 0:
            instability[i] = abs(pitch_lags[i] - pitch_lags[i - 1]) / max(pitch_lags[i], 1)
    return np.clip(_normalize(instability), 0.0, 1.0)


def score_gain_jump(pcm: np.ndarray, frame_size: int) -> np.ndarray:
    """Score based on energy jump magnitude."""
    energy = compute_energy(pcm, frame_size)
    energy_db = 10 * np.log10(energy + 1e-10)
    delta = np.zeros_like(energy_db)
    delta[1:] = np.abs(energy_db[1:] - energy_db[:-1])
    return np.clip(_normalize(delta), 0.0, 1.0)


def score_prev_loss(n_frames: int, loss_mask: np.ndarray = None) -> np.ndarray:
    """Score based on whether previous frames were lost.
    loss_mask: True = frame available, False = lost."""
    if loss_mask is None:
        return np.zeros(n_frames)
    scores = np.zeros(n_frames)
    for i in range(1, n_frames):
        if not loss_mask[i - 1]:
            scores[i] = 1.0
        elif i >= 2 and not loss_mask[i - 2]:
            scores[i] = 0.6
    return scores


def score_a4(pcm: np.ndarray, frame_size: int, sr: int,
             loss_mask: np.ndarray = None,
             w_fc: float = 0.25, w_tr: float = 0.2, w_pitch: float = 0.15,
             w_gain: float = 0.15, w_hist: float = 0.15, w_net: float = 0.1) -> np.ndarray:
    """Compute A4 EVS-style frame criticality.

    I_EVS[n] = clamp(
        w_fc * S_frameclass + w_tr * S_transition + w_pitch * S_pitchinstab +
        w_gain * S_gainjump + w_hist * S_prevloss + w_net * S_netstress, 0, 1)
    """
    n_frames = len(pcm) // frame_size
    s_fc = classify_frames(pcm, frame_size, sr)
    s_tr = score_transition(pcm, frame_size, sr)
    s_pitch = score_pitch_instability(pcm, frame_size, sr)
    s_gain = score_gain_jump(pcm, frame_size)
    s_hist = score_prev_loss(n_frames, loss_mask)
    # Network stress placeholder (no real network state in PoC)
    s_net = np.zeros(n_frames)

    importance = (
        w_fc * s_fc + w_tr * s_tr + w_pitch * s_pitch +
        w_gain * s_gain + w_hist * s_hist + w_net * s_net
    )
    return np.clip(importance, 0.0, 1.0)
