"""Layer 1 audio quality metrics: PESQ, STOI, SI-SDR."""
import numpy as np
import librosa
from pesq import pesq as pesq_fn
from pystoi import stoi


def compute_stoi(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    """Compute STOI (Short-Time Objective Intelligibility)."""
    min_len = min(len(ref), len(deg))
    return stoi(ref[:min_len], deg[:min_len], sr, extended=False)


def compute_estoi(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    """Compute Extended STOI."""
    min_len = min(len(ref), len(deg))
    return stoi(ref[:min_len], deg[:min_len], sr, extended=True)


def compute_si_sdr(ref: np.ndarray, deg: np.ndarray) -> float:
    """Compute Scale-Invariant Signal-to-Distortion Ratio (dB)."""
    min_len = min(len(ref), len(deg))
    ref = ref[:min_len].astype(np.float64)
    deg = deg[:min_len].astype(np.float64)

    ref = ref - np.mean(ref)
    deg = deg - np.mean(deg)

    dot = np.dot(ref, deg)
    s_ref_sq = np.dot(ref, ref)
    if s_ref_sq < 1e-10:
        return -np.inf

    s_target = (dot / s_ref_sq) * ref
    e_noise = deg - s_target
    si_sdr = 10 * np.log10(
        np.dot(s_target, s_target) / (np.dot(e_noise, e_noise) + 1e-10)
    )
    return float(si_sdr)


def compute_pesq(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    """Compute PESQ (wideband). Resamples to 16kHz as required by PESQ."""
    min_len = min(len(ref), len(deg))
    ref16 = ref[:min_len]
    deg16 = deg[:min_len]
    if sr != 16000:
        ref16 = librosa.resample(ref16.astype(np.float32), orig_sr=sr, target_sr=16000)
        deg16 = librosa.resample(deg16.astype(np.float32), orig_sr=sr, target_sr=16000)
    try:
        return float(pesq_fn(16000, ref16, deg16, "wb"))
    except Exception:
        return float("nan")


def compute_all_quality(ref: np.ndarray, deg: np.ndarray, sr: int) -> dict:
    """Compute all Layer 1 quality metrics."""
    return {
        "PESQ": compute_pesq(ref, deg, sr),
        "STOI": compute_stoi(ref, deg, sr),
        "ESTOI": compute_estoi(ref, deg, sr),
        "SI-SDR": compute_si_sdr(ref, deg),
    }
