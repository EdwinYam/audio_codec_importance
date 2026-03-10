"""Layer 3: Importance method diagnostic metrics."""
import numpy as np
from scipy.stats import spearmanr


def oracle_spearman(predicted: np.ndarray, oracle: np.ndarray) -> float:
    """Spearman rank correlation between predicted importance and oracle damage."""
    if len(predicted) < 3:
        return 0.0
    corr, _ = spearmanr(predicted, oracle)
    return float(corr) if not np.isnan(corr) else 0.0


def precision_at_k(predicted: np.ndarray, oracle: np.ndarray, k_frac: float = 0.2) -> float:
    """Of the top-K predicted frames, what fraction are truly in the top-K by oracle?"""
    n = len(predicted)
    k = max(1, int(n * k_frac))
    pred_top = set(np.argsort(predicted)[::-1][:k])
    oracle_top = set(np.argsort(oracle)[::-1][:k])
    return len(pred_top & oracle_top) / k


def damage_captured_ratio(
    protected_indices: np.ndarray, oracle_damage: np.ndarray
) -> float:
    """What fraction of total oracle damage is covered by the protected frames?"""
    total = oracle_damage.sum()
    if total < 1e-10:
        return 1.0
    captured = oracle_damage[protected_indices].sum() if len(protected_indices) > 0 else 0.0
    return float(captured / total)


def concealment_stats(mask: np.ndarray) -> dict:
    """Compute concealment-related call system metrics from a frame mask."""
    n_frames = len(mask)
    n_lost = np.sum(~mask)
    concealment_rate = n_lost / n_frames if n_frames > 0 else 0.0

    # Consecutive concealment bursts
    burst_lengths = []
    current_burst = 0
    for available in mask:
        if not available:
            current_burst += 1
        else:
            if current_burst > 0:
                burst_lengths.append(current_burst)
            current_burst = 0
    if current_burst > 0:
        burst_lengths.append(current_burst)

    return {
        "concealment_rate": concealment_rate,
        "n_lost_frames": int(n_lost),
        "mean_burst_len": float(np.mean(burst_lengths)) if burst_lengths else 0.0,
        "max_burst_len": int(max(burst_lengths)) if burst_lengths else 0,
        "n_bursts": len(burst_lengths),
    }
