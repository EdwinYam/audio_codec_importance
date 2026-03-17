"""Threshold strategies for streaming duplication decisions."""
import numpy as np
from collections import deque


def fixed_threshold(score: float, threshold: float) -> bool:
    """Duplicate if score exceeds a fixed threshold."""
    return score > threshold


def adaptive_mean_std(score: float, history: deque, k: float = 1.0) -> bool:
    """Threshold = running_mean + k * running_std of past scores."""
    if len(history) < 2:
        return True  # protect early frames by default
    arr = np.array(history)
    threshold = arr.mean() + k * arr.std()
    return score > threshold


def adaptive_quantile(score: float, history: deque, quantile: float = 0.7) -> bool:
    """Threshold = running quantile of past importance scores."""
    if len(history) < 2:
        return True
    threshold = np.quantile(list(history), quantile)
    return score > threshold


def should_duplicate(score: float, method: str, history: deque = None,
                     threshold: float = 0.5) -> bool:
    """Unified dispatch for threshold methods.

    Args:
        score: importance score for current frame
        method: threshold method name
        history: deque of past importance scores (for adaptive methods)
        threshold: fixed threshold value (for fixed methods)

    Returns:
        True if the frame should be duplicated
    """
    if method == "no_duplicate":
        return False
    if method == "all_duplicate":
        return True
    if method.startswith("fixed_"):
        return fixed_threshold(score, threshold)
    if method == "adaptive_mean_std":
        return adaptive_mean_std(score, history)
    if method.startswith("adaptive_quantile"):
        return adaptive_quantile(score, history)
    raise ValueError(f"Unknown threshold method: {method}")


def parse_method_config(method_name: str) -> dict:
    """Parse method name into config dict.

    Examples:
        "fixed_0.3" -> {"method": "fixed_0.3", "threshold": 0.3}
        "adaptive_quantile_70" -> {"method": "adaptive_quantile_70", "quantile": 0.7}
        "adaptive_mean_std" -> {"method": "adaptive_mean_std"}
        "no_duplicate" -> {"method": "no_duplicate"}
        "all_duplicate" -> {"method": "all_duplicate"}
    """
    if method_name.startswith("fixed_"):
        val = float(method_name.split("_", 1)[1])
        return {"method": method_name, "threshold": val}
    if method_name.startswith("adaptive_quantile_"):
        val = int(method_name.split("_")[-1])
        return {"method": method_name, "quantile": val / 100.0}
    return {"method": method_name}
