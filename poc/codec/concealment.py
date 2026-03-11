"""Concealment methods for lost codec frames."""
import numpy as np


def apply_concealment(
    tokens: np.ndarray, mask: np.ndarray, method: str = "zero_fill"
) -> np.ndarray:
    """Apply concealment to lost frames in token domain.

    Args:
        tokens: (n_frames, n_codebooks) token indices
        mask: (n_frames,) boolean, True = received, False = lost
        method: "zero_fill", "neighbor_copy", or "linear_interp"

    Returns:
        tokens with lost frames concealed (copy, original unchanged)
    """
    out = tokens.copy()
    n_frames = tokens.shape[0]
    lost = np.where(~mask)[0]

    if len(lost) == 0:
        return out

    if method == "zero_fill":
        out[lost] = 0

    elif method == "neighbor_copy":
        # Copy from nearest previous received frame; leading gaps use next received
        for i in lost:
            # Search backward for a received frame
            prev = None
            for j in range(i - 1, -1, -1):
                if mask[j]:
                    prev = j
                    break
            if prev is not None:
                out[i] = tokens[prev]
            else:
                # No previous received frame; search forward
                for j in range(i + 1, n_frames):
                    if mask[j]:
                        out[i] = tokens[j]
                        break
                else:
                    # No received frames at all
                    out[i] = 0

    elif method == "linear_interp":
        # For each gap, pick prev tokens if closer to start, else next tokens
        # (discrete nearest-neighbor interpolation in token domain)
        for i in lost:
            prev = None
            nxt = None
            for j in range(i - 1, -1, -1):
                if mask[j]:
                    prev = j
                    break
            for j in range(i + 1, n_frames):
                if mask[j]:
                    nxt = j
                    break

            if prev is not None and nxt is not None:
                # Fractional position within the gap
                frac = (i - prev) / (nxt - prev)
                if frac < 0.5:
                    out[i] = tokens[prev]
                else:
                    out[i] = tokens[nxt]
            elif prev is not None:
                out[i] = tokens[prev]
            elif nxt is not None:
                out[i] = tokens[nxt]
            else:
                out[i] = 0
    else:
        raise ValueError(f"Unknown concealment method: {method}")

    return out
