"""Gilbert-Elliott burst loss model."""
import numpy as np


def apply_burst_loss(
    n_frames: int,
    target_plr: float,
    burst_ratio: float = 2.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Generate a frame mask using a two-state Gilbert-Elliott model.

    States: Good (low loss) and Bad (high loss).
    burst_ratio controls how bursty the loss is (higher = more bursty).

    Returns mask: True = frame received, False = lost.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Derive transition probabilities from target PLR and burst ratio
    # p_loss_good ~ 0 (no loss in good state)
    # p_loss_bad = high
    p_loss_bad = min(0.8, target_plr * burst_ratio * 2)
    p_loss_good = 0.0

    # Steady-state: pi_bad * p_loss_bad = target_plr
    pi_bad = target_plr / max(p_loss_bad, 1e-6)
    pi_bad = np.clip(pi_bad, 0.01, 0.99)
    pi_good = 1.0 - pi_bad

    # Transition probabilities
    # p_g2b / (p_g2b + p_b2g) = pi_bad
    avg_burst_len = max(2.0, burst_ratio * 3)
    p_b2g = 1.0 / avg_burst_len
    p_g2b = p_b2g * pi_bad / max(pi_good, 1e-6)
    p_g2b = np.clip(p_g2b, 0.001, 0.5)

    # Simulate
    mask = np.ones(n_frames, dtype=bool)
    state = 0  # 0 = Good, 1 = Bad
    for i in range(n_frames):
        if state == 0:
            if rng.random() < p_loss_good:
                mask[i] = False
            if rng.random() < p_g2b:
                state = 1
        else:
            if rng.random() < p_loss_bad:
                mask[i] = False
            if rng.random() < p_b2g:
                state = 0

    return mask
