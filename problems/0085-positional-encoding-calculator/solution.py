import numpy as np

def pos_encoding(position: int, d_model: int) -> np.ndarray:
    """Return sinusoidal encodings for positions 0 through position - 1."""
    if position < 0 or d_model <= 0:
        return -1

    positions = np.arange(position)[:, np.newaxis]
    dimensions = np.arange(d_model)
    angles = positions / (10000.0 ** (2 * (dimensions // 2) / d_model))

    encoding = np.empty((position, d_model))
    encoding[:, 0::2] = np.sin(angles[:, 0::2])
    encoding[:, 1::2] = np.cos(angles[:, 1::2])
    return encoding.astype(np.float16)