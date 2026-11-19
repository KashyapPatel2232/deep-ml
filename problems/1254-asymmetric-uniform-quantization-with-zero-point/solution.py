import numpy as np

def asymmetric_quantize(x: np.ndarray, bits: int = 8):
    """
    Asymmetric uniform quantization.

    Returns q (int array), scale (float), zero_point (int), x_hat (float array).
    Unsigned grid: qmin=0, qmax=2^bits-1.
    """
    # Your code here
    x_max = np.max(x)
    x_min = np.min(x)

    q_min = 0
    q_max = 2**bits - 1

    if x_max - x_min < 1e-12:
        q = np.zeros_like(x, dtype=np.min_scalar_type(q_max))
        return q, 1.0, 0, x.copy()

    scale = (x_max - x_min) / (q_max - q_min)
    zero_point = np.clip(np.round(q_min - x_min / scale), q_min, q_max)
    q = np.clip(np.round(x/scale + zero_point), q_min, q_max).astype(np.min_scalar_type(q_max))
    x_hat = scale * (q - zero_point)
    return q, scale, zero_point, x_hat
