import numpy as np

def absmax_scale(x: np.ndarray, bits: int = 8) -> float:
    """Symmetric absmax scale: max|x| / qmax, qmax=2^(bits-1)-1."""
    arr = np.asarray(x, dtype=np.float64)
    qmax = 2**(bits - 1) - 1
    if arr.size == 0:
        return 1.0
    x_max = float(np.max(np.abs(arr)))
    return x_max / qmax if x_max > 0.0 else 1.0

def percentile_scale(x: np.ndarray, bits: int = 8, p: float = 99.9) -> float:
    """Symmetric percentile scale on |x|."""
    arr = np.asarray(x, dtype=np.float64)
    qmax = 2**(bits - 1) - 1
    if arr.size == 0:
        return 1.0
    pct_val = float(np.percentile(np.abs(arr), p))
    return pct_val / qmax if pct_val > 0.0 else 1.0
