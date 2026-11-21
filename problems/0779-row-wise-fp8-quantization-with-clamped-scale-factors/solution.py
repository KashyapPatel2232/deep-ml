import numpy as np

def row_wise_fp8_quantize(X: np.ndarray) -> dict:
    """
    Perform row-wise FP8 quantization with clamped per-row scale factors.

    Args:
        X: 2D numpy array of shape (num_rows, num_cols)

    Returns:
        Dictionary with keys:
            'quantized': quantized 2D numpy array, same shape as X
            'scales':    1D numpy array of per-row scale factors (length num_rows)
    """
    FP8_MAX = 448.0
    scales = np.max(np.abs(X), axis = -1, keepdims = True) / FP8_MAX
    scales = np.minimum(scales, 1200.0)
    scales = np.where(scales == 0, 1.0, scales) 
    quantized = np.clip(X / scales, -FP8_MAX, FP8_MAX)
    return {'quantized':quantized, 'scales':scales.squeeze(-1)}