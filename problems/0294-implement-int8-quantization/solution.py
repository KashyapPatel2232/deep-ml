import numpy as np


def int8_quantize(x: list[float]) -> dict:
    """
    Perform symmetric INT8 quantization on a floating-point array.

    Args:
        x: Input list of floating-point values

    Returns:
        Dictionary with 'quantized', 'scale', and 'dequantized' keys
    """
    arr = np.asarray(x, dtype=np.float32)
    if arr.size == 0:
        return {"quantized": [], "scale": 1.0, "dequantized": []}

    max_abs = float(np.max(np.abs(arr)))
    scale = max_abs / 127.0 if max_abs > 0 else 1.0

    q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
    dq = [round(float(val), 4) for val in (q.astype(np.float32) * scale)]

    return {
        "quantized": q.tolist(),
        "scale": round(float(scale), 6),
        "dequantized": dq,
    }