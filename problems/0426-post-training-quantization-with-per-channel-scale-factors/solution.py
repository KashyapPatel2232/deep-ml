import numpy as np

def per_channel_quantize(weight: np.ndarray, bits: int = 8) -> tuple:
    """
    Perform symmetric per-channel post-training quantization.

    Args:
        weight: Weight matrix of shape (out_channels, in_features)
        bits: Target bit-width for quantization (default: 8)

    Returns:
        Tuple of (quantized_weights, scale_factors, dequantized_weights)
        - quantized_weights: int array of shape (out_channels, in_features)
        - scale_factors: float array of shape (out_channels,)
        - dequantized_weights: float array of shape (out_channels, in_features)
    """
    Q_max = 2 ** (bits - 1) - 1
    scale_factors = np.max(np.abs(weight), axis = -1, keepdims = True) / Q_max
    scale_factors[scale_factors == 0.0] = 1.0
    quantized_weights = np.clip(np.round(weight / scale_factors), -2 ** (bits - 1), 2 ** (bits - 1) - 1).astype(np.int8)

    dequantized_weights = quantized_weights * scale_factors

    return quantized_weights, scale_factors.reshape(-1), dequantized_weights