import torch

def per_channel_quantize(weight: torch.Tensor, bits: int = 8) -> tuple:
    """
    Perform symmetric per-channel post-training quantization.

    Args:
        weight: Weight matrix tensor of shape (out_channels, in_features)
        bits: Target bit-width for quantization (default: 8)

    Returns:
        Tuple of (quantized_weights, scale_factors, dequantized_weights)
        - quantized_weights: int tensor of shape (out_channels, in_features)
        - scale_factors: float tensor of shape (out_channels,)
        - dequantized_weights: float tensor of shape (out_channels, in_features)
    """
    Q_max = 2 ** (bits - 1) - 1
    scale_factors = torch.amax(torch.abs(weight), dim = -1, keepdim = True) / Q_max
    scale_factors = torch.where(scale_factors == 0, torch.ones_like(scale_factors), scale_factors)
    quantized_weights = torch.clamp(torch.round(weight / scale_factors), -2 ** (bits - 1), 2 ** (bits - 1) - 1).to(torch.int8)

    dequantized_weights = quantized_weights.to(weight.dtype) * scale_factors

    return quantized_weights, scale_factors.reshape(-1), dequantized_weights