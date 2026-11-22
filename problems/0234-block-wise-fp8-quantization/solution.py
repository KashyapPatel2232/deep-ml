import numpy as np

def fp8_block_quantize(
    tensor: np.ndarray,
    block_size: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantize a tensor to FP8-E4M3 format using block-wise scaling.
    
    Args:
        tensor: Input tensor of shape (N,) where N is divisible by block_size
        block_size: Number of elements per quantization block
        
    Returns:
        quantized: Quantized values of shape (N,), clipped to [-448, 448]
        scales: Per-block scale factors of shape (N // block_size,)
    """
    # Your code here
    FP_MAX = 448
    length = len(tensor)
    blocks = length // block_size
    output = []
    scales = []
    for i in range(blocks):
        temp = tensor[i*block_size: (i+1)* block_size]
        scale = (np.max(np.abs(temp)) + 1e-12) / FP_MAX
        quant = np.clip(np.round(temp / scale), -FP_MAX, FP_MAX)
        output.extend(quant)
        scales.append(scale)
    return np.asarray(output), np.asarray(scales)


def fp8_block_dequantize(
    quantized: np.ndarray,
    scales: np.ndarray,
    block_size: int = 128
) -> np.ndarray:
    """
    Dequantize FP8-E4M3 values back to full precision.
    
    Args:
        quantized: Quantized values of shape (N,)
        scales: Per-block scale factors of shape (N // block_size,)
        block_size: Number of elements per quantization block
        
    Returns:
        Dequantized tensor of shape (N,)
    """
    # Your code here
    FP_MAX = 448
    length = len(quantized)
    blocks = length // block_size
    output = []
    for i in range(blocks):
        temp = quantized[i* block_size: (i+1)* block_size]
        scale = scales[i]
        dequant =  temp * scale
        output.extend(dequant)
    return np.asarray(output)