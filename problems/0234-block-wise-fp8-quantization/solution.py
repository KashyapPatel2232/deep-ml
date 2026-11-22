import torch

def fp8_block_quantize(
    tensor: torch.Tensor,
    block_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
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
        scale = (torch.max(torch.abs(temp)) + 1e-12) / FP_MAX
        quant = torch.clamp(torch.round(temp / scale), -FP_MAX, FP_MAX)
        output.extend(quant)
        scales.append(scale)
    return torch.tensor(output), torch.tensor(scales)


def fp8_block_dequantize(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    block_size: int = 128
) -> torch.Tensor:
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
    return torch.tensor(output)