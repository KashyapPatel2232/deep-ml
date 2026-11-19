import torch

def int8_quantize(x: list[float]) -> dict:
    """
    Perform symmetric INT8 quantization on a floating-point array using PyTorch.
    
    Args:
        x: Input list of floating-point values
        
    Returns:
        Dictionary with 'quantized', 'scale', and 'dequantized' keys
    """
    x = torch.tensor(x)
    scale = (torch.max(torch.abs(x)) / 127).item()
    if scale == 0:
        scale = 1.0
    q = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
    dq = [round(float(value), 4) for value in (q * scale).tolist()]
    return {'quantized': q.tolist(), 'scale': round(scale, 6), 'dequantized': dq}