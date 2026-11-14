import torch

def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """
    Apply Rotary Positional Embeddings (RoPE) to input embeddings.
    
    Args:
        x: Input embeddings of shape (seq_len, d), d must be even
        positions: Position indices of shape (seq_len,)
        base: Base for frequency computation (default: 10000.0)
    
    Returns:
        Embeddings with rotary positional encoding applied, shape (seq_len, d)
    """
    seq_len, d = x.shape
    inv_freq = 1.0 / (base ** (torch.arange(0, d, 2, device=x.device, dtype=x.dtype) / d))

    angles = torch.outer(positions.to(device=x.device, dtype=x.dtype), inv_freq)
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    x_even = x[:, 0::2]
    x_odd = x[:, 1::2]

    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos

    return torch.stack((out_even, out_odd), dim=-1).reshape(seq_len, d)