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
    # Your code here
    seq_len, d = x.shape
    index = torch.arange(d//2, dtype = int)
    output = torch.zeros_like(x)
    for row, pos in enumerate(positions):
        for i in index:
            theta = pos * (1 / (base ** (2*i / d)))
            output[row][2*i] = x[row][2*i] * torch.cos(theta) - x[row][2*i + 1] * torch.sin(theta)
            output[row][2*i + 1] = x[row][2*i] * torch.sin(theta) + x[row][2*i + 1] * torch.cos(theta)
    return output