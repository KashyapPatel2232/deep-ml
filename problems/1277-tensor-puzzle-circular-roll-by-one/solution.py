import torch

def roll(a: torch.Tensor) -> torch.Tensor:
    """Circular left-shift by one via index arithmetic (no torch.roll)."""
    # Your code here
    n = a.shape[0]
    return a[(torch.arange(n) + 1)%n]
