import torch

def flip(a: torch.Tensor) -> torch.Tensor:
    """Reverse a vector via index arithmetic (no torch.flip)."""
    # Your code here
    n = len(a)
    rev = torch.zeros_like(a)
    for i in range(len(a)):
        rev[n - 1 - i] = a[i]

    return rev
