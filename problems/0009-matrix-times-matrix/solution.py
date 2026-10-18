import torch

def matrixmul(a, b) -> torch.Tensor:
    """
    Multiply two matrices using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2D tensor of shape (m, n) or a scalar tensor -1 if dimensions mismatch.
    """
    a = torch.tensor(a)
    b = torch.tensor(b)
    if len(a[0]) != len(b):
        return -1

    val = a @ b
    return val
