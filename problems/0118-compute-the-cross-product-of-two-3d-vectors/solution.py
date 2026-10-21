import torch

def cross_product(a, b) -> torch.Tensor:
    """
    Compute the cross product of two 3D vectors a and b.
    Parameters:
        a (array-like or torch.Tensor): A 3-element vector.
        b (array-like or torch.Tensor): A 3-element vector.
    Returns:
        torch.Tensor: The cross product tensor.
    """
    # Your code here
    a = torch.tensor(a)
    b = torch.tensor(b)
    return torch.linalg.cross(a,b)