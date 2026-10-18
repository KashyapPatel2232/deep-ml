import torch

def orthogonal_projection(v: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """
    Compute the orthogonal projection of vector v onto line L using PyTorch.

    :param v: The vector to be projected (torch.Tensor)
    :param L: The line vector defining the direction of projection (torch.Tensor)
    :return: torch.Tensor representing the projection of v onto L, rounded to 3 decimal places
    """
    v = torch.tensor(v)
	L = torch.tensor(L)
	return ((torch.sum(v * L)/ torch.sum(L**2)) * L)
