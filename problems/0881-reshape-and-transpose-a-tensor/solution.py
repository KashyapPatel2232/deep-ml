import torch

def flatten_then_reshape(x: torch.Tensor, new_shape) -> torch.Tensor:
    # TODO: flatten x to 1-D, then rearrange into new_shape
    res = x.flatten()
    return res.reshape(new_shape)

def transpose_last_two(x: torch.Tensor) -> torch.Tensor:
    # TODO: swap the last two dimensions of x
    return x.transpose(-2,-1)
