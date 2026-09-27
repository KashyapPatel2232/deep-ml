import torch
from typing import List

def log_softmax(scores: List[float]) -> torch.Tensor:
    """
    Compute the log-softmax of a 1D list of scores using PyTorch.
    Args:
        scores: list of floats
    Returns:
        torch.Tensor of log-softmax values
    """
    # Your code here
    out_tensor = torch.tensor(scores)
    soft_max = torch.exp(out_tensor)/sum(torch.exp(out_tensor))
    res = torch.log(soft_max)
    return res
