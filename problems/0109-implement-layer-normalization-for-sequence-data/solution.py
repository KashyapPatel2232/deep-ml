import torch

def layer_normalization(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """
    Perform Layer Normalization.
    """
    # Your code here
    B, S, F = X.shape
    mean = X.mean(dim = 2, keepdims = True)
    var = X.var(dim = 2, keepdims = True, unbiased = False)
    norm = (X - mean)/torch.sqrt(var + epsilon)
    norm_X = norm * gamma + beta
    return norm_X