import torch

def instance_normalization(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """
    Perform Instance Normalization over a 4D tensor X of shape (B, C, H, W).
    gamma: scale parameter of shape (C,)
    beta: shift parameter of shape (C,)
    epsilon: small value for numerical stability
    Returns: normalized tensor of same shape as X
    """
    B, C, H , W = X.shape
    mean = X.mean(dim = (2,3), keepdims = True)
    var = X.var(dim = (2,3), keepdims = True, unbiased = False)
    norm = (X - mean) / torch.sqrt(var + epsilon)
    norm_x = norm * gamma + beta
    return norm_x