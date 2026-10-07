import torch

def batch_normalization(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """Perform Batch Normalization on a 4D tensor in BCHW format."""
    mean_c = X.mean(dim = (0,2,3), keepdim=True)
    var_c = ((X-mean_c)**2).mean(dim = (0,2,3), keepdim=True)
    norm = (X-mean_c) / (torch.sqrt(var_c + epsilon))
    out = gamma * norm + beta
    return out