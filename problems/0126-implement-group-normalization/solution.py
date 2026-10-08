import torch

def group_normalization(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, num_groups: int, epsilon: float = 1e-5) -> torch.Tensor:
    """
    Perform Group Normalization on a 4D input tensor.
    
    Args:
        X: torch tensor of shape (B, C, H, W), input data
        gamma: torch tensor of shape (1, C, 1, 1), scale parameter
        beta: torch tensor of shape (1, C, 1, 1), shift parameter
        num_groups: number of groups for normalization
        epsilon: small constant to avoid division by zero
    
    Returns:
        norm_X: torch tensor of shape (B, C, H, W), normalized output
    """
    B, C, H, W = X.shape
    grp_sz = C // num_groups 

    X_reshaped = X.view(B, num_groups, grp_sz, H, W)

    mean = X_reshaped.mean(dim = (2,3,4), keepdims = True)
    var = X_reshaped.var(dim = (2,3,4), keepdims = True, unbiased = False)

    norm = (X_reshaped - mean) / torch.sqrt(var + epsilon)
    norm = norm.view(B, C, H, W)
    norm_X = gamma * norm + beta

    return norm_X