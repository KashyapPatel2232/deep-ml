import torch

def softmax_derivative(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the Jacobian matrix of the softmax function using PyTorch.
    
    Args:
        x: Input tensor
        
    Returns:
        Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
    """
    # Your code here - you can use torch.autograd.functional.jacobian
    # or compute it directly from softmax output
    sft = torch.exp(x)/sum(torch.exp(x))
    diag_sft = torch.diag(sft)
    outer_sft = torch.outer(sft, sft)
    return diag_sft-outer_sft
