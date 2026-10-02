import torch
import torch.nn.functional as F

def cross_entropy_derivative(logits: torch.Tensor, target: int) -> torch.Tensor:
    """
    Compute the derivative of cross-entropy loss with respect to logits.
    
    Args:
        logits: Raw model outputs tensor
        target: Index of the true class
        
    Returns:
        Gradient tensor
    """
    # Your code here - can use autograd or the analytical formula
    y = torch.zeros_like(logits)
    y[target] = 1.0
    return torch.softmax(logits, dim = 0) - y
