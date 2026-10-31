import torch

def entropy_and_cross_entropy(P: torch.Tensor, Q: torch.Tensor) -> tuple[float, float]:
    """
    Compute entropy of P and cross-entropy between P and Q.
    
    Args:
        P: True probability distribution (torch.Tensor)
        Q: Predicted probability distribution (torch.Tensor)
    
    Returns:
        Tuple of (entropy H(P), cross-entropy H(P,Q))
    """
    # Your code here
    pos_p = P > 0
    entropy = -torch.sum(P[pos_p] * torch.log(P[pos_p]))
    cross_entropy = -torch.sum(P[pos_p] * torch.log(Q[pos_p]))
    return entropy.item(), cross_entropy.item()