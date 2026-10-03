import torch

def rmsprop_update(params: torch.Tensor, grads: torch.Tensor, cache: torch.Tensor,
                   lr: float = 0.01, beta: float = 0.9, epsilon: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform RMSProp optimization update.
    
    Args:
        params: Tensor of parameter values
        grads: Tensor of gradients for each parameter
        cache: Tensor of cache values (moving average of squared gradients)
        lr: Learning rate
        beta: Decay rate for moving average
        epsilon: Small constant for numerical stability
    
    Returns:
        Tuple of (updated_params, updated_cache)
    """
    # Your code here
    cache = beta * cache + (1 - beta) * (grads**2)
    lr_eff = lr/(torch.sqrt(cache) + epsilon)
    params = params - (lr_eff) * grads
    return params, cache