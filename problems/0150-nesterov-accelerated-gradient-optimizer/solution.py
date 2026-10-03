import torch

def nag_optimizer(parameter: torch.Tensor, grad_fn, velocity: torch.Tensor, learning_rate: float = 0.01, momentum: float = 0.9) -> tuple:
    """
    Update parameters using the Nesterov Accelerated Gradient optimizer.
    Uses a "look-ahead" approach to improve convergence by applying momentum before computing the gradient.
    Args:
        parameter: Current parameter value (torch.Tensor)
        grad_fn: Function that computes the gradient at a given position
        velocity: Current velocity (momentum term) (torch.Tensor)
        learning_rate: Learning rate (default=0.01)
        momentum: Momentum coefficient (default=0.9)
    Returns:
        tuple: (updated_parameter, updated_velocity) as torch.Tensors
    """
    # Your code here
    parameter_ahead = parameter - momentum * velocity
    velocity = momentum * velocity + learning_rate * grad_fn(parameter_ahead)
    parameter = parameter - velocity
    return torch.round(parameter, decimals=5), torch.round(velocity, decimals=5)