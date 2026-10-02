import torch
from typing import Callable, Tuple

def numerical_gradient_check(f: Callable, x: torch.Tensor, analytical_grad: torch.Tensor, epsilon: float = 1e-7) -> Tuple[torch.Tensor, float]:
    """
    Perform numerical gradient checking using centered finite differences.

    Args:
        f: A function that takes a torch.Tensor and returns a scalar
        x: torch.Tensor, the point at which to check gradient
        analytical_grad: torch.Tensor, the analytically computed gradient
        epsilon: float, small value for finite difference approximation

    Returns:
        tuple: (numerical_grad, relative_error)
    """
    x = torch.tensor(x, dtype=torch.float64)
    analytical_grad = torch.tensor(analytical_grad , dtype=torch.float64)
    numerical_grad = torch.zeros_like(x)
    for i in range(len(x)):
           x_plus = x.clone()
           x_minus = x.clone()

           x_plus[i] += epsilon
           x_minus[i] -= epsilon

           numerical_grad[i] = (f(x_plus)-f(x_minus))/(2*epsilon)

    
    if (torch.sum(numerical_grad**2 + analytical_grad**2)) != 0.0:
        relative_error = torch.sqrt(torch.sum((numerical_grad - analytical_grad)**2))/(torch.sqrt(torch.sum(numerical_grad**2)) + torch.sqrt(torch.sum(analytical_grad**2)))
    else:
         relative_error = 1.0

    return numerical_grad, relative_error.item()