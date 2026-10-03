import torch

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10) -> torch.Tensor:
    """
    Implements Adam optimization algorithm using PyTorch's built-in optimizer.

    Args:
        f: The objective function to be optimized
        grad: A function that computes the gradient (unused; autograd is used instead)
        x0: Initial parameter values (torch.Tensor)
        learning_rate: The step size (default: 0.001)
        beta1: Exponential decay rate for the first moment estimates (default: 0.9)
        beta2: Exponential decay rate for the second moment estimates (default: 0.999)
        epsilon: A small constant for numerical stability (default: 1e-8)
        num_iterations: Number of iterations to run the optimizer (default: 10)

    Returns:
        torch.Tensor: Optimized parameters
    """
    x0 = torch.tensor(x0, dtype = torch.float64)
    m = 0.0
    v = 0.0
    # Your code here
    x_opt = x0
    for iteration in range(1, num_iterations+1):
        g = grad(x_opt)
        m = beta1 * m + (1-beta1) * g
        v = beta2 * v + (1-beta2) * (g**2)
        m_cap = m/(1-beta1**iteration)
        v_cap = v/(1-beta2**iteration)
        x_opt = x_opt - (learning_rate * m_cap)/(torch.sqrt(v_cap) + epsilon)
    
    return x_opt 
