import torch

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> torch.Tensor:
    """
    Compute the Jacobian matrix using numerical differentiation.
    
    Args:
        f: Function that takes a list and returns a list
        x: Point at which to evaluate the Jacobian
        h: Step size for finite differences
    
    Returns:
        Jacobian matrix as torch.Tensor
    """
    cols = []
    
    for j in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[j] += h
        x_minus[j] -= h
        
        f_plus = torch.tensor(f(x_plus), dtype=torch.float64)
        f_minus = torch.tensor(f(x_minus), dtype=torch.float64)
        
        col = (f_plus - f_minus) / (2 * h)
        cols.append(col)
        
    return torch.stack(cols, dim=1).to(torch.float32)