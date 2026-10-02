import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Args:
      x: numpy array, the point at which to check gradient
      analytical_grad: numpy array, the analytically computed gradient
      epsilon: float, small value for finite difference approximation
      
    Returns:
      tuple: (numerical_grad, relative_error)
    """
    numerical_grad = np.zeros_like(x)
    
    # Iterate over all elements in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        
        # Save the original value
        original_value = x[idx]
        
        # Compute f(x + epsilon)
        x[idx] = original_value + epsilon
        fx_plus = f(x)
        
        # Compute f(x - epsilon)
        x[idx] = original_value - epsilon
        fx_minus = f(x)
        
        # Restore original value
        x[idx] = original_value
        
        # Compute the partial derivative
        numerical_grad[idx] = (fx_plus - fx_minus) / (2 * epsilon)
        
        it.iternext()
        
    # Compute relative error
    numerator = np.linalg.norm(analytical_grad - numerical_grad)
    denominator = np.linalg.norm(analytical_grad) + np.linalg.norm(numerical_grad)
    relative_error = numerator / denominator if denominator != 0 else 0.0
        
    return numerical_grad, relative_error