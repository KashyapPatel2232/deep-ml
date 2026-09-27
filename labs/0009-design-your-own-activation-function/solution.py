import numpy as np
import math
def activation(x):
    '''
    Apply an activation function element-wise.
    
    Args:
        x: numpy array of any shape (raw neuron outputs)
    
    Returns:
        numpy array of same shape (activated outputs)
    
    Requirements:
        - Must be non-linear (not just returning x)
        - Must work on arrays of any shape
        - Must be deterministic
    '''

    
    return np.maximum(1e-3*x,x)
