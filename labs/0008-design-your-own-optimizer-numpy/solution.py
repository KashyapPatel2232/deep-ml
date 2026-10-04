import numpy as np

def optimizer_step(param, grad, state, lr):
    '''
    Update a parameter using its gradient.
    
    Args:
        param: numpy array - current parameter values (any shape)
        grad: numpy array - gradient of loss w.r.t. param (same shape)
        state: dict - persists between calls, use to store any needed values
                      Example: state = {'step': 5, 'momentum': np.array([...])}
                      First call: state = {} (empty dict)
        lr: float - learning rate
    
    Returns:
        new_param: numpy array - updated parameter (must be same shape as param)
        state: dict - updated state dictionary
    '''
    # TODO: Implement your optimizer update rule
    # Hint: Think about gradient descent and how to use the gradient to update the parameter
    m = np.zeros_like(grad)
    v = np.zeros_like(grad)
    t = 7
    beta1 = 0.9
    beta2 = 0.999
    weight_decay = 0.1
    epsilon = 1e-8

    m = beta1 * m + (1-beta1) * grad
    v = beta2 * v + (1-beta2)* (grad**2)

    m_cap = m /(1-beta1**t)
    v_cap = v/(1-beta2**2)

    w = param.copy()

    w -= lr * weight_decay * grad
    w -= lr * m_cap /(np.sqrt(v_cap) + epsilon)

    state['step'] = t + 1
    state['momentum'] = m
    state['second moment'] = v
    
    new_param = w  # Replace with your update rule
    
    return new_param, state