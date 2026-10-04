import numpy as np

def optimizer_step(param, grad, state, lr):
    t = state.get('step', 0) + 1
    m = state.get('m', np.zeros_like(grad))
    v = state.get('v', np.zeros_like(grad))
    
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    total_steps = 315
    progress = min(t / total_steps, 1.0)
    
    decay_factor = 0.01 + 0.5 * (1.0 - 0.01) * (1 + np.cos(np.pi * progress))

    effective_lr = (lr * 0.2) * decay_factor

    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad**2)
    
    m_cap = m / (1 - beta1**t)
    v_cap = v / (1 - beta2**t)
    
    m_nesterov = beta1 * m_cap + ((1 - beta1) * grad) / (1 - beta1**t)
    
    w = param.copy()
    w -= effective_lr * m_nesterov / (np.sqrt(v_cap) + epsilon)

    state['step'] = t
    state['m'] = m
    state['v'] = v
    
    return w, state
