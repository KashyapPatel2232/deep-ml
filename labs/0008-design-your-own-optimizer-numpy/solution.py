import numpy as np

def optimizer_step(param, grad, state, lr):
    # 1. Retrieve state variables
    t = state.get('step', 0) + 1
    m = state.get('m', np.zeros_like(grad))
    v = state.get('v', np.zeros_like(grad))
    
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    
    # 2. Internal Cosine Annealing Schedule
    # 4000 samples / 64 batch size * 5 epochs = ~315 steps
    total_steps = 315
    progress = min(t / total_steps, 1.0)
    
    # Smoothly decays from 1.0 down to 0.01 over the 5 epochs
    decay_factor = 0.01 + 0.5 * (1.0 - 0.01) * (1 + np.cos(np.pi * progress))
    
    # Scale down the aggressive 0.01 base LR and apply the decay
    effective_lr = (lr * 0.2) * decay_factor
    
    # 3. Update biased moment estimates
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad**2)
    
    # 4. Bias correction
    m_cap = m / (1 - beta1**t)
    v_cap = v / (1 - beta2**t)
    
    # 5. Nesterov Acceleration (The Nadam magic)
    # Re-injects the current gradient into the momentum to 'look ahead'
    m_nesterov = beta1 * m_cap + ((1 - beta1) * grad) / (1 - beta1**t)
    
    # 6. Apply update
    w = param.copy()
    w -= effective_lr * m_nesterov / (np.sqrt(v_cap) + epsilon)
    
    # 7. Save state
    state['step'] = t
    state['m'] = m
    state['v'] = v
    
    return w, state