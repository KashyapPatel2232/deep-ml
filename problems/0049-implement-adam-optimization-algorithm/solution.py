import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
    # Note: `f` is accepted only for interface parity with the PyTorch/Tinygrad
    # variants, which derive gradients from it via autograd. This version uses
    # `grad` only — the objective value `f` is never evaluated.
    # Your code here
    x0 = np.array(x0, dtype = np.float64)
    m = 0
    v = 0
    x_opt = x0
    for iteration in range(1, num_iterations+1):
        m = beta1 * m + (1-beta1)* grad(x_opt)
        v = beta2 * v + (1-beta2) * (grad(x_opt)**2)
        m_cap = m/(1-beta1**iteration)
        v_cap = v/(1-beta2**iteration)
        x_opt -= (m_cap*learning_rate)/(np.sqrt(v_cap) + epsilon)

    return x_opt