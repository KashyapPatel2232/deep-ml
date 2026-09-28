import numpy as np

def activation(x):
    alpha = 1.0
    gain = 1.45
    return gain * np.where(x > 0, x, alpha * (np.exp(np.clip(x, -50, 0)) - 1))