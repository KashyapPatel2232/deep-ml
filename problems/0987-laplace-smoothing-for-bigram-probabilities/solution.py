import numpy as np

def smooth_bigram_probs(N, k):
    """
    N: 2D list or array of bigram counts (V x V)
    k: float, add-k smoothing constant (k >= 0)
    Returns: 2D list of smoothed bigram probabilities (V x V).
    """
    N = np.asarray(N)
    P = np.zeros_like(N)
    N_k = N + k
    P = N_k / np.sum(N_k, axis = 1, keepdims=True)
    return P
