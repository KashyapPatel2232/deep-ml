import numpy as np

def suggest_rank(delta_W: np.ndarray, energy_threshold: float) -> int:
    """
    Return the smallest rank k such that the top-k singular values of delta_W
    capture at least `energy_threshold` of the total squared-singular-value energy.
    """
    delta_W = np.array(delta_W)
    
    if np.allclose(delta_W, 0.0):
        return 0
        
    S = np.linalg.svd(delta_W, compute_uv=False)
    S_squared = S**2

    total_energy = np.sum(S_squared)
    cumulative_energy = np.cumsum(S_squared) / total_energy
    
    valid_indices = np.where(cumulative_energy >= (energy_threshold - 1e-9))[0]
    
    if len(valid_indices) > 0:
        return int(valid_indices[0] + 1)
    else:
        return len(S)