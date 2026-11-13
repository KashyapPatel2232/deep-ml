import numpy as np

def temperature_sampling(logits: np.ndarray, temperature: float) -> list:
    """
    Compute temperature-scaled softmax probabilities from logits.
    
    Args:
        logits: 1D numpy array of raw model output scores
        temperature: float controlling distribution sharpness
    
    Returns:
        List of probabilities after temperature scaling
    """
    # Your code here
    if temperature == 0:
        probs = np.zeros_like(logits)
        probs[np.argmax(logits)] = 1
        return probs

    z = logits / temperature
    z = z - np.max(z, axis = -1, keepdims = True)
    z = np.exp(z) / np.sum(np.exp(z), axis = -1, keepdims = True)
    return z