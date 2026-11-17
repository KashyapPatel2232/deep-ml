import numpy as np

def beam_search_decode(log_probs: np.ndarray, beam_width: int, top_k: int = 1) -> list:
    """
    Perform beam search decoding over a sequence of log-probability distributions.
    
    Args:
        log_probs: numpy array of shape (T, V) with log-probabilities at each step
        beam_width: number of beams to maintain at each step
        top_k: number of top sequences to return
    
    Returns:
        List of (sequence, score) tuples sorted by score descending
    """
    # Your code here
    beams = [(0.0, [])]
    for t in range(log_probs.shape[0]):
        all_candidates = []

        for score, seq in beams:
            for v in range(log_probs.shape[1]):
                next_score = score + log_probs[t, v]
                next_seq = seq + [v]
                all_candidates.append((next_score, next_seq))

        all_candidates.sort(key = lambda x: x[0], reverse = True)
        beams = all_candidates[:beam_width]
    
    num_results = min(top_k, len(beams))
    results = []

    for i in range(num_results):
        score, seq = beams[i]
        results.append((seq, round(float(score), 4)))
    return results