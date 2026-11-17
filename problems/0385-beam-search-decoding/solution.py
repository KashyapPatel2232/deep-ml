import torch

def beam_search_decode(log_probs: torch.Tensor, beam_width: int, top_k: int = 1) -> list:
    """
    Perform beam search decoding over a sequence of log-probability distributions.
    
    Args:
        log_probs: torch.Tensor of shape (T, V) with log-probabilities at each step
        beam_width: number of beams to maintain at each step
        top_k: number of top sequences to return
    
    Returns:
        List of (sequence, score) tuples sorted by score descending
    """
    # Your code here
    T, V = log_probs.shape
    beams = [(0.0, torch.tensor([], dtype = torch.long, device = log_probs.device))]

    for t in range(T):
        all_candidates = []
        step_log_probs = log_probs[t]
        
        for score, seq in beams:
            next_scores = score + step_log_probs

            top_scores, top_indices = torch.topk(next_scores, k = min(beam_width, V))

            for next_scores, v in zip(top_scores.tolist(), top_indices.tolist()):
                next_seq = torch.cat([seq, torch.tensor([v], dtype = torch.long, device = seq.device)])
                all_candidates.append((next_scores, next_seq))

        all_candidates.sort(key = lambda x: x[0], reverse= True)

        beams = all_candidates[:beam_width]

    num_results = min(top_k, len(beams))
    results = []

    for i in range(num_results):
        score, seq = beams[i]
        results.append((seq.tolist(), round(float(score), 4)))

    return results
