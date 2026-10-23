def compute_throughput(intervals):
    """
    Compute step-wise, cumulative, and overall tokens-per-second.

    Args:
        intervals: list of (tokens_processed, elapsed_seconds) tuples.

    Returns:
        dict with keys 'step_tps', 'cumulative_tps', 'overall_tps'.
    """
    step_tps = []
    cumulative = []
    overall = []
    cum_token = 0
    cum_time = 0

    if len(intervals) == 0:
        return {'step_tps':[], 'cumulative_tps':[], 'overall_tps':0.0}

    for tokens, interval in intervals:
        cum_token += tokens
        cum_time += interval
        if interval == 0.0:
            raise ValueError("ValueError")
        else:
            step_tps.append(round(tokens / interval, 2))
        cumulative.append(round(cum_token/cum_time,2))

    overall = cum_token / cum_time
    return {'step_tps':step_tps, 'cumulative_tps':cumulative, 'overall_tps':overall}


