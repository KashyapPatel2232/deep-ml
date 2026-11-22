import numpy as np

def quantization_quality_check(log_probs_original: list, log_probs_quantized: list, thresholds: dict = None) -> dict:
    """
    Evaluate quantization quality by comparing perplexity before and after quantization.
    
    Args:
        log_probs_original: Token-level log-probabilities from the original (full precision) model
        log_probs_quantized: Token-level log-probabilities from the quantized model
        thresholds: Optional dict with 'excellent' and 'acceptable' percentage thresholds
    
    Returns:
        dict with perplexity metrics and quality classification
    """
    log_probs_original = np.asarray(log_probs_original)
    log_probs_quantized = np.asarray(log_probs_quantized)

    p_original = np.round(np.exp(-np.mean(log_probs_original)), 4).item()
    p_quantized = np.round(np.exp(-np.mean(log_probs_quantized)), 4).item()
    p_delta = p_quantized - p_original
    p_rel_percentage = ((p_quantized - p_original) / p_original) * 100
    if thresholds is None:
        quality = None
        if p_rel_percentage > 1 and p_rel_percentage <=5:
            quality= 'acceptable'
        elif p_rel_percentage <= 1:
            quality = 'excellent'
        else:
            quality = 'poor'
    else:
        quality = None
        if p_rel_percentage > thresholds['excellent'] and p_rel_percentage <= thresholds['acceptable']:
            quality= 'acceptable'
        elif p_rel_percentage <= thresholds['excellent']:
            quality = 'excellent'
        else:
            quality = 'poor'

    return {'perplexity_original': p_original, 'perplexity_quantized': p_quantized, 'perplexity_delta': p_delta, 'relative_delta_percent': p_rel_percentage, 'quality': quality}