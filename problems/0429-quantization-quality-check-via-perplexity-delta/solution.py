import torch

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
    log_probs_original = torch.tensor(log_probs_original, dtype=torch.float64)
    log_probs_quantized = torch.tensor(log_probs_quantized, dtype=torch.float64)

    p_original = torch.exp(-torch.mean(log_probs_original)).item()
    p_quantized = torch.exp(-torch.mean(log_probs_quantized)).item()
    p_delta = p_quantized - p_original
    p_rel_percentage = (p_delta / p_original) * 100
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

    return {
        'perplexity_original': round(p_original, 4),
        'perplexity_quantized': round(p_quantized, 4),
        'perplexity_delta': round(p_delta, 4),
        'relative_delta_percent': round(p_rel_percentage, 4),
        'quality': quality,
    }