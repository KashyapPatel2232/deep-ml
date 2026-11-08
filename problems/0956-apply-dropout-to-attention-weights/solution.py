import numpy as np

def attention_dropout(attn_weights, values, dropout_rate, mask):
    """Apply dropout mask to attention weights and compute context vectors."""
    attn_weights = np.asarray(attn_weights)
    values = np.asarray(values)
    mask = np.asarray(mask)
    attn_weights_bar = attn_weights * mask / (1-dropout_rate)
    context = attn_weights_bar @ values
    return context
