import numpy as np

def embedding_quantization_quality(embeddings: np.ndarray, num_bits: int = 8) -> dict:
    """
    Measure the quality of uniform embedding quantization using cosine similarity.
    
    Args:
        embeddings: Input embeddings of shape (n, d)
        num_bits: Number of bits for quantization
    
    Returns:
        Dictionary with mean_cosine_similarity, min_cosine_similarity,
        and max_cosine_similarity (each rounded to 6 decimal places)
    """

    q_max = 2**num_bits - 1
    similarities = []

    for embed in embeddings:
        v_min = np.min(embed)
        scale = (np.max(embed) - v_min) / q_max

        if scale == 0:
            quantized = embed.copy()
        else:
            quantized = np.round((embed - v_min) / scale) * scale + v_min

        denom = np.linalg.norm(embed) * np.linalg.norm(quantized)
        cosine_sim = 0.0 if denom == 0 else np.dot(embed, quantized) / denom
        similarities.append(cosine_sim)

    return {
        'mean_cosine_similarity': round(float(np.mean(similarities)), 6),
        'min_cosine_similarity': round(float(np.min(similarities)), 6),
        'max_cosine_similarity': round(float(np.max(similarities)), 6),
    }
