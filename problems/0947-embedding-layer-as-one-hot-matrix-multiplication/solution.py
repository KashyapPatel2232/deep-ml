import numpy as np

def embedding_via_one_hot(token_ids, W):
    """
    Compute token embeddings via one-hot encoding and matrix multiplication.

    Args:
        token_ids: list or 1D array of integer token IDs
        W: numpy array of shape (vocab_size, embed_dim)

    Returns:
        numpy array of shape (len(token_ids), embed_dim)
    """
    num_tokens = len(token_ids)
    vocab_size, _ = W.shape
    H = np.zeros((num_tokens, vocab_size), dtype=float)
    for i in range(num_tokens):
        H[i, token_ids[i]] = 1

    return H@ W