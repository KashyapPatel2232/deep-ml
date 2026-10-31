def vocab_size(tokens, special_tokens=None):
    """Return the vocabulary size from a list of preprocessed tokens.

    Args:
        tokens: list[str] of preprocessed tokens
        special_tokens: optional list[str] of special tokens to include
    Returns:
        int: number of unique tokens in the combined vocabulary
    """
    uniq = list(set(tokens))
    total = len(uniq)
    if special_tokens is None:
        return total
    for token in list(set(special_tokens)):
        if token not in tokens:
            total += 1
    return total