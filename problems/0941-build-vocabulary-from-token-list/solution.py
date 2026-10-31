def build_vocab(tokens):
    """
    Build a vocabulary dictionary from a list of tokens.

    Args:
        tokens: list of string tokens

    Returns:
        Dict mapping each unique token (sorted) to a unique integer ID starting from 0.
    """
    unique_tokens = sorted(list(set(tokens)))
    mapping = {}
    for i in range(len(unique_tokens)):
        mapping[unique_tokens[i]] = i

    return mapping
