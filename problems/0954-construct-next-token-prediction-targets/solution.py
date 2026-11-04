def make_input_target(token_ids, context_size):
    """Build next-token prediction (input, target) pair.

    Args:
        token_ids: list[int] of token IDs (length >= context_size + 1)
        context_size: int, number of tokens in the input window
    Returns:
        list: [x, y] where x and y are lists of length context_size,
              with y being x shifted right by one position.
    """
    if context_size == 0:
        x = []
        y = []
        return x, y
    elif context_size+1 > len(token_ids):
        raise ValueError()
    x = token_ids[:context_size]
    y = token_ids[1:context_size+1]
    return [x, y]