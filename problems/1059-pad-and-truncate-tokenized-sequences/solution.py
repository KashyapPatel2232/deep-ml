import numpy as np
def pad_and_truncate(sequences, pad_token_id, max_length=None):
    """
    Pad and/or truncate tokenized sequences to a uniform length.

    Args:
        sequences: list of lists of int token ids
        pad_token_id: int, token id used for padding
        max_length: int or None, target length. If None, use the longest sequence length.

    Returns:
        list of lists of int token ids, each of length max_length.
    """
    max_len = 0
    if max_length is None:
        for ls in sequences:
            if max_len < len(ls):
                max_len = len(ls)
    else:
        max_len = max_length

    output = []
    for ls in sequences:
        if len(ls) < max_len:
            temp = ls + [pad_token_id] * (max_len - len(ls))
            output.append(temp)
        elif len(ls) == max_len:
            temp = ls
            output.append(temp)
        else:
            temp = ls[:max_len]
            output.append(temp)
    return output