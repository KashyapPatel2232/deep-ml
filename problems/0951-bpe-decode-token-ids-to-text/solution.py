def bpe_decode(ids, vocab):
    """
    Args:
        ids: list[int] - token IDs to decode
        vocab: dict[int, str] - mapping from token ID to token string
    Returns:
        str - the decoded text
    """
    string = ""
    for i in range(len(ids)):
        token = vocab[ids[i]]
        if token[0] == 'G':
            string = string + ' ' + token[1:]
        else:
            string += token

    return string