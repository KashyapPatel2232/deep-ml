def most_frequent_pair(sequences):
    """
    Args:
        sequences: list[list[int]] - list of token ID sequences
    Returns:
        tuple(int, int) or None - the most frequent adjacent pair, with ties
        broken by first appearance. Returns None if no pair exists.
    """
    if not sequences:
        return None

    dic = {}
    first_appearance = {}
    pair_count = 0

    for seq in sequences:
        for i in range(len(seq)-1):
            pair = (seq[i], seq[i+1])
            if pair not in dic:
                dic[pair] = 0
                first_appearance[pair] = pair_count
                pair_count += 1
            dic[pair] += 1

    if not dic:
        return None

    return max(dic.keys(), key = lambda k: (dic[k], -first_appearance[k]))

