import numpy as np

def bigram_avg_nll(P, words, stoi):
    """Average negative log-likelihood of words under a bigram model."""
    if not words:
        return 0.0
    loss = 0
    N = 0
    for ls in words:
        aug_words = ['.'] + list(ls) + ['.']
        for i in range(len(aug_words) -1):
            N += 1
            a = stoi[aug_words[i]]
            b = stoi[aug_words[i+1]]
            loss += np.log(P[a][b])
    return -round(loss / N,4)
