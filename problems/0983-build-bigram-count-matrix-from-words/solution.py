import numpy as np

def bigram_counts(words):
    # Your code here
    merge = []
    for ls in words:
        merge += list(ls)

    vocab = ['.'] + sorted(list(set(merge)))
    output = np.zeros((len(vocab), len(vocab)))

    for seq in words:
        aug_word = ['.'] + list(seq) + ['.']
        for i in range(len(aug_word) - 1):
            a = aug_word[i]
            b = aug_word[i+1]
            output[vocab.index(a)][vocab.index(b)] += 1
    return output