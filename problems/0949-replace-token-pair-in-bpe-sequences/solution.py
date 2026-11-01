def replace_pair(sequences, pair, new_id):
    # Your code here
    output = []
    for seq in sequences:
        i = 0
        ls = []
        while i < len(seq):
            if i < len(seq) - 1 and (seq[i], seq[i+1]) == pair:
                ls.append(new_id)
                i += 2
            else:
                ls.append(seq[i])
                i += 1
        output.append(ls)

    return output
