def bpe_encode(text, token_to_id, merges):
    # Your code here
    if not text:
        return []

    mapped = []
    for i in range(len(text)):
        mapped.append(token_to_id[text[i]])

    for pair in merges:
        output = []
        i = 0
        while i < len(mapped):
            if i < len(mapped) - 1 and (mapped[i], mapped[i+1]) == pair:
                output.append(merges[pair])
                i += 2
            else:
                output.append(mapped[i])
                i += 1
        mapped = output

    return mapped