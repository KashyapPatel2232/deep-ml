from collections import Counter, defaultdict

def train_tokenizer(corpus, vocab_size):
    """
    Build a tokenizer from the training corpus for downstream language modeling.

    The harness will use your tokenizer to train a bigram language model.
    Better tokenizers create more predictable token sequences, leading to
    lower bits-per-character on held-out data.

    Args:
        corpus: list[str] -- training documents (names like 'emma', 'olivia', ...)
        vocab_size: int   -- maximum vocabulary size (256)

    Returns:
        encode: callable, str -> list[int]
            Convert text to token IDs. All IDs must be in [0, vocab_size).
        decode: callable, list[int] -> str
            Convert token IDs back to text. Must satisfy decode(encode(s)) == s.
    """
    # TODO: Build your tokenizer
    # A simple approach: map each character to an integer.
    # A better approach: learn subword tokens from the corpus
    # (common pairs, frequent substrings, BPE, etc.)
    # The better your tokenizer compresses the data, the higher your score.
    chars = sorted(set("".join(corpus)))
    token_to_id = {c: i for i, c in enumerate(chars)}
    id_to_token = {i:c for i, c in enumerate(chars)}

    word_counts = Counter(corpus)
    words = [[token_to_id[c] for c in word] for word in word_counts]
    freqs = list(word_counts.values())

    merges = {}
    merge_priority = {}

    num_merges = vocab_size - len(chars)

    for rank in range(num_merges):
        pair_counts = defaultdict(int)
        for word, freq in zip(words, freqs):
            for i in range(len(word) - 1):
                pair_counts[(word[i], word[i+1])] += freq

        if not pair_counts:
            break

        best_pair = max(pair_counts, key=lambda p:(pair_counts[p], -p[0], -p[1]))

        if pair_counts[best_pair] < 5:
            break

        new_id = len(token_to_id)
        new_str = id_to_token[best_pair[0]] + id_to_token[best_pair[1]]
        token_to_id[new_str] = new_id
        id_to_token[new_id] = new_str

        merges[best_pair] = new_id
        merge_priority[best_pair] = rank

        p0, p1 = best_pair
        for idx,word in enumerate(words):
            if len(word) < 2:
                continue
            new_word = []
            i = 0
            n = len(word)
            while i < n:
                if i < n-1 and word[i] == p0 and word[i+1] ==p1:
                    new_word.append(new_id)
                    i += 2
                else:
                    new_word.append(word[i])
                    i+=1
            words[idx] =new_word
    cache = {}

    def encode(text):
        if not text:
            return []
        if text in cache:
            return list(cache[text])

        ids =[token_to_id[c] for c in text]

        while len(ids) >=2:
            best_pair = None
            best_rank = float("inf")
            for i in range(len(ids) -1):
                pair = (ids[i], ids[i+1])
                rank = merge_priority.get(pair, float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            new_id = merges[best_pair]
            p0, p1 = best_pair
            new_ids = []
            i = 0
            n = len(ids)
            while i < n:
                if i < n - 1 and ids[i] == p0 and ids[i + 1] == p1:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
            
        cache[text] = ids
        return list(ids)
    def decode(token_ids):
        return "".join(id_to_token[t] for t in token_ids)

    return encode, decode
