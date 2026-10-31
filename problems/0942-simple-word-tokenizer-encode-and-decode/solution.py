import re

def encode(text, vocab):
    split = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    output = [token.strip() for token in split if token.strip()]
    mapping = [vocab[token] for token in output]
    return mapping

def decode(ids, vocab):
    inverted_vocab = {v: k for k, v in vocab.items()}
    ls = [inverted_vocab[token_id] for token_id in ids]
    text = " ".join(ls)
    output = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
    return output