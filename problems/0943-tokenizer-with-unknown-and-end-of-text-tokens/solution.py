import re

def tokenize(vocab, text, mode):
    """
    vocab: dict[str, int] containing '<|unk|>' and '<|endoftext|>'
    text: str (if mode='encode') or list[int] (if mode='decode')
    mode: 'encode' or 'decode'
    """
    if mode == "encode":
        ls = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        text = [token.strip() for token in ls if token.strip()]
        output = []
        for i in range(len(text)):
            if text[i] in vocab.keys():
                output.append(vocab[text[i]])
            else:
                output.append(vocab['<|unk|>'])
        return output
    else:
        reverse_vocab = {v:k for k,v in vocab.items()}
        ls_str = [reverse_vocab[ids] for ids in text]
        string = " ".join(ls_str)
        M_string = re.sub(r'\s+([,.:;?_!"()\']|--|\s)', r'\1',string)
        return M_string
    return None
