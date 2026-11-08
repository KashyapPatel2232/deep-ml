import numpy as np

def sample_mlp_lm(C, W1, b1, W2, b2, block_size, itos, seed, max_tokens):
    # Your code here
    C = np.array(C)
    W1 = np.array(W1)
    b1 = np.array(b1)
    W2 = np.array(W2)
    b2 = np.array(b2)
    
    rng = np.random.default_rng(seed)
    output = []
    context = [0]*block_size

    for _ in range(max_tokens):
        e = C[context].reshape(1, -1)
        hidden = np.tanh(e @ W1 + b1)
        logits = hidden @ W2 + b2
        logits_stable =  logits - np.max(logits, axis = -1, keepdims=True)
        P = np.exp(logits_stable)/ np.sum(np.exp(logits_stable), axis = -1, keepdims = True)
        tok_id = rng.choice(len(P.flatten()), p = P.flatten())
        if tok_id == 0:
            break
        output.append(itos[tok_id])
        context = context[1:] + [int(tok_id)]

    return ''.join(output)