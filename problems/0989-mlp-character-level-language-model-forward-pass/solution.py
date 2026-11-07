import numpy as np

def mlp_forward_loss(X, Y, C, W1, b1, W2, b2):
    """
    Forward pass of an MLP character-level language model.
    Returns mean cross-entropy loss as a float.
    """
    embedding = []
    for i in range(len(X)):
        temp = []
        for j in range(len(X[0])):
            temp.append(C[X[i][j]])

        embedding.append(temp)
    embedding = np.asarray(embedding)
    out_emb = embedding.reshape(len(X), -1)
    hidden_layer = np.tanh(out_emb @ W1 + b1)
    logits = hidden_layer @ W2 + b2
    logits_stable =  logits - np.max(logits, axis = -1, keepdims=True)
    P = np.exp(logits_stable)/ np.sum(np.exp(logits_stable), axis = -1, keepdims = True)
    loss = -np.sum(np.log(P[np.arange(len(Y)), Y]))/ len(Y)
    return loss
