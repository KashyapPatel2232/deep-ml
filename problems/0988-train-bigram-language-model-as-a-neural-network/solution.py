import numpy as np

def train_bigram_nn(xs, ys, vocab_size, lr=10.0, num_iters=100, alpha=0.0):
    """
    Train a one-layer neural-net bigram language model with full-batch gradient descent.

    Args:
        xs: list[int] of length N - input character indices in [0, vocab_size)
        ys: list[int] of length N - target next-character indices in [0, vocab_size)
        vocab_size: int V - number of distinct characters
        lr: float - learning rate
        num_iters: int - number of gradient descent steps
        alpha: float - L2 regularization strength on mean(W**2)

    Returns:
        list[list[float]] - trained weight matrix W of shape (V, V)
    """
    N = len(xs)
    V = vocab_size
    W = np.zeros((V, V))
    xs_one_hot = np.zeros((N, V))
    ys_one_hot = np.zeros((N,V))
    for i in range(N):
        xs_one_hot[i, xs[i]] = 1
        ys_one_hot[i, ys[i]] = 1

    if num_iters != 0:
        for i in range(num_iters):
            logits = xs_one_hot @ W
            P = np.exp(logits) / np.sum(np.exp(logits), axis = -1, keepdims = True)
            loss = -np.mean(np.log(np.sum(P*ys_one_hot, axis = -1, keepdims=True)), axis = -1) + alpha* np.mean(W**2)
            grad = xs_one_hot.T @ (P-ys_one_hot)/N + 2*alpha * W / (V**2)
            W = W - lr * grad


    return W