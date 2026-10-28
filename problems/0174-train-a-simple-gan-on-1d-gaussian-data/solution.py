import numpy as np


def train_gan(
    mean_real: float,
    std_real: float,
    latent_dim: int = 1,
    hidden_dim: int = 16,
    learning_rate: float = 0.001,
    epochs: int = 5000,
    batch_size: int = 128,
    seed: int = 42,
):
    np.random.seed(seed)

    W1G = np.random.normal(0, 0.01, (latent_dim, hidden_dim))
    W2G = np.random.normal(0, 0.01, (hidden_dim, 1))
    W1D = np.random.normal(0, 0.01, (1, hidden_dim))
    W2D = np.random.normal(0, 0.01, (hidden_dim, 1))

    b1G = np.zeros((1, hidden_dim))
    b2G = np.zeros((1, 1))
    b1D = np.zeros((1, hidden_dim))
    b2D = np.zeros((1, 1))

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def gen_forward(z):
        h1 = z @ W1G + b1G
        a1 = np.maximum(0, h1)
        x_gen = a1 @ W2G + b2G
        return x_gen, a1, h1

    for _ in range(epochs):
        x_real = np.random.normal(mean_real, std_real, batch_size).reshape(
            batch_size, 1
        )
        z = np.random.normal(0, 1, (batch_size, latent_dim))
        x_fake, _, _ = gen_forward(z)

        h_real = x_real @ W1D + b1D
        a_real = np.maximum(0, h_real)
        d_real = sigmoid(a_real @ W2D + b2D)

        h_fake = x_fake @ W1D + b1D
        a_fake = np.maximum(0, h_fake)
        d_fake = sigmoid(a_fake @ W2D + b2D)

        scale = 1.0 / batch_size

        delta_real = (d_real - 1.0) * scale
        delta_fake = d_fake * scale

        dW2D = a_real.T @ delta_real + a_fake.T @ delta_fake
        db2D = np.sum(delta_real + delta_fake, axis=0, keepdims=True)

        dh_real = (delta_real @ W2D.T) * (h_real > 0)
        dh_fake = (delta_fake @ W2D.T) * (h_fake > 0)

        dW1D = x_real.T @ dh_real + x_fake.T @ dh_fake
        db1D = np.sum(dh_real + dh_fake, axis=0, keepdims=True)

        W1D -= learning_rate * dW1D
        b1D -= learning_rate * db1D
        W2D -= learning_rate * dW2D
        b2D -= learning_rate * db2D

        z = np.random.normal(0, 1, (batch_size, latent_dim))
        x_fake, a1G, h1G = gen_forward(z)

        h_fake = x_fake @ W1D + b1D
        a_fake = np.maximum(0, h_fake)
        d_fake = sigmoid(a_fake @ W2D + b2D)

        delta_g = (d_fake - 1.0) * scale
        dh_fake = (delta_g @ W2D.T) * (h_fake > 0)
        dx_fake = dh_fake @ W1D.T

        dW2G = a1G.T @ dx_fake
        db2G = np.sum(dx_fake, axis=0, keepdims=True)

        dh1G = (dx_fake @ W2G.T) * (h1G > 0)
        dW1G = z.T @ dh1G
        db1G = np.sum(dh1G, axis=0, keepdims=True)

        W1G -= learning_rate * dW1G
        b1G -= learning_rate * db1G
        W2G -= learning_rate * dW2G
        b2G -= learning_rate * db2G

    return gen_forward