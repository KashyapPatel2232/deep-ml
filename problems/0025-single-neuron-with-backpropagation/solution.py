import torch
import torch.nn.functional as F

def train_neuron(features, labels, initial_weights, initial_bias, learning_rate, epochs):
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    initial_weights = torch.tensor(initial_weights, dtype=torch.float32, requires_grad=True)
    initial_bias = torch.tensor(initial_bias, dtype=torch.float32, requires_grad=True)

    mse_values = []

    for _ in range(epochs):
        out = features @ torch.transpose(initial_weights, dim0= -1, dim1 = 0) + initial_bias
        out = torch.sigmoid(out)
        mse_loss = F.mse_loss(out, labels)
        mse_values.append(mse_loss.item())

        mse_loss.backward()

        with torch.no_grad():
            initial_weights -= learning_rate * initial_weights.grad
            initial_bias -= learning_rate * initial_bias.grad

            initial_weights.grad.zero_()
            initial_bias.grad.zero_()

    return initial_weights.tolist(), initial_bias.tolist(), mse_values