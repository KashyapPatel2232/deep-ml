import numpy as np


class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        """Initialize the RNN with small random weights and zero biases."""
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))

    def forward(self, inputs):
        """Return one output for each row (timestep) in inputs."""
        inputs = np.asarray(inputs, dtype = float)

        hidden_states = [np.zeros((self.hidden_size, 1))]
        outputs = []

        for input_t in inputs:
            input_t = input_t.reshape(self.input_size, 1)
            hidden_t = np.tanh(self.W_xh @ input_t + self.W_hh @ hidden_states[-1] + self.b_h)
            output_t = self.W_hy @ hidden_t + self.b_y
            hidden_states.append(hidden_t)
            outputs.append(output_t)

        self._inputs = inputs
        self._hidden_states = hidden_states

        return np.array(outputs)

    def backward(self, inputs, targets, learning_rate):
        """Run backpropagation through time and update the model weights."""
        outputs = self.forward(inputs)

        targets = np.asarray(targets, dtype=float).reshape(outputs.shape)

        d_outputs = outputs - targets
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        dh_next = np.zeros((self.hidden_size, 1))

        for timestep in reversed(range(len(self._inputs))):
            input_t = self._inputs[timestep].reshape(self.input_size, 1)
            hidden_t = self._hidden_states[timestep + 1]
            hidden_prev = self._hidden_states[timestep]
            dy = d_outputs[timestep].reshape(self.output_size, 1)

            dW_hy += dy @ hidden_t.T
            db_y += dy

            dh = self.W_hy.T @ dy + dh_next
            da = dh * (1-hidden_t **2)
            dW_xh += da @ input_t.T
            dW_hh += da @ hidden_prev.T
            db_h += da
            dh_next = self.W_hh.T @ da

        self.W_xh -= learning_rate * dW_xh
        self.W_hh -= learning_rate * dW_hh
        self.W_hy -= learning_rate * dW_hy
        self.b_h -= learning_rate * db_h
        self.b_y -= learning_rate * db_y
        return outputs
