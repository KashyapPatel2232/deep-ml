import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def forward(self, x, initial_hidden_state, initial_cell_state):
        c_t = initial_cell_state
        h_t = initial_hidden_state
        gates = []

        for i in range(len(x)):
            x_t = np.asarray(x[i]).reshape(-1, 1)
            combined = np.concatenate([h_t, x_t], axis=0)  # shape: (2, 1)

            f_t = 1 / (1 + np.exp(-(self.Wf @ combined + self.bf)))
            i_t = 1 / (1 + np.exp(-(self.Wi @ combined + self.bi)))
            c_bar = np.tanh(self.Wc @ combined + self.bc)
            o_t = 1 / (1 + np.exp(-(self.Wo @ combined + self.bo)))

            c_t = f_t * c_t + i_t * c_bar
            h_t = o_t * np.tanh(c_t)

            gates.append((f_t, i_t, c_bar, o_t))

        return gates, h_t, c_t