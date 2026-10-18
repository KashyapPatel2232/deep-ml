import torch

class LSTM:
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights and biases as float64 tensors
        self.Wf = torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float64)
        self.Wi = torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float64)
        self.Wc = torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float64)
        self.Wo = torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float64)

        self.bf = torch.zeros(hidden_size, 1, dtype=torch.float64)
        self.bi = torch.zeros(hidden_size, 1, dtype=torch.float64)
        self.bc = torch.zeros(hidden_size, 1, dtype=torch.float64)
        self.bo = torch.zeros(hidden_size, 1, dtype=torch.float64)

    def forward(self, x: torch.Tensor, initial_hidden_state: torch.Tensor, initial_cell_state: torch.Tensor):
        """
        Processes a sequence of inputs and returns the hidden states,
        final hidden state, and final cell state.

        Args:
            x: Input tensor of shape (seq_len, input_size)
            initial_hidden_state: Initial hidden state of shape (hidden_size, 1)
            initial_cell_state: Initial cell state of shape (hidden_size, 1)

        Returns:
            outputs: Tensor of hidden states at each time step
            h: Final hidden state tensor
            c: Final cell state tensor
        """
        c_t = initial_cell_state
        h_t = initial_hidden_state
        gates = []

        for i in range(len(x)):
            xt = x[i].unsqueeze(-1)
            combined = torch.concat((h_t, xt))

            f_t = torch.sigmoid(self.Wf @ combined + self.bf)
            i_t = torch.sigmoid(self.Wi @ combined + self.bi)
            c_bar = torch.tanh(self.Wc @ combined + self.bc)
            o_t = torch.sigmoid(self.Wo @ combined + self.bo)

            c_t = f_t * c_t + i_t * c_bar
            h_t = o_t * torch.tanh(c_t)

            gates.append(h_t)

        output = torch.stack(gates, dim = 0)

        return output, h_t, c_t