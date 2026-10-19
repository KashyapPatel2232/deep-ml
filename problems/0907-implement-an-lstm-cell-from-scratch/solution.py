import math
import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        # TODO: create W_ih, W_hh, b_ih, b_hh with the shapes specified above
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.empty(4*hidden_size, hidden_size))
        self.b_ih = nn.Parameter(torch.empty(4* hidden_size))
        self.b_hh = nn.Parameter(torch.empty(4* hidden_size))

        self.initialize_params()

    def initialize_params(self):
        std = 1/math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -std, std)


    def forward(self, x, state):
        h_prev, c_prev = state
        # TODO: compute the four gates and return (h_new, c_new)
        g = x @ self.W_ih.T + self.b_ih + h_prev @ self.W_hh.T + self.b_hh
        i, f, c_bar, o = g.chunk(4, dim = 1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        c_bar = torch.tanh(c_bar)

        c_new = f * c_prev + i * c_bar
        h_new = o * torch.tanh(c_new)

        return h_new, c_new