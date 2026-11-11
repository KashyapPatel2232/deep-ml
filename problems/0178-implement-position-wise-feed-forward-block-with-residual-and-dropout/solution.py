import torch
import torch.nn as nn
import torch.nn.functional as F

class FFNBlock(nn.Module):
    """
    Position-wise feed-forward block: Dropout(W2 @ ReLU(W1 @ x + b1) + b2) + x

    Name the submodules exactly as follows — the tests set their parameters directly:
        self.linear1 -> nn.Linear(d_model, d_hidden)   # W1, b1
        self.linear2 -> nn.Linear(d_hidden, d_model)   # W2, b2
        self.dropout -> nn.Dropout(dropout_p)
    """
    def __init__(self, d_model, d_hidden, dropout_p=0.1):
        # Your code here
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        # Your code here
        output = x + self.dropout(self.linear2(F.relu(self.linear1(x))))
        return output