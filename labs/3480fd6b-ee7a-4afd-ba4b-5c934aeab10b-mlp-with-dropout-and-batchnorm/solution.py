import torch
import torch.nn as nn


class RegularizedMLP(nn.Module):
    """MLP with BatchNorm1d and Dropout for binary classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_p: float = 0.3):
        super().__init__()
        # TODO: Build at least two hidden blocks:
        #   Linear -> BatchNorm1d -> ReLU (or similar) -> Dropout
        # Final layer: Linear to 1 logit.
        # Store layers on self (Sequential is fine).
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.layer =  nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_dim, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return shape (N,) logits for batch x of shape (N, input_dim)."""
        # TODO: run x through your network and squeeze the last dim if needed
        output = self.layer(x).squeeze(-1)
        return output


def train_model(model, X_train, y_train, epochs=150, lr=1e-2):
    """Train model in-place with BCEWithLogitsLoss + Adam. Return model."""
    # TODO:
    # - criterion = nn.BCEWithLogitsLoss()
    # - optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # - loop epochs: zero_grad -> forward -> loss -> backward -> step
    # - y_train is float 0/1 with shape (N,)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    for i in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
    
    return model