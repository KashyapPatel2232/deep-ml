import torch
import torch.nn as nn


class DeepNetwork(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()

        # Input projection + batch normalization
        self.input_layer = nn.Linear(input_size, hidden_size, bias=False)
        self.input_bn = nn.BatchNorm1d(hidden_size)

        # 30 hidden layers (satisfies depth >= 30 constraint)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(30)]
        )
        self.bns_list = nn.ModuleList(
            [nn.BatchNorm1d(hidden_size) for _ in range(30)]
        )

        # Output classification layer
        self.output_layer = nn.Linear(hidden_size, num_classes)

        # Activation
        self.activation = nn.ReLU()

        self._enable_gradient_clipping(max_val=1.0)

    def _enable_gradient_clipping(self, max_val: float = 1.0):
        """Attaches a hook to every parameter to clamp gradients automatically

        during backward().
        """
        for param in self.parameters():
            if param.requires_grad:
                param.register_hook(
                    lambda grad, m=max_val: torch.clamp(grad, -m, m)
                )

    def forward(self, x):
        # Flatten input: (batch_size, 784)
        x = x.view(x.size(0), -1)

        # Initial projection
        x = self.activation(self.input_bn(self.input_layer(x)))

        # 30 residual blocks
        for layer, bn in zip(self.layers, self.bns_list):
            residual = x
            out = self.activation(bn(layer(x)))
            x = residual + out

        # Output logits tensor: (batch_size, 10)
        return self.output_layer(x)