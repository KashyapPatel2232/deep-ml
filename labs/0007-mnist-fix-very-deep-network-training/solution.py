import torch
import torch.nn as nn


class DeepNetwork(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()

        # Input projection + batch normalization
        self.input_layer = nn.Linear(input_size, hidden_size, bias=False)
        self.input_bn = nn.BatchNorm1d(hidden_size)

        # 30 hidden layers (15 blocks x 2 layers; depth >= 30)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(30)]
        )
        self.bns_list = nn.ModuleList(
            [nn.BatchNorm1d(hidden_size) for _ in range(30)]
        )

        # Normalization layer for the accumulated representation
        self.final_bn = nn.BatchNorm1d(hidden_size)

        # Output classification layer
        self.output_layer = nn.Linear(hidden_size, num_classes)

        # LeakyReLU preserves bidirectional updates
        self.activation = nn.LeakyReLU(negative_slope=0.1)

        # Initialize weights for fast convergence
        self._init_weights()

        # Gradient clipping via backward hooks
        self._enable_gradient_clipping(max_val=1.0)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, a=0.1, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _enable_gradient_clipping(self, max_val: float = 1.0):
        """Clamps gradients element-wise during backward()."""
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

        # 15 two-layer residual blocks
        for i in range(0, 30, 2):
            residual = x

            # Sub-layer 1: Linear -> BN -> LeakyReLU
            out = self.layers[i](x)
            out = self.bns_list[i](out)
            out = self.activation(out)

            # Sub-layer 2: Linear -> BN -> LeakyReLU
            out = self.layers[i + 1](out)
            out = self.bns_list[i + 1](out)
            out = self.activation(out)

            # Pure identity addition: NO activation wrapping the sum
            x = residual + out

        # Standardize representation before logits
        x = self.final_bn(x)

        # Output logits tensor: (batch_size, 10)
        return self.output_layer(x)