import torch
import torch.nn as nn


def build_model() -> nn.Module:
    """Return a high-accuracy tiny CNN for MNIST classification.

    Total trainable parameters: 2,012 (Budget: <= 2,048).
    """

    class TinyNet(nn.Module):

        def __init__(self):
            super().__init__()

            self.act = nn.GELU()
            self.pool = nn.MaxPool2d(2)

            # Stem: (1, 28, 28) -> (16, 14, 14)
            self.stem_conv = nn.Conv2d(
                1, 16, kernel_size=3, padding=1, bias=False
            )
            self.stem_bn = nn.BatchNorm2d(16)

            # Block 1: (16, 14, 14) -> (25, 7, 7)
            self.dw1 = nn.Conv2d(
                16, 16, kernel_size=3, padding=1, groups=16, bias=False
            )
            self.bn_dw1 = nn.BatchNorm2d(16)
            self.pw1 = nn.Conv2d(16, 25, kernel_size=1, bias=False)
            self.bn_pw1 = nn.BatchNorm2d(25)

            # Block 2 (Residual): (25, 7, 7) -> (25, 7, 7)
            self.dw2 = nn.Conv2d(
                25, 25, kernel_size=3, padding=1, groups=25, bias=False
            )
            self.bn_dw2 = nn.BatchNorm2d(25)
            self.pw2 = nn.Conv2d(25, 25, kernel_size=1, bias=False)
            self.bn_pw2 = nn.BatchNorm2d(25)

            # Head: GAP -> Linear Classifier
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(25, 10, bias=True)

            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    nn.init.zeros_(m.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Stem
            x = self.pool(self.act(self.stem_bn(self.stem_conv(x))))

            # Block 1
            x = self.act(self.bn_dw1(self.dw1(x)))
            x = self.pool(self.act(self.bn_pw1(self.pw1(x))))

            # Block 2 (Linear bottleneck residual)
            residual = x
            out = self.act(self.bn_dw2(self.dw2(x)))
            out = self.bn_pw2(self.pw2(out))
            x = self.act(residual + out)

            # Head
            x = self.gap(x).flatten(1)
            return self.classifier(x)

    return TinyNet()