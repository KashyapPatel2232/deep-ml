import torch, torch.nn as nn

def build_model() -> nn.Module:
    """
    Return a tiny nn.Module for MNIST classification (10 classes).
    IMPORTANT: If total trainable params > 2048, final accuracy will be set to 0.
    Tip: Consider very small convs, global average pooling, and tiny linear head.
    """
    class TinyNet(nn.Module):
         def __init__(self):
             super().__init__()
             
             self.stem_conv = nn.Conv2d(1, 16, kernel_size = 3, padding = 1, bias = False)
             self.stem_bn = nn.BatchNorm2d(16)
             self.pool = nn.MaxPool2d(2)

             self.dw1 = nn.Conv2d(16, 16, kernel_size = 3, padding = 1, groups = 16, bias = False)
             self.bn_dw1 = nn.BatchNorm2d(16)
             self.pw1 = nn.Conv2d(16, 24, kernel_size = 1, bias = False)
             self.bn_pw1 = nn.BatchNorm2d(24)

             self.dw2 = nn.Conv2d(24, 24, kernel_size = 3, padding = 1, groups = 24, bias = False)
             self.bn_dw2 = nn.BatchNorm2d(24)
             self.pw2 = nn.Conv2d(24, 24, kernel_size = 1, bias = False)
             self.bn_pw2 = nn.BatchNorm2d(24)

             self.act = nn.ReLU(inplace = True)
             self.gap = nn.AdaptiveAvgPool2d(1)
             self.classifier = nn.Linear(24, 10, bias = True)

         def forward(self, x):
             x = self.pool(self.act(self.stem_bn(self.stem_conv(x))))

             x = self.act(self.bn_dw1(self.dw1(x)))
             x = self.pool(self.act(self.bn_pw1(self.pw1(x))))

             residual = x
             out = self.act(self.bn_dw2(self.dw2(x)))
             out = self.act(self.bn_pw2(self.pw2(out)))
             x = residual + out

             x = self.gap(x).flatten(1)
             
             return self.classifier(x)

    return TinyNet()