import torch
import torch.nn as nn
import torch.optim as optim


class TinyCNN(nn.Module):
    """Small CNN: Conv2d -> ReLU -> pool -> (optional extras) -> Linear."""

    def __init__(self, img_size=8, n_classes=2):
        super().__init__()
        # TODO: define conv, activation, pooling, and classifier layers
        # Input shape: (N, 1, img_size, img_size)
        self.stem_conv = nn.Conv2d(1, 16, kernel_size= 3, padding=1, bias = False)
        self.stem_bn = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)

        self.dw1 = nn.Conv2d(16, 24, kernel_size=3, padding=1, bias=False)
        self.bn_dw1 = nn.BatchNorm2d(24)
        self.pw1 = nn.Conv2d(24, 24, kernel_size=3, padding=1, bias = False)
        self.bn_pw1 = nn.BatchNorm2d(24)

        self.dw2 = nn.Conv2d(24, 24, kernel_size=3, padding=1, bias=False)
        self.bn_dw2 = nn.BatchNorm2d(24)
        self.pw2 = nn.Conv2d(24, 24, kernel_size=3, padding=1, bias = False)
        self.bn_pw2 = nn.BatchNorm2d(24)

        self.act = nn.ReLU(inplace= True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(24, n_classes, bias = True)

    def forward(self, x):
        # TODO: implement forward pass; return logits of shape (N, n_classes)
        x = self.pool(self.act(self.stem_bn(self.stem_conv(x))))

        x = self.act(self.bn_dw1(self.dw1(x)))
        x = self.pool(self.act(self.bn_pw1(self.pw1(x))))

        residual = x
        out = self.act(self.bn_dw2(self.dw2(x)))
        out = self.bn_pw2(self.pw2(out))
        x = self.act(out + residual)

        x = self.gap(x).flatten(1)
        return self.classifier(x)
        

def build_model(img_size=8, n_classes=2):
    """Return an instance of your TinyCNN (or equivalent nn.Module)."""
    # TODO: return TinyCNN(img_size=img_size, n_classes=n_classes)
    return TinyCNN(img_size=img_size, n_classes=n_classes)

def train_model(model, train_x, train_y, epochs=15, lr=0.01, batch_size=32, seed=0):
    """Train model on train_x/train_y and return the trained model.

    Args:
        model: nn.Module from build_model
        train_x: FloatTensor (N, 1, H, W)
        train_y: LongTensor (N,)
        epochs: number of full passes over the data
        lr: optimizer learning rate
        batch_size: mini-batch size
        seed: RNG seed for shuffling / init determinism

    Returns:
        Trained model (same instance is fine).
    """
    # TODO:
    # - torch.manual_seed(seed)
    # - CrossEntropyLoss + Adam (or SGD)
    # - mini-batch loop for `epochs` epochs
    # - model.train(); zero_grad -> forward -> loss -> backward -> step
    torch.manual_seed(seed = seed)
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = lr)
    for epoch in range(epochs):
        indices = torch.randperm(train_x.size(0))

        shuffled_x = train_x[indices]
        shuffled_y = train_y[indices]

        for start in range(0, train_x.size(0), batch_size):
            end = start + batch_size
            x_batch = shuffled_x[start: end]
            y_batch = shuffled_y[start: end]
            
            optimizer.zero_grad()
            output = model(x_batch)
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()

    return model
