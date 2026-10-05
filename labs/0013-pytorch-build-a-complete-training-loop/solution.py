import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr):
    """
    Train a PyTorch model and return training history.
    
    This is the standard PyTorch training pattern you'll use everywhere.
    Now you can use torch.optim to handle the gradient updates!
    
    Args:
        model: nn.Module to train
        X_train: training features, shape (N, ...)
        y_train: training labels, shape (N,)
        X_val: validation features, shape (M, ...)
        y_val: validation labels, shape (M,)
        epochs: number of training epochs
        batch_size: mini-batch size
        lr: learning rate
    
    Returns:
        history: List of dicts, one per epoch, with keys:
            - 'epoch': epoch number (starting from 1)
            - 'train_loss': average training loss for the epoch
            - 'val_loss': validation loss after the epoch
            - 'val_accuracy': validation accuracy after the epoch
    
    Steps:
        1. Create optimizer: optim.Adam(model.parameters(), lr=lr)
        2. Create loss function: nn.CrossEntropyLoss()
        3. For each epoch:
            a. Shuffle training data
            b. Loop over mini-batches:
                - optimizer.zero_grad()
                - Forward pass
                - Compute loss
                - loss.backward()
                - optimizer.step()
            c. Compute validation accuracy
            d. Append metrics to history
        4. Return history
    
    Hints:
        - torch.randperm(n) gives a random permutation for shuffling
        - Use model.train() before training, model.eval() before validation
        - Use torch.no_grad() during validation
        - logits.argmax(dim=1) gives predicted classes
    """
    # TODO: Implement the training loop
    history = []

    optimizer = optim.Adam(model.parameters(), lr = lr)

    # train_ds = TensorDataset(X_train, y_train)
    # valid_ds = TensorDataset(X_val, y_val)
    n_batch = len(X_train)//batch_size

    for i in range(epochs):
        # train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle= True)
        # valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle= False)
        model.train()
        rec = {'epoch':0, 'train_loss': 0.0, 'val_loss': 0.0, 'val_accuracy': 0.0}
        rec['epoch'] = i + 1
        permut = torch.randperm(len(X_train))
        X_shuffle = X_train[permut]
        y_shuffle = y_train[permut]

        for start in range(0, len(X_train), batch_size):
            end = min(start + batch_size, len(X_train))
            batch_x = X_shuffle[start:end]
            target_y = y_shuffle[start:end]

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = nn.CrossEntropyLoss()(preds, target_y)
            loss.backward()
            optimizer.step()

            rec["train_loss"] += loss.item() * batch_x.size(0)

        rec["train_loss"] /= len(X_train)

        
        model.eval()
        with torch.no_grad():
            preds = model(X_val)
            loss = nn.CrossEntropyLoss()(preds, y_val)
            pred = preds.argmax(dim=1)
            rec['val_loss'] = loss.item()
            rec['val_accuracy'] = (pred == y_val).float().mean().item()

        history.append(rec)
    
    return history