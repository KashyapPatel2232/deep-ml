import numpy as np

class NeuralNetwork:
    '''
    Build a neural network from scratch using only NumPy.
    Required architecture: 784 → 128 (ReLU) → 10 (Softmax)
    '''
    def __init__(self, input_size=784, hidden_size=128, output_size=10, lr=0.01):
        '''
        Initialize network parameters.
        Use small random initialization (e.g., Xavier/He initialization).
        '''
        self.lr = lr
        # TODO: Initialize weights and biases
        # Hint: Use np.random.randn() with proper scaling
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / (hidden_size + output_size))
        self.b2 = np.zeros(output_size)
    
    def forward(self, X):
        '''
        Forward pass through the network.
        
        Args:
            X: Input batch, shape (N, 784)
        
        Returns:
            probs: Class probabilities, shape (N, 10)
        
        Must cache intermediate values for backward pass!
        '''
        # TODO: Implement forward pass
        # Layer 1: X @ W1 + b1, then ReLU
        # Layer 2: hidden @ W2 + b2, then Softmax
        self.out_l1 = X @ self.W1 + self.b1
        self.out_l2= np.maximum(self.out_l1, 0.0)
        self.out_hidden = self.out_l2 @ self.W2 + self.b2
        self.output = np.exp(self.out_hidden - np.max(self.out_hidden, axis = -1, keepdims = True)) / np.sum(np.exp(self.out_hidden - np.max(self.out_hidden, axis = -1, keepdims = True)), axis = -1, keepdims = True)
        return self.output
    
    def backward(self, X, y, probs):
        '''
        Backward pass - compute gradients for all parameters.
        
        Args:
            X: Input batch, shape (N, 784)
            y: True labels, shape (N,)
            probs: Predicted probabilities from forward pass, shape (N, 10)
        
        Returns:
            loss: Scalar cross-entropy loss
        
        Must update self.W1, self.b1, self.W2, self.b2 using computed gradients!
        '''
        # TODO: Implement backward pass
        # 1. Compute loss
        # 2. Compute gradient of loss w.r.t. softmax output
        # 3. Backprop through linear layer 2
        # 4. Backprop through ReLU
        # 5. Backprop through linear layer 1
        # 6. Update all parameters using gradients
        y_one_hot = np.eye(probs.shape[1])[y]
        loss = -np.mean(np.sum(y_one_hot * np.log(np.clip(probs, 1e-12, 1.0)), axis = 1))
        dloss = (probs - y_one_hot)/len(y)
        dW2 = self.out_l2.T @ dloss
        dout_l2 = dloss @ self.W2.T
        db2 = np.sum(dloss, axis = 0)
        drelu = dout_l2 * (self.out_l1 > 0)
        dW1 = X.T @ drelu
        db1 = np.sum(drelu, axis = 0)

        self.W1 -= 2*self.lr* dW1
        self.b1 -= 2*self.lr * db1

        self.W2 -= 2*self.lr* dW2
        self.b2 -= 2*self.lr * db2

        return loss
    
    def train_step(self, X, y):
        '''
        Complete training step: forward + backward + update.
        
        Args:
            X: Input batch, shape (N, 784)
            y: True labels, shape (N,)
        
        Returns:
            loss: Scalar loss value
        '''
        probs = self.forward(X)
        loss = self.backward(X, y, probs)
        return loss
    
    def predict(self, X):
        '''
        Predict class labels.
        
        Args:
            X: Input batch, shape (N, 784)
        
        Returns:
            predictions: Predicted class labels, shape (N,)
        '''
        probs = self.forward(X)
        return np.argmax(probs, axis=1)