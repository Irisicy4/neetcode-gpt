import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)

        x      = np.array(x,      dtype=float)
        W1     = np.array(W1,     dtype=float)
        b1     = np.array(b1,     dtype=float)
        W2     = np.array(W2,     dtype=float)
        b2     = np.array(b2,     dtype=float)
        y_true = np.array(y_true, dtype=float)
        
        # the last dimension of A must match the first dimension of B:
        y1 = np.dot(W1,x) + b1
        x2 = np.maximum(y1,0)
        y2 = np.dot(W2,x2) + b2

        e = y2-y_true
        n = len(y_true)

        loss = np.mean(e**2)
        db2 = 2/n*e
        dW2 = np.outer(db2,x2)
        
        dx2 = np.dot(W2.T,db2)
        db1 = dx2 * (y1 > 0)
        dW1 = np.outer(db1,x)

        return {
            'loss':round(loss,4),
            'dW1':np.round(dW1,4),
            'db1':np.round(db1,4),
            'dW2':np.round(dW2,4),
            'db2':np.round(db2,4)
            }
