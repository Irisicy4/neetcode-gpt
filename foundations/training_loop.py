import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # n_samples defined once here so it can be used in gradient formulas below
        n = X.shape[0]
        
        # w must be 1D of length n_features to match X @ w matrix multiply
        # np.zeros(shape) takes a single tuple/int, not two separate args
        w = np.zeros(X.shape[1])
        b = 0.0

        for _ in range(epochs):
            y_hat = X @ w + b
            
            # MSE line removed: loss value is never used inside the loop,
            # only the gradients dw and db drive the updates

            # (2/n) * X.T @ (y_hat - y) is the MSE gradient w.r.t. w
            # n was undefined before — now correctly set to X.shape[0]
            dw = (2/n) * np.dot(X.T, (y_hat - y))
            
            # scalar gradient w.r.t. b: mean of residuals scaled by 2
            db = (2/n) * np.sum(y_hat - y)

            w = w - lr * dw
            b = b - lr * db

        return (np.round(w, 5), round(b, 5))
