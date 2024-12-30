import numpy as np

from src.models.lib.helper_functions import log_bin_forward, cross_entropy_loss


class LogisticRegression():
    def __init__(self):
        self.w = None
        self.b = None


    def fit(self, X: np.array, t: np.array, lr: float, epochs: int):
        # Initialize w and b to random weights
        # I recommend using np.random.randn

        self.w = np.random.randn(X.shape[1], 1)
        self.b = np.random.randn()

        t = t.reshape(-1, 1)
        loss_array = []

        # this is really just a shallow NN with one layer
        for i in range(epochs):
            # Forward
            # def log_bin_forward(X: np.array, w: np.array, b: float) -> np.array:
            y = log_bin_forward(X, self.w, self.b)

            # Loss
            loss = cross_entropy_loss(y, t)
            loss_array.append(loss) # for graphing progess over the # of steps

            # Backward
            # def log_bin_backward(X: np.array, y: np.array, t: np.array, w: np.array, b: float) -> np.array:

            dw, db = log_bin_backward(X, y, t, self.w, self.b)

            # Update
            self.w += dw * lr
            self.b += db * lr

        return loss_array

    def predict(self, X: np.array) -> np.array:

        y = log_bin_forward(X, self.w, self.b)

        return y