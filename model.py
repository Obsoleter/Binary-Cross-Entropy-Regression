from typing import Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import minmax_scale, scale
from sklearn.model_selection import train_test_split, GridSearchCV


# Tools
def to_class(z: np.ndarray) -> np.ndarray:
    """Returns V"""

    # Copy
    v = z.copy()

    # Turn
    v[v < 0.5] = 0
    v[v >= 0.5] = 1

    return v

def accuracy(v: np.ndarray, v_pred: np.ndarray) -> float:
    count = np.absolute(v - v_pred).sum()
    return 1 - count / v.size


# Binary Cross Entropy
EPS = 1e-15

def binary_cross_entropy(z: np.ndarray, z_pred: np.ndarray) -> np.ndarray:
    # Set up
    z_pred = z_pred.copy()

    # Handle input
    z_pred[z_pred == 0] = EPS
    z_pred[z_pred == 1] = 1 - EPS

    # Estimate function value
    return z * np.log(z_pred) + (1 - z) * np.log(1 - z_pred)

def bce_derivative(z: np.ndarray, z_pred: np.ndarray) -> np.ndarray:
    # Set up
    z_pred = z_pred.copy()

    # Handle input
    z_pred[z_pred == 0] = EPS
    z_pred[z_pred == 1] = 1 - EPS

    # Estimate function value
    return z / z_pred - (1 - z) / (1 - z_pred)


# Sigmoid
def sigmoid(y: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-y))

def sigmoid_derivative(y: np.ndarray) -> np.ndarray:
    return sigmoid(y) * (1 - sigmoid(y))


# Gradient
def derivative(z: np.ndarray, y_pred: np.ndarray, X_coef: Union[np.ndarray, float] = 1.0) -> float:
    # Handle Input
    z_pred = sigmoid(y_pred)

    # Derivatives
    logd = bce_derivative(z, z_pred)
    sigd = sigmoid_derivative(y_pred)
    mul = logd * sigd * X_coef

    return -1 * mul.mean()


# Training
class EntropyRegression:
    def __init__(self, w1 = 1, w2 = 1, b = 0, epochs: int = 5, step: float = 1e-1) -> None:
        # Coefs
        self.w1 = w1
        self.w2 = w2
        self.b = b

        # Params
        self.epochs = epochs
        self.step = step

    def fit(self, X: np.ndarray, z: np.ndarray, verbose: bool = False):
        acc = None

        for epoch in range(1, self.epochs + 1):
            # Input
            y_pred = self.predict(X)
            z_pred = sigmoid(y_pred)
            v_pred = to_class(z_pred)

            # Estimate cost
            cost = -1 * binary_cross_entropy(z, z_pred).mean()
            acc = accuracy(z, v_pred)

            # Get gradients
            w1g = derivative(z, y_pred, X[:, 0])
            w2g = derivative(z, y_pred, X[:, 1])
            bg = derivative(z, y_pred)

            # Make steps
            self.w1 -= w1g * self.step
            self.w2 -= w2g * self.step
            self.b -= bg * self.step

            # Print info
            if verbose:
                print(f"Epoch: {str(epoch).ljust(10)}Loss: {str(cost).ljust(25)}Accuracy: {str(acc).ljust(25)}W1: {str(self.w1).ljust(25)}W2: {str(self.w2).ljust(25)}B: {str(self.b).ljust(25)}")
            
        return acc

    def predict(self, X: np.ndarray):
        x1 = X[:, 0] * self.w1
        x2 = X[:, 1] * self.w2
        return x1 + x2 + self.b

    def reset(self, w1 = 1, w2 = 1, b = 0):
        self.w1 = w1
        self.w2 = w2
        self.b = b

    def score(self, X: np.ndarray, z: np.ndarray):
        self.reset()
        return self.fit(X, z)

    def get_params(self, deep: bool = True):
        return {
            "epochs": self.epochs,
            "step": self.step
        }

    def set_params(self, epochs, step):
        self.epochs = epochs
        self.step = step
        return self


# Get Data
data = pd.read_csv("output.csv")

# Decrease price
data['Price'] /= 1000

# Y - Regression Output
# Z - Probability
# V - Class (0 or 1)
X = np.column_stack((
    data['Price'],
    data['Pink slip'],
))

v = np.array(data['Sold?'])

# Scale
# X = scale(X)

# Split
X_train, X_test, v_train, v_test = train_test_split(X, v)


# Cross Validation
# Find the best step for a small epochs argument
grid = {
    "epochs": [2, 5, 10],
    "step": np.arange(0.01, 0.2, 0.01)
}
cv = GridSearchCV(EntropyRegression(), grid, cv=5)
cv.fit(X_train, v_train)
print(cv.best_params_)


# Regression
regr = EntropyRegression(**cv.best_params_)
# regr = EntropyRegression(epochs=50, step=1.5e-1)
regr.fit(X_train, v_train, True)


# Confusion matrix
z_pred = sigmoid(regr.predict(X_test))
v_pred = to_class(z_pred)

acc = accuracy(v_test, v_pred)
print(f"Testing Accuracy: {acc}")

ConfusionMatrixDisplay.from_predictions(v_test, v_pred)
plt.show()