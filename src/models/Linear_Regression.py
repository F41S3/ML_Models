import numpy as np
from src.models.lib.helper_functions import calculate_gradients

"""
URL: https://datagy.io/python-numpy-normalize/
Source: Datagy
Explanation: Kept receiving overflow errors for large paramater values, normalizing
cost function data to minimize these errors for further exploraiton.
"""
def prediction_cost(t, y):
  N = t.shape[0]

  cost = np.divide(np.power(np.sum(y - t), 2),2*N)

  return cost

def LR(X, t, lr, iterations):
  # Ensure X is a 2D array. If X is 1D (e.g., shape (N,)), reshape it to (N, 1)
  if X.ndim == 1:
    X = X.reshape(-1, 1)

  # Adding 1 column in X for bias
  b = np.ones((len(X), 1))             # shape : (N, 1)
  X_b = np.append(X, b, axis=1)        # shape : (N, features + 1)
  weights = np.zeros((X_b.shape[1], 1))  # shape : (features + 1, 1)
  t = t.reshape(-1, 1)                 # shape : (N, 1)
  cost_list = []

  for _ in range(iterations):
    # calculate current predictions
    # X_b (input features with bias)
    # and w (learned weights/parameters)
    y = np.dot(X_b, weights)

    # find prediction cost
    cost = prediction_cost(t, y)

    # compute partial derivatives (aka gradients in a vector)
    gradients = calculate_gradients(X, t, y)

    # update the weights
    weights = weights - lr * gradients

    # append the value to the cost list
    cost_list.append(cost)

  y = np.dot(X_b, weights)
  return cost_list, y, weights