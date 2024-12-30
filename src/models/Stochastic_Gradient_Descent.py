import numpy as np

from src.models.Linear_Regression import calculate_gradients
from src.models.lib.helper_functions import prediction_cost


def stochastic_gradient_descent(X, t, lr, iterations, m):
  if X.ndim == 1:
        X = X.reshape(-1, 1)

  b = np.ones((len(X), 1))             # shape : (N, 1)
  X_b = np.append(X, b, axis=1)        # shape : (N, features + 1)
  weights = np.zeros((X_b.shape[1], 1))  # shape : (features + 1 ,1)
  t = t.reshape(-1, 1)                 # shape : (N, 1)
  cost_list = []


  for _ in range(iterations):
    # your code here (~ 5-6 lines)
    batch_indexes = np.random.choice(len(X_b), m)
    X_batch = X_b[batch_indexes]
    t_batch = t[batch_indexes]

    # same computation as before only with a tiny batch size
    y = np.dot(X_batch, weights)

    # compute cost as before
    cost = prediction_cost(t_batch, y)

    # compute gradient as before
    gradients = calculate_gradients(X_batch, t_batch, y)

    # compute updated weights (theta, changed for my personal understanding)
    weights = weights - lr * gradients

    cost_list.append(cost)

  y = np.dot(X_b, weights)

  return cost_list, y, weights