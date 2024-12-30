import numpy as np
from src.models.Linear_Regression import calculate_gradients
from src.models.lib.helper_functions import prediction_cost

def stochastic_gradient_descent_with_regularization(X, t, lr, iterations, reg_term, dims, batch_size):
  # your code here
  X_d = np.vander(X, dims, increasing=True)
  t = t.reshape(-1, 1)
  weights = np.zeros((X_d.shape[1], 1))
  cost_list = []


  for _ in range(iterations):
    # your code here (~ 5-6 lines)
    batch_indexes = np.random.choice(len(X_d), batch_size)
    X_batch = X_d[batch_indexes]
    t_batch = t[batch_indexes]

    # same computation as before only with a tiny batch size
    y = np.dot(X_batch, weights)

    reg_gradient = reg_term * weights

    reg_cost = np.dot(weights.T, weights)

    # compute cost as before but, now with the added cost of the weights
    cost = prediction_cost(t_batch, y) + reg_cost[0]

    # compute gradient as before
    gradients = calculate_gradients(X_batch, t_batch, y) + reg_gradient

    # compute updated weights (theta, changed for my personal understanding)
    weights = weights - lr * gradients

    cost_list.append(cost)

  y = np.dot(X_d, weights)

  return cost_list, y, weights