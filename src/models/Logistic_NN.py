import numpy as np

from src.models.lib.helper_functions import cross_entropy_loss


def sigmoid(x):
  x = x - np.max(x)
  return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
  x = x - np.max(x)
  return np.exp(-x) / (1 + np.exp(-x))**2

"""
ran into overflow issues
URL: https://www.geeksforgeeks.org/the-role-of-softmax-in-neural-networks-detailed-explanation-and-applications/
"""
def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=0)


def softmax_derivative(x):
  return 1

"""
activation function
slides not posted
used this source
URL: https://www.geeksforgeeks.org/relu-activation-function-in-deep-learning/

"""
def relu(x):
  # must be element wise and return a vector
  output = np.maximum(0, x)
  return output

def relu_derivative(x):
  output = np.maximum(0, 1)
  return output

def tanh(x):
  return np.tanh(x)

def tanh_derivative(x):
  return 1 - np.tanh(x)**2

activation_functions = {
  "sigmoid": (sigmoid, sigmoid_derivative),
  "softmax": (softmax, softmax_derivative),
  "relu": (relu, relu_derivative),
  "tanh": (tanh, tanh_derivative)
}

class Layer:
  def __init__(self, hidden_nodes: int, activation: str) -> None:
    self.nodes = hidden_nodes
    self.act, self.der = activation_functions[activation]

class NeuralNet:
  def __init__(self, layers: list[Layer]) -> None:
    self.layers = layers
    self.weights = []
    self.biases = []

  def init_weights(self, X):
    """
    Initialize weights and biases for each layer based on input data.

    Parameters:
        X: (N, D) numpy array of input data
    """

    # In between each layer you have a weight matrix and a bias vector.

    # you should initialize the weights using np.random.randn and initalize
    # the biases to either 0 or randn. I've provided a bit of structure to
    # help you out, but if you want to do this differently, you can rewrite
    # the whole method, you don't have to use this structure.

    # Initialize weights and biases for each layer based on input data.
    N, D = X.shape

    # Input layer to first hidden layer
    n_inputs = D
    n_outputs = self.layers[0].nodes
    self.weights.append(np.random.randn(n_inputs, n_outputs))
    self.biases.append(np.zeros((1, n_outputs)))

    for i in range(1, len(self.layers)):
      # Hidden layers to output layer or next hidden layer
      n_inputs = self.layers[i - 1].nodes
      n_outputs = self.layers[i].nodes
      self.weights.append(np.random.randn(n_inputs, n_outputs))
      self.biases.append(np.zeros((1, n_outputs)))
    """
    self.weights.append(np.random.randn(D, self.layers[0].nodes))
    self.biases.append(np.random.randn(self.layers[0].nodes, 1))

    for i in range(1, len(self.layers)):
      self.weights.append(np.random.randn(self.layers[i-1].nodes, self.layers[i].nodes))
      self.biases.append((np.random.randn(self.layers[i].nodes, 1)))
    """

  def one_hot_encode(self, t, num_classes):
    """
    One-hot encode the target variable t.

    Parameters:
        t: (N,) numpy array of class labels
        num_classes: the number of unique classes

    Returns:
        one_hot: (N, num_classes) numpy array where each row is one-hot encoded

    URL: https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-one-hot-encoded-array-in-numpy

    """
    # creates array that matches onehot definition for final loss checks
    one_hot = np.zeros((t.shape[0], num_classes))
    one_hot[np.arange(t.size), t] = 1
    return one_hot

  def forward_pass(self, X):
    """
    Perform a forward pass through the network.

    Parameters:
        X: (N, D) numpy array of input data

    Returns:
        y: (N, M) numpy array of network output after the final layer
    """
    a = X
    self.activations = [a]
    self.logits = []

    for layer, W, b in zip(self.layers, self.weights, self.biases):
      tmp = np.dot(a, W)
      z = tmp + b
      a = layer.act(z)

      self.logits.append(z)
      self.activations.append(a)

    self.y = a
    return self.y

  def backward_pass(self, X, t):
    """
    Perform a backward pass through the network to compute gradients.

    Parameters:
        X: (N, D) numpy array of input data
        t: (N, num_classes) numpy array of one-hot encoded target labels

    Returns:
        w_grads: list of weight gradients for each layer
        b_grads: list of bias gradients for each layer
    """
    N = X.shape[0]

    dy = self.y - t

    w_grads = []
    b_grads = []

    for i in reversed(range(len(self.layers))):
      layer = self.layers[i]
      z = self.logits[i]
      y_curr = self.activations[i]


      dw = np.dot(y_curr.T, dy)

      db = np.sum(dy, axis=0, keepdims=True)

      w_grads.append(dw)
      b_grads.append(db)


      if i > 0:
        # if not first layer, propagate error
        W = self.weights[i]
        dy = np.dot(dy, W.T) * layer.der(y_curr)


    return w_grads[::-1], b_grads[::-1]

  def update_weights(self, w_grads, b_grads, learning_rate):
    """
    Update the weights and biases of the network using the computed gradients.

    Parameters:
      w_grads: list of weight gradients for each layer
      b_grads: list of bias gradients for each layer
      learning_rate: float, the learning rate for gradient descent
    """
    for i in range(len(self.layers)):
      dW, db = w_grads[i], b_grads[i]

      self.weights[i] -= dW * learning_rate / len(w_grads)
      self.biases[i] -= db * learning_rate / len(b_grads)

  def fit(self, X, t, epochs, learning_rate):
    """
    Train the neural network on the given data.

    Parameters:
      X: (N, D) numpy array of input data
      t: (N,) numpy array of target class labels
      epochs: int, number of training epochs
      learning_rate: float, the learning rate for gradient descent

    Returns:
      loss_array: list of loss values computed at each epoch
    """

    num_classes = np.unique(t).size


    t = t.astype(int)  # ensure ints
    t = t.reshape(-1)  # flatten for one-hot-encode function



    # use function from above here.
    t_one_hot = self.one_hot_encode(t, num_classes)
    num_classes = np.unique(t).size
    self.init_weights(X)

    N = len(t)
    loss_array = []
    for _ in range(epochs):
      # your code here (~3-5 lines)
      y = self.forward_pass(X)
      w_grads, b_grads = self.backward_pass(X, t_one_hot)
      self.update_weights(w_grads, b_grads, learning_rate)

      # compute loss

      loss_array.append(cross_entropy_loss(y, t_one_hot))

    return loss_array

  def predict(self, X):
    """
    Predict class labels for the given input data.

    Parameters:
        X: (N, D) numpy array of input data

    Returns:
        predictions: (N,) numpy array of predicted class labels
    """

    y_pred = self.forward_pass(X)

    # convert from one-hot back to classes
    return np.argmax(y_pred, axis=1)