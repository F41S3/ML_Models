import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch

def prediction_cost(t, y):
  N = t.shape[0]

  cost = np.divide(np.power(np.sum(y - t), 2),2*N)

  return cost

def calculate_gradients(X, t, y):
  N = X.shape[0]
  X_t = X.T
  diff = y - t

  # leaving 1 here in the numerator so that we can have fewer multiplication
  # operations total. Leaves learning rate to only be perfromed in LR
  gradients = 1/N * X_t.dot(diff)
  return gradients

def cross_entropy_loss(prediction: np.array, truth: np.array) -> float:
    """
    Compute the cross-entropy loss. Note that since you are provided arrays
    rather than a single points, this is actually the cost function.

    Parameters:
        y: (N, 1) numpy array of predicted probabilities (values between 0 and 1).
        t: (N, 1) numpy array of true labels (0 or 1).

    Returns:
        loss: Cross-entropy loss as a float.
    """
    # We do this to ensure we never hit log(0)
    epsilon = 1e-15
    prediction = np.clip(prediction, epsilon, 1 - epsilon)

    # your code here
    cross_entropy = -truth * np.log(prediction) - (1-truth) * np.log(1-prediction)


    loss = np.mean(cross_entropy)


    return loss

def sigmoid(z: np.array) -> np.array:
    '''
    Given an array of real numbers, apply the sigmoid function element-wise.
    Return:
        A numpy array of the same shape as z.
    '''
    sigma = 1 / (1 + np.exp(-z))

    return sigma

def log_bin_forward(X: np.array, w: np.array, b: float) -> np.array:
    '''
    logistic regression, binary classification, forward pass.
    Parameters:
        X: (N, D) numpy array
        w: (D, 1) numpy array
        b: float
    Return:
        y: (N, 1) numpy array
    '''

    z = np.matmul(X, w) + b
    y = sigmoid(z)  # essentially returns "probabilities" for each feature

    return y

def log_bin_backward(X: np.array, y: np.array, t: np.array, w: np.array, b: float) -> np.array:
    '''
        logistic regression, binary classification, backprop.
        Parameters:
            X: (N, D) numpy array
            y: (N, 1) numpy array
            t: (N, 1) numpy array
            w: (D, 1) numpy array
            b: float
        Return:
            dw: (D, 1) numpy array
            db: float
    '''
    N = X.shape[0]

    dw = - np.matmul(X.T, (y - t)) / N
    db = - np.sum(y - t) / N

    return dw, db

def computeParams(dataset):
    """
    Given a set of points {x1, x2, ..., xn} as dataset,
    compute the mean mu and standard deviation sigma of the dataset.
    """
    mean = np.mean(dataset)
    std_dev = np.std(dataset)
    # Keep in mind that while you can use np functions to compute these,
    # you need to exactly how to do this by hand if required to do so.

    return mean, std_dev

def computePDF(x, mean, std_dev):
  """
    This is the univariate gaussian function
    Computes the probability density of a point x under a Gaussian distribution
    g(x, mu, sigma).

    Args:
      x: The point at which to evaluate the probability density.
      mu: The mean of the Gaussian distribution.
      sigma: The standard deviation of the Gaussian distribution.

    Returns:
      The probability density of x under the Gaussian distribution.
  """

  # YOUR CODE HERE
  coefficient = 1 / (np.sqrt(2 * np.pi) * std_dev)

  exponent = -0.5 * (((x - mean)**2) / (std_dev**2))

  return coefficient * np.exp(exponent)

def computeResponsibilities(x, mean1, mean2, std_dev1, std_dev2, pi1, pi2):
  """
  Compute the responsibility of each Gaussian for the point x, including mixture weights.

  Args:
    x: The point at which to compute responsibilities.
    mu1, sigma1: Mean and standard deviation of Gaussian 1.
    mu2, sigma2: Mean and standard deviation of Gaussian 2.
    pi1, pi2: Mixture weights for Gaussian 1 and Gaussian 2.

  Returns:
    r1: The responsibility of Gaussian 1 for the point x.
        In other words the probability that x came from Gaussian 1.
    r2: The responsibility of Gaussian 2 for the point x.
        In other words the probability that x came from Gaussian 2.

    r1=P(g1|x)=exp(−12(x−μ1σ1))/π1exp(−12(x−μ1σ1))π1+exp(−12(x−μ2σ2))π2
    r2=P(g2|x)=exp(−12(x−μ2σ2))/π2exp(−12(x−μ1σ1))π1+exp(−12(x−μ2σ2))π2
  """
  P_x_g1 = computePDF(x, mean1, std_dev1) * pi1
  P_x_g2 = computePDF(x, mean2, std_dev2) * pi2

  # YOUR CODE HERE
  # NOTE: CAN EASILY BE OPTIMIZED BY MAKING A DIFF VERSION OF COMPUTEPDF (dont need 1/sqrt() due to it cancelling)
  r1 = (P_x_g1) / (P_x_g1 + P_x_g2)
  r2 = (P_x_g2) / (P_x_g1 + P_x_g2)

  return r1, r2


def computeWeightedParams(dataset, weights):
  """
    Given a set of points {x1, x2, ..., xn} as dataset,
    compute the weighted mean mu and standard deviation sigma of the dataset.
  """

  sum = np.sum(weights)
  weighted_mean = np.dot(weights, dataset) / sum

  std_dev = np.sqrt(np.dot(weights, (dataset - weighted_mean)**2) / sum)

  return weighted_mean, std_dev

def calculate_rvals(data, means, covariances, proportions):
    n_points = data.shape[0]
    n_gaussians = len(means)
    rvals = np.zeros((n_points, n_gaussians))

    for i in range(n_gaussians):
        diff = data - means[i]
        cov_inv = np.linalg.inv(covariances[i])
        exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
        normalization = 1 / (2 * np.pi * np.sqrt(np.linalg.det(covariances[i])))
        rvals[:, i] = proportions[i] * normalization * np.exp(exponent)

    rvals /= np.sum(rvals, axis=1, keepdims=True)
    return rvals

# Function to mix colors based on responsibilities
def mix_colors(rvals, colors):
    mixed_colors = np.dot(rvals, colors)
    return mixed_colors

def plot_dataset_and_gaussians(data, means=None, covariances=None, proportions=None,
                               color_mode='uniform', plot_data=True, plot_gaussians=True):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor('black')  # Set background to black

    # Define colors for each Gaussian
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Red, Green, Blue
    if len(means) > 3 :
      color_mode = 'uniform'
    if len(means) == 2:
      colors = np.array([[1,0,0], [0,0,1]])
    if len(means) == 1:
      color_mode = 'uniform'


    if plot_data:
        if color_mode == 'uniform':
            ax.scatter(data[:, 0], data[:, 1], s=5, alpha=0.6, color='cyan', label='Data Points')
        elif color_mode == 'responsibilities' and means is not None and covariances is not None and proportions is not None:
            rvals = calculate_rvals(data, means, covariances, proportions)
            mixed_colors = mix_colors(rvals, colors)
            ax.scatter(data[:, 0], data[:, 1], s=5, alpha=0.6, c=mixed_colors, label='Data Points')

    if plot_gaussians and means is not None and covariances is not None:
        for mean, cov in zip(means, covariances):
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
            width, height = 2 * np.sqrt(eigenvalues)
            ellipse = Ellipse(
                xy=mean, width=width, height=height, angle=angle,
                edgecolor='lime', facecolor='none', lw=2
            )
            ax.add_patch(ellipse)

    # Add legend for the Gaussian responsibility colors
    if color_mode == 'responsibilities' and means is not None and covariances is not None and proportions is not None:
      if len(means) == 3:
        legend_elements = [
            Patch(facecolor='red', edgecolor='white', label='Gaussian 1'),
            Patch(facecolor='green', edgecolor='white', label='Gaussian 2'),
            Patch(facecolor='blue', edgecolor='white', label='Gaussian 3')
        ]
        ax.legend(handles=legend_elements, facecolor='white', edgecolor='yellow', fontsize=10)

      if len(means) == 2:
        legend_elements = [
            Patch(facecolor='red', edgecolor='white', label='Gaussian 1'),
            Patch(facecolor='blue', edgecolor='white', label='Gaussian 2')
        ]
        ax.legend(handles=legend_elements, facecolor='white', edgecolor='yellow', fontsize=10)


    ax.set_xlabel("X-axis", fontsize=12, color='black')
    ax.set_ylabel("Y-axis", fontsize=12, color='black')
    ax.set_title("2D Gaussian Mixture and Data Points", fontsize=14, fontweight='normal', color='black')
    plt.grid(True, linestyle='--', alpha=0.3, color='white')
    plt.tight_layout()
    plt.show()