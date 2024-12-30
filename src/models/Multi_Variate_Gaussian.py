import numpy as np

from src.models.lib.helper_functions import plot_dataset_and_gaussians


class EM2D:
  def __init__(self):
    """
      Initialize the EM algorithm for 2D data with default tolerance and iteration settings.

      Parameters:
      - tol (float): Convergence tolerance.
      - max_iter (int): Maximum number of iterations.
    """
    self.tol = None
    self.max_iter = None
    self.mu1 = None
    self.sigma1 = None
    self.mu2 = None
    self.sigma2 = None
    self.w = None
    self.pi1 = None
    self.pi2 = None

  def initialize_params(self, dataset):
    """
      Initialize parameters based on the range of the 2D dataset.
    """
    n, d = dataset.shape
    # We've adjusted this function for 2D data
    min_vals, max_vals = np.min(dataset, axis=0), np.max(dataset, axis=0)

    self.mu1 = np.random.uniform(min_vals, max_vals)
    self.mu2 = np.random.uniform(min_vals, max_vals)

    data_range = max_vals - min_vals
    self.sigma1 = np.eye(d) * (np.mean(data_range) / 4)
    self.sigma2 = np.eye(d) * (np.mean(data_range) / 4)

    self.pi1 = 0.5
    self.pi2 = 0.5

    self.w = np.ones((n, 2)) / 2

  def gaussian_pdf(self, x, mean, cov):
    """
      Calculate the Gaussian probability density function in 2D.
    """

    cov_inv = np.linalg.inv(cov)  # use numpy to compute the inverse

    # YOUR CODE HERE TO COMPLETE THE COMPUTATION OF THE GAUSSIAN

    diff = x - mean

    exponent = -0.5 * np.sum(np.dot(diff, cov_inv) * diff, axis=1)

    # val = np.array([np.dot(col.T, cov_inv) for col in dataset])
    #
    # val2 = np.array([np.dot(col.T, cov_inv) for col in val])
    #
    # val3 = val2 * diff
    # # print(val.shape, val2.shape, val3.shape, diff.shape)
    #
    # exponent = -0.5 * val3

    result = np.exp(exponent) / (2 * np.pi * np.sqrt(np.linalg.det(cov)))
    return result

  def e_step(self, dataset):
    """
      Perform the E-step: compute responsibilities for each data point.
    """
    n = len(dataset)
    responsibilities = np.zeros((n, 2))

    # YOUR CODE GOES HERE

    P_x_g1 = self.pi1 * self.gaussian_pdf(dataset, self.mu1, self.sigma1)
    P_x_g2 = self.pi2 * self.gaussian_pdf(dataset, self.mu2, self.sigma2)

    denominator = P_x_g1 + P_x_g2
    responsibilities = np.column_stack((P_x_g1 / denominator, P_x_g2 / denominator))
    return responsibilities

  def m_step(self, dataset, responsibilities):
    """
      Perform the M-step: update the parameters based on the responsilities.
    """
    # compute means
    old_mu1 = self.mu1
    old_mu2 = self.mu2
    rsum0 = np.sum(responsibilities[:, 0], axis=0)
    rsum1 = np.sum(responsibilities[:, 1], axis=0)

    rsum = np.sum(responsibilities)

    # compute means's
    self.mu1 = np.sum(dataset * responsibilities[:, 0].reshape(-1,1), axis=0) / rsum0
    self.mu2 = np.sum(dataset * responsibilities[:, 1].reshape(-1,1), axis=0) / rsum1

    # compute pi's
    self.pi1 = rsum0 / responsibilities.shape[0]
    self.pi2 = rsum1 / responsibilities.shape[0]

    # compute covariance
    self.sigma1 = np.cov(dataset.T, aweights=(responsibilities[:, 0]/rsum0).flatten(), bias=True)
    self.sigma2 = np.cov(dataset.T, aweights=(responsibilities[:, 1]/rsum1).flatten(), bias=True)
    return old_mu1, old_mu2

  def fit(self, dataset, tol=1e-4, max_iter=1000, plot_data=False):
    """
      Initialize parameters based on the dataset and run the EM algorithm for 2D data.
    """
    self.initialize_params(dataset)

    for iteration in range(max_iter):
      if (plot_data == True):
        # plot the data in current state with current params
        # print("dataset=", dataset.shape,
        #       "means=", self.mu1.shape, self.mu2.shape,
        #       "variances=", self.sigma1.shape, self.sigma2.shape,
        #       "pies=", self.pi1, self.pi2)
        plot_dataset_and_gaussians(dataset, means=[self.mu1, self.mu2], covariances=[self.sigma1, self.sigma2],
                                   proportions=[self.pi1, self.pi2], color_mode='responsibilities')

      # YOUR CODE HERE
      responsibilities = self.e_step(dataset)

      old_mu1, old_mu2 = self.m_step(dataset, responsibilities)

      # Check for convergence
      if np.linalg.norm(self.mu1 - old_mu1) < tol and np.linalg.norm(self.mu2 - old_mu2) < tol:
        print(f'Converged after {iteration + 1} iterations.')
        break
    else:
      print(f'Did not converge within {max_iter} iterations.')

  def get_parameters(self):
    """
      Return the parameters of the 2D Gaussian distributions.
    """
    return {
      'mu1': np.round(self.mu1, 2),
      'sigma1': np.round(self.sigma1, 2),
      'pi1': np.round(self.pi1, 2),
      'mu2': np.round(self.mu2, 2),
      'sigma2': np.round(self.sigma2, 2),
      'pi2': np.round(self.pi2, 2),
    }
