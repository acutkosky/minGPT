import numpy as np
import torch
from scipy.stats import ortho_group
from matplotlib import pyplot as plt
from mingpt.acceleration import Acceleration_NonConvex_small

# returns smoothed absolute loss and its derivative
# use smoothed loss so that things with second-order bounds can
# take advantage of smoothness.
def smoothed_absolute_loss(t):
    if t<1:
        return t**2
    else:
        return 2*torch.abs(t)

### optimization problem: linear regression with smoothed absolute loss
# This function returns another function that generates loss values and gradients
# given an iterate value.
def loss_function_factory(dims, noise_magnitude, do_smoothed=True, optimal_point_scaling=10.0, eigenvalue_decay=10.0):
  '''
  creates a loss function for use in optimization experiments.

  arguments:
    dims: number of dimensions in the problem.
    noise_magnitude: amount of noise to add to gradients.
    do_smoothed: whether to use smoothed absolute loss.
    optimal_point_scaling: scales how far the optimum is from the origin.
    eigenvalue_decay: scales how fast the eigenvalues of the the matrix M decay.

  returns:
    a loss function.
  '''

    # produce a somewhat-poorly-conditioned matrix
  if dims != 1:
      random_orthogonal_matrix = ortho_group.rvs(dims)
  else:
      random_orthogonal_matrix = np.array([1.0])
  eigenvalues = np.exp(np.arange(dims)/eigenvalue_decay)
  eigenvalues = eigenvalues/np.max(eigenvalues)

  matrix = np.dot(np.dot(random_orthogonal_matrix, np.diag(eigenvalues)),random_orthogonal_matrix.T)
  matrix = torch.tensor(matrix)

  optimal_point = optimal_point_scaling * np.random.normal(np.zeros(dims), np.ones(dims))
  optimal_point = torch.tensor(optimal_point, requires_grad=False)



  if do_smoothed:
    loss_func = smoothed_absolute_loss
  else:
    loss_func = absolute_loss


  def get_loss(x):
      gap = x - optimal_point
      skewed_gap = torch.matmul(matrix, gap)
      skewed_distance = torch.norm(skewed_gap)
      loss = loss_func(skewed_distance)
      # grad = grad * np.dot(matrix, skewed_gap/np.linalg.norm(skewed_gap))

      # noise = np.random.normal(np.zeros_like(x), np.ones_like(x))
      # noise = np.dot(matrix, noise)

      return loss#, grad + noise_magnitude * noise

  return get_loss

def train(optimizer_factory):
    dims = 100
    iters = 10000
    loss_fn = loss_function_factory(dims=dims, noise_magnitude=0.0, do_smoothed=True, optimal_point_scaling=10.0, eigenvalue_decay=10.0)

    x = torch.zeros(dims,requires_grad=True)
    # opt = torch.optim.SGD([x],lr=1.0)
    opt = optimizer_factory([x], lr=1.0)

    losses = []
    for i in range(iters):
        loss_val = loss_fn(x)
        losses.append(loss_val.item())

        opt.zero_grad()
        loss_val.backward()
        opt.step()

    plt.plot(losses)
    plt.yscale('log')
    plt.xscale('log')

    plt.show()


train(Acceleration_NonConvex_small)
