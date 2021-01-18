import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# 4, 6, 14
np.random.seed(10)

WEIGHTS_DIST = 'first'

sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

class TwoLayerNet:
  def __init__(self, w1, w2):
    self.w1 = w1
    self.w2 = w2

def diag(a):
  return np.array([[a[0], a[1]], [a[1], a[2]]])

def form_weights(i, j, fixed, dist='first'):
  weights = None
  if dist == 'first':
    weights = [i, j, fixed[0]], [fixed[1], fixed[2], fixed[3]]
  elif dist == 'second':
    # This does not result in non-convexity
    weights = [fixed[0], fixed[1], fixed[2]], [i, j, fixed[3]]

  return list(map(lambda x: diag(x), weights))

def forward(x, w):
  out = w[0] @ x
  out = w[1] @ sigmoid(out)
  return out

def loss(y_hat, Y):
  return sum(((y_hat - Y) ** 2).flatten())

def calc_loss(i, j, fixed, Y):
  y_hat = forward(X, form_weights(i, j, fixed, WEIGHTS_DIST))
  return loss(y_hat, Y)

def create_landscape(axis, fixed, Y):
  return [[calc_loss(i, j, fixed, Y) for j in axis] for i in axis]

axis = np.arange(-10, 10, 0.25)

def plot(fixed, Y):
  plt.contour(axis, axis, create_landscape(axis, fixed, Y), levels=20)
  plt.show()

def plot_3d(fixed, Y):
  fig = plt.figure()
  ax = fig.gca(projection='3d')

  Z = np.array(create_landscape(axis, fixed, Y))
  axis_x, axis_y = np.meshgrid(axis, axis)

  ax.plot_surface(axis_x, axis_y, Z, cmap=cm.Spectral_r, linewidth=0)
  plt.show()

if __name__ == "__main__":
  rand = np.random.normal(0, 1, 6)
  fixed = rand[2:]

  X = np.random.normal(0, 1, (2, 10))
  Y = forward(X, form_weights(rand[0], rand[1], fixed))

  rand_init = np.random.normal(0, 1, 2)
  model = TwoLayerNet(*form_weights(*rand_init, fixed, WEIGHTS_DIST))

  # Fixed weights set to same as Y, change to random to achieve non-convexity
  # plot(fixed, Y)
  plot_3d(fixed, Y)
