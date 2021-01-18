import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# 4, 6, 14
np.random.seed(14)

WEIGHTS_DIST = 'first'

sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

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

def create_landscape(axis, Y):
  rand = np.random.normal(0, 1, 4)
  return [[calc_loss(i, j, rand, Y) for j in axis] for i in axis]

axis = np.arange(-10, 10, 0.25)

def plot(Y):
  plt.contour(axis, axis, create_landscape(axis, Y), levels=20)
  plt.show()

def plot_3d(Y):
  fig = plt.figure()
  ax = fig.gca(projection='3d')

  Z = np.array(create_landscape(axis, Y))
  axis_x, axis_y = np.meshgrid(axis, axis)

  ax.plot_surface(axis_x, axis_y, Z, cmap=cm.Spectral_r, linewidth=0)
  plt.show()

if __name__ == "__main__":
  rand = np.random.normal(0, 1, 6)

  X = np.random.normal(0, 1, (2, 10))
  Y = forward(X, form_weights(rand[0], rand[1], rand[2:]))

  # plot(Y)
  plot_3d(Y)
