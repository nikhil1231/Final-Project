import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

def diag(a):
  return np.array([[a[0], a[1]], [a[1], a[2]]])

def forward(x, w1, w2):
  out = diag(w1) @ x
  out = diag(w2) @ sigmoid(out)
  return out

X = np.random.normal(0, 1, (2, 10))
Y = forward(X, np.random.normal(0, 1, 3), np.random.normal(0, 1, 3))

def loss(y_hat, Y):
  return sum(((y_hat - Y) ** 2).flatten())

def calc_loss(i, j, fixed):
  y_hat = forward(X, [i, j, fixed[0]], [fixed[1], fixed[2], fixed[3]])
  return loss(y_hat, Y)

def create_landscape(axis):
  rand = np.random.normal(0, 1, 4)
  return [[calc_loss(i, j, rand) for j in axis] for i in axis]

axis = np.arange(-10, 10, 0.25)

def plot():
  plt.contour(axis, axis, create_landscape(axis), levels=20)
  plt.show()

def plot_3d():
  fig = plt.figure()
  ax = fig.gca(projection='3d')

  Z = np.array(create_landscape(axis))
  axis_x, axis_y = np.meshgrid(axis, axis)

  ax.plot_surface(axis_x, axis_y, Z, cmap=cm.Spectral_r, linewidth=0)
  plt.show()

# plot()
plot_3d()
