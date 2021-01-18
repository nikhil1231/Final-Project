import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

rand = np.random.normal(0, 1, (2, 2))

a = [rand[0,0], rand[0,1], rand[1,1]]

sax = lambda a, x: np.matmul([[a[0], a[1]], [a[1], a[2]]], x)
sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

X = np.random.normal(0, 1, (2, 10))
Y = sax(a, X)

def loss(a, b, X, Y):

  out = sax(a, X)
  out = sigmoid(sax(b, out))

  return sum(((out - Y) ** 2).flatten())

def calc_loss(i, j):
  return loss([i, j, a[2]], [i, j, a[2]], X, Y)

def create_landscape(axis):
  return [[calc_loss(i, j) for j in axis] for i in axis]

axis = np.arange(-5, 5, 0.25)

def plot():
  plt.contour(axis, axis, create_landscape(axis), levels=20)
  plt.show()

def plot_3d():
  fig = plt.figure()
  ax = fig.gca(projection='3d')

  Z = np.array(create_landscape(axis))
  axis_x, axis_y = np.meshgrid(axis, axis)

  ax.plot_surface(axis_x, axis_y, Z, cmap=cm.Spectral_r,
                        linewidth=0, antialiased=False)
  plt.show()

# plot()
plot_3d()
