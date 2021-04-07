import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def f_convex(x, y):
  return x**2 + y**2

def f_non_convex(x, y):
  return x**4 - x**2 + x/10 + y**2

def plot(convex):
  fig = plt.figure()
  ax = fig.gca(projection='3d')

  x = y = np.arange(-1.0, 1.0, 0.05)
  X, Y = np.meshgrid(x, y)

  f = f_convex if convex else f_non_convex

  zs = np.array(f(np.ravel(X), np.ravel(Y)))
  Z = zs.reshape(X.shape)

  ax.plot_surface(X, Y, Z, cmap=cm.terrain, linewidth=0, alpha=0.7)

  if convex:
    ax.scatter(0, 0, 0, marker='x', s=150, color='black')
  else:
    ax.scatter([-0.731, 0.681], [0, 0], [-0.27, -0.15], marker='x', s=150, color='black')
    ax.scatter(0.681, 0, -0.15, marker='x', s=150, color='black')

  plt.xlim([-1, 1])
  plt.ylim([-1, 1])
  plt.xlabel('x')
  plt.ylabel('y')
  ax.set_zlabel('z')
  plt.show()

plot(convex=False)
