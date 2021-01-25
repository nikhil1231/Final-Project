import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# 6
np.random.seed(6)

WEIGHTS_DIST = 'first'
MODEL_FIXED_SAME = False
AXIS_SIZE = 15

sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

class TwoLayerNet:
  def __init__(self, w1, w2):
    self.w1 = w1
    self.w2 = w2

  def forward(self, x):
    self.a1 = sigmoid(self.w1 @ x)
    self.a2 = self.w2 @ self.a1
    return self.a2

  def backward(self, x, y, lr):
    error = self.a2 - y

    dw2 = error @ self.a1.T
    dw1 = self.w2.T @ error * self.a1 * (1 - self.a1) @ x.T

    # These values are always fixed, so set grad to 0
    dw1[1,1] = 0
    dw2[1,1] = 0

    if WEIGHTS_DIST == 'first':
      self.w1 -= lr * dw1 / len(x)
    elif WEIGHTS_DIST == 'second':
      self.w2 -= lr * dw2 / len(x)

    # Need to make sure these are the same?
    diff = self.w1[0,1] - self.w1[1,0]
    self.w1[0,1] -= diff / 2
    self.w1[1,0] += diff / 2


def diag(a):
  return np.array([[a[0], a[1]], [a[1], a[2]]])

'''
  Distribute parameters i and j into a pair of 2x2 matrices, to form the weights.
'''
def form_weights(i, j, fixed, dist='first'):
  weights = None
  if dist == 'first':
    weights = [i, j, fixed[0]], [fixed[1], fixed[2], fixed[3]]
  elif dist == 'second':
    # This does not result in non-convexity
    weights = [fixed[0], fixed[1], fixed[2]], [i, j, fixed[3]]
  elif dist == 'equal':
    # weights = [i, fixed[0], fixed[1]], [j, fixed[2], fixed[3]]
    weights = [fixed[0], i, fixed[1]], [fixed[2], j, fixed[3]]
  elif dist == 'rotational':
    weights = [0, i, 0], [0, j, 0]

  return list(map(lambda x: diag(x), weights))

def forward(x, w):
  out = w[0] @ x
  out = w[1] @ sigmoid(out)
  return out

def loss(y_hat, Y):
  return sum(((y_hat - Y) ** 2).flatten()) / 2

def train(epochs, m, X, Y, lr):
  sgd_path = []
  losses = []
  for _ in range(epochs):
    y_hat = m.forward(X)

    _loss = loss(y_hat, Y)
    losses.append(_loss)

    m.backward(X, Y, lr)

    sgd_path.append((m.w1[0,0], m.w1[0,1], _loss))

  return sgd_path, losses

def calc_loss(i, j, fixed, Y):
  y_hat = forward(X, form_weights(i, j, fixed, WEIGHTS_DIST))
  return loss(y_hat, Y)

def create_landscape(axis, fixed, Y):
  return [[calc_loss(i, j, fixed, Y) for i in axis] for j in axis]

axis = np.arange(-AXIS_SIZE, AXIS_SIZE, AXIS_SIZE/100)

'''
  Plot 2D contours of the loss landscape
'''
def plot(fixed, Y, paths=None):
  plt.contour(axis, axis, create_landscape(axis, fixed, Y), levels=20, cmap=cm.terrain)
  if paths:
    for path in paths:
      params = list(zip(*path))
      xs, ys = params[0], params[1]
      plt.plot(xs, ys, color='red')
      plt.scatter(xs[-1], ys[-1], color='red', marker='o')

  plt.xlim([-AXIS_SIZE, AXIS_SIZE])
  plt.ylim([-AXIS_SIZE, AXIS_SIZE])
  plt.show()

'''
  Plot 3D contours of the loss landscape
'''
def plot_3d(fixed, Y, paths=None):
  fig = plt.figure()
  ax = fig.gca(projection='3d')

  Z = np.array(create_landscape(axis, fixed, Y))
  axis_x, axis_y = np.meshgrid(axis, axis)

  ax.plot_surface(axis_x, axis_y, Z, cmap=cm.terrain, linewidth=0, alpha=0.7)
  if paths:
    for path in paths:
      ax.plot(*zip(*path), color='red')
      ax.scatter(*path[-1], marker='o', color='red')

  plt.xlim([-AXIS_SIZE, AXIS_SIZE])
  plt.ylim([-AXIS_SIZE, AXIS_SIZE])
  plt.show()


'''
  Plot chart of loss over epochs
'''
def plot_losses(losses, epochs):
  epoch_axis = np.arange(1, epochs + 1)
  plt.plot(epoch_axis, losses)
  plt.show()

if __name__ == "__main__":
  rand = np.random.normal(0, 1, 6)
  fixed = rand[2:]

  X = np.random.normal(0, 1, (2, 10))
  Y = forward(X, form_weights(rand[0], rand[1], fixed))

  epochs = 10000
  lr = 0.1

  num_paths = 3
  sgd_paths = []

  # Model fixed weights set to same as Y, change to random to achieve non-convexity
  model_fixed = fixed if MODEL_FIXED_SAME else np.random.normal(0, 1, 4)

  for _ in range(num_paths):
    rand_init = (np.random.rand(2) * 2 - 1) * AXIS_SIZE
    model = TwoLayerNet(*form_weights(*rand_init, model_fixed, WEIGHTS_DIST))

    sgd_paths.append(train(epochs, model, X, Y, lr)[0])

  # plot_losses(losses, epochs)
  plot(model_fixed, Y, sgd_paths)
  plot_3d(model_fixed, Y, sgd_paths)
