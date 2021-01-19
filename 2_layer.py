import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# 4, 6, 14
# np.random.seed(10)

WEIGHTS_DIST = 'first'
AXIS_SIZE = 10

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
    self.w1[0,1] = self.w1[1,0]


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

def plot(fixed, Y, path=None):
  plt.contour(axis, axis, create_landscape(axis, fixed, Y), levels=20)
  if path:
    params = list(zip(*path))
    plt.plot(*params[:2], color='red')
    plt.scatter(*path[-1][:2], color='red', marker='o')
  plt.show()

def plot_3d(fixed, Y, path=None):
  fig = plt.figure()
  ax = fig.gca(projection='3d')

  Z = np.array(create_landscape(axis, fixed, Y))
  axis_x, axis_y = np.meshgrid(axis, axis)

  ax.plot_surface(axis_x, axis_y, Z, cmap=cm.terrain, linewidth=0, alpha=0.7)
  if path:
    ax.plot(*zip(*path), color='red')
    ax.scatter(*path[-1], marker='o', color='red')
  plt.show()

def plot_losses(losses, epochs):
  epoch_axis = np.arange(1, epochs + 1)
  plt.plot(epoch_axis, losses)
  plt.show()

if __name__ == "__main__":
  rand = np.random.normal(0, 1, 6)
  fixed = rand[2:]

  X = np.random.normal(0, 1, (2, 10))
  Y = forward(X, form_weights(rand[0], rand[1], fixed))

  rand_init = (np.random.rand(2) * 2 - 1) * AXIS_SIZE
  model = TwoLayerNet(*form_weights(*rand_init, fixed, WEIGHTS_DIST))

  epochs = 10000

  sgd_path, losses = train(epochs, model, X, Y, 0.1)

  # Fixed weights set to same as Y, change to random to achieve non-convexity
  # plot_losses(losses, epochs)
  plot(fixed, Y, sgd_path)
  plot_3d(fixed, Y, sgd_path)
