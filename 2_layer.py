import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# 6, 8
# np.random.seed(1)

WEIGHTS_DIST = 'equal'
MODEL_FIXED_SAME = True
AXIS_SIZE = 15

parameter_positions = {
  'first': [(0, 0, 0), (0, 0, 1)],
  'second': [(1, 0, 0), (1, 0, 1)],
  'equal': [(0, 0, 1), (1, 0, 1)],
}

sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
inv_sigmoid = lambda y: np.log(y / (1.0 - y))

class TwoLayerNet:
  def __init__(self, w0, w1, w2=np.array([None])):
    self.w = [w0, w1, w2]

  def forward(self, x):
    self.a1 = sigmoid(self.w[0] @ x)
    self.a2 = self.w[1] @ self.a1
    if self.w[2].any() != None:
      self.a3 = self.w[2] @ sigmoid(self.a2)
      return self.a3
    return self.a2

  def backward(self, x, y, lr):

    dw = [None] * 2

    # w[2] will always be fixed, so we can "remove" it before calculating grads
    if self.w[2].any() != None:
      inv = np.linalg.inv(self.w[2])
      y = inv_sigmoid(inv @ y)

    error = self.a2 - y

    dw[1] = error @ self.a1.T
    dw[0] = self.w[1].T @ error * self.a1 * (1 - self.a1) @ x.T

    # Only update weights for non-fixed parameters
    if WEIGHTS_DIST != 'rotational':
      pos = parameter_positions[WEIGHTS_DIST]
      for _pos in pos:
        self.w[_pos[0]][_pos[1], _pos[2]] -= lr * dw[_pos[0]][_pos[1], _pos[2]] / len(x)

        # If the same parameter appears twice on the diagonal, fix them to be the same
        if _pos[1:3] == (0, 1):
          self.w[_pos[0]][1, 0] -= lr * dw[_pos[0]][1, 0] / len(x)

          diff = self.w[_pos[0]][0, 1] - self.w[_pos[0]][1, 0]
          self.w[_pos[0]][0, 1] -= diff / 2
          self.w[_pos[0]][1, 0] += diff / 2


def diag(a):
  return np.array([[a[0], a[1]], [a[1], a[2]]])

def forward_diag(a):
  return np.array([[a[0], a[1]], [a[2], a[0]]])

'''
  Distribute parameters i and j into 2x2 matrices, to form the weights.
'''
def form_weights(i, j, fixed):
  weights = [[fixed[0], fixed[1], fixed[2]], [fixed[3], fixed[4], fixed[5]]]

  if WEIGHTS_DIST == 'rotational':
    weights = [np.cos(i), -np.sin(i), np.sin(i)], [np.cos(j), -np.sin(j), np.sin(j)]
    return list(map(lambda x: forward_diag(x), weights))
  else:
    pos = parameter_positions[WEIGHTS_DIST]
    weights[pos[0][0]][pos[0][1] + pos[0][2]] = i
    weights[pos[1][0]][pos[1][1] + pos[1][2]] = j

    if WEIGHTS_DIST == 'equal' or WEIGHTS_DIST == 'second':
      weights += [[fixed[6], fixed[7], fixed[8]]]

  return list(map(lambda x: diag(x), weights))

def forward(x, w):
  out = w[0] @ x
  out = w[1] @ sigmoid(out)
  if len(w) > 2:
    out = w[2] @ sigmoid(out)
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

    pos = parameter_positions[WEIGHTS_DIST]
    sgd_path.append((m.w[pos[0][0]][pos[0][1], [pos[0][2]]], m.w[pos[1][0]][pos[1][1], [pos[1][2]]], _loss))

    m.backward(X, Y, lr)

  return sgd_path, losses

def calc_loss(i, j, fixed, Y):
  y_hat = forward(X, form_weights(i, j, fixed))
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
  rand = np.random.normal(0, 1, 12)
  fixed = rand[2:]

  X = np.random.normal(0, 1, (2, 10))
  Y = forward(X, form_weights(rand[0], rand[1], fixed))

  # fixed batch vs dist, range of values, orthoganol matices, adding another layer to 'equal' dist.
  epochs = 50000
  lr = 0.1

  num_paths = 3
  sgd_paths = []

  # Model fixed weights set to same as Y, change to random to achieve non-convexity
  model_fixed = fixed if MODEL_FIXED_SAME else np.random.normal(0, 1, 12)

  for _ in range(num_paths):
    rand_init = (np.random.rand(2) * 2 - 1) * AXIS_SIZE
    model = TwoLayerNet(*form_weights(*rand_init, model_fixed))

    sgd_paths.append(train(epochs, model, X, Y, lr)[0])

  # plot_losses(losses, epochs)
  # plot(model_fixed, Y, sgd_paths)
  # plot_3d(model_fixed, Y)
  plot_3d(model_fixed, Y, sgd_paths)
