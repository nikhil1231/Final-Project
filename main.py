import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# 4, 6 for good rotational, 7 for uphill rotational
np.random.seed(6)

WEIGHTS_DIST = 'rotational'
MODEL_FIXED_SAME = True
RAND_SD = 2
RAND_DIST = 'uniform'
AXIS_SIZE = 5

parameter_positions = {
  'first': [(0, 0, 0), (0, 0, 1)],
  'second': [(1, 0, 0), (1, 0, 1)],
  'equal': [(0, 0, 1), (1, 0, 1)],
  'rotational': [(0, 0, 0), (1, 0, 0)],
  'skew': [(0, 0, 1), (1, 0, 1)],
}

sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
inv_sigmoid = lambda y: np.log(y / (1.0 - y))

class TwoLayerNet:
  def __init__(self, w):
    self.w = w
    self.is_3_layer = len(w) == 3

  def forward(self, x):
    if self.is_3_layer:
      self.a1, self.a2, self.a3 = forward(x, self.w)
      return self.a3
    else:
      self.a1, self.a2 = forward(x, self.w)
      return self.a2

  def backward(self, x, y, lr):

    dw = [None] * 2

    # w[2] will always be fixed, so we can "remove" it before calculating grads
    if self.is_3_layer:
      inv = np.linalg.inv(self.w[2])
      y = inv_sigmoid(inv @ y)

    error = self.a2 - y

    dw[1] = error @ self.a1.T
    dw[0] = self.w[1].T @ error * self.a1 * (1 - self.a1) @ x.T

    # Only update weights for non-fixed parameters
    pos = parameter_positions[WEIGHTS_DIST]
    for _pos in pos:
      self.w[_pos[0]][_pos[1], _pos[2]] -= lr * dw[_pos[0]][_pos[1], _pos[2]] / len(x)

      # If the same parameter appears twice on the diagonal, fix them to be the same
      if _pos[1:3] == (0, 1):
        self.w[_pos[0]][1, 0] -= lr * dw[_pos[0]][1, 0] / len(x)

        if WEIGHTS_DIST == 'skew':
          self.w[_pos[0]][1, 0] = -self.w[_pos[0]][1, 0]

        diff = self.w[_pos[0]][0, 1] - self.w[_pos[0]][1, 0]
        self.w[_pos[0]][0, 1] -= diff / 2
        self.w[_pos[0]][1, 0] += diff / 2

        if WEIGHTS_DIST == 'skew':
          self.w[_pos[0]][1, 0] = -self.w[_pos[0]][1, 0]

class RotationalNet:
  def __init__(self, i, j):
    self.i = i
    self.j = j
    self.w = self.form_shell_weights(i, j)
    self._w = form_weights(i, j, [0]*6)

  def form_shell_weights(self, i, j):
    return [
      np.array([
        [i, 0.],
        [0., 0.]
      ]),
      np.array([
        [j, 0.],
        [0., 0.]
      ])
    ]

  def forward(self, x):
    self.a1 = sigmoid(self._w[0] @ x)
    self.a2 = self._w[1] @ self.a1
    return self.a2

  def backward(self, x, y, lr):
    dw = [None] * 2

    error = self.a2 - y

    # Derivative of rotational matrix
    da = lambda a: np.array([
      [-np.sin(a), -np.cos(a)],
      [np.cos(a), -np.sin(a)]
    ])

    daj = da(self.j) @ self.a1
    dw[1] = error @ daj.T

    dai = da(self.i) @ x
    dw[0] = self._w[1].T @ error * self.a1 * (1 - self.a1) @ dai.T

    avg_dw1 = np.mean(dw[1].flatten())
    avg_dw0 = np.mean(dw[0].flatten())

    self.j -= avg_dw1 * lr / len(x)
    self.i -= avg_dw0 * lr / len(x)

    self.w = self.form_shell_weights(self.i, self.j)
    self._w = form_weights(self.i, self.j, [0]*6)

def diag(a):
  return np.array([[a[0], a[1]], [a[1], a[2]]])

def forward_diag(a):
  return np.array([[a[0], a[1]], [a[2], a[0]]])

def skew_symmetric(i):
  return np.array([[0, i], [-i, 0]])

'''
  Distribute parameters i and j into 2x2 matrices, to form the weights.
'''
def form_weights(i, j, fixed):
  weights = [[fixed[0], fixed[1], fixed[2]], [fixed[3], fixed[4], fixed[5]]]

  if WEIGHTS_DIST == 'rotational':
    weights = [np.cos(i), -np.sin(i), np.sin(i)], [np.cos(j), -np.sin(j), np.sin(j)]
    return list(map(lambda x: forward_diag(x), weights))
  elif WEIGHTS_DIST == 'skew':
    return [skew_symmetric(i), skew_symmetric(j), diag([fixed[6], fixed[7], fixed[8]])]
  else:
    pos = parameter_positions[WEIGHTS_DIST]
    weights[pos[0][0]][pos[0][1] + pos[0][2]] = i
    weights[pos[1][0]][pos[1][1] + pos[1][2]] = j

    if WEIGHTS_DIST == 'equal' or WEIGHTS_DIST == 'second':
      weights += [[fixed[6], fixed[7], fixed[8]]]

  return list(map(lambda x: diag(x), weights))

def forward(x, w):
  a1 = sigmoid(w[0] @ x)
  a2 = w[1] @ a1
  if len(w) > 2:
    a3 = w[2] @ sigmoid(a2)
    return a1, a2, a3
  return a1, a2

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
  y_hat = forward(X, form_weights(i, j, fixed))[-1]
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
def plot_3d(i, j, fixed, Y, paths=None):
  fig = plt.figure()
  ax = fig.gca(projection='3d')

  Z = np.array(create_landscape(axis, fixed, Y))
  axis_x, axis_y = np.meshgrid(axis, axis)

  ax.plot_surface(axis_x, axis_y, Z, cmap=cm.terrain, linewidth=0, alpha=0.7)
  ax.scatter(i, j, 0, marker='x', s=150, color='black')
  if paths:
    for path in paths:
      ax.plot(*zip(*path), color='red')
      ax.scatter(*path[-1], marker='o', color='red')

  plt.xlim([-AXIS_SIZE, AXIS_SIZE])
  plt.ylim([-AXIS_SIZE, AXIS_SIZE])
  plt.xlabel('i')
  plt.ylabel('j')
  ax.set_zlabel('Loss')
  plt.show()


'''
  Plot chart of loss over epochs
'''
def plot_losses(losses, epochs):
  epoch_axis = np.arange(1, epochs + 1)
  plt.plot(epoch_axis, losses)
  plt.show()

def get_rand(shape):
  if RAND_DIST == 'normal':
    return np.random.normal(0, RAND_SD, shape)
  elif RAND_DIST == 'uniform':
    return np.random.uniform(-RAND_SD, high=RAND_SD, size=shape)

if __name__ == "__main__":
  rand = get_rand(12)
  fixed = rand[2:]

  X = get_rand((2, 10))
  Y = forward(X, form_weights(rand[0], rand[1], fixed))[-1]

  epochs = 10000
  lr = 0.1

  num_paths = 3
  sgd_paths = []

  # Model fixed weights set to same as Y, change to random to achieve non-convexity
  model_fixed = fixed if MODEL_FIXED_SAME else get_rand(12)

  for _ in range(num_paths):
    rand_init = (np.random.rand(2) * 2 - 1) * AXIS_SIZE

    if WEIGHTS_DIST == 'rotational':
      model = RotationalNet(*rand_init)
    else:
      model = TwoLayerNet(form_weights(*rand_init, model_fixed))

    path, losses = train(epochs, model, X, Y, lr)
    sgd_paths.append(path)

  # plot_losses(losses, epochs)
  # plot(model_fixed, Y, sgd_paths)
  # plot_3d(rand[0], rand[1], model_fixed, Y)
  plot_3d(rand[0], rand[1], model_fixed, Y, sgd_paths)
