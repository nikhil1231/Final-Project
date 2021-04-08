import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math

NUM_SAMPLES = 10
BATCH_SIZE = 10

WEIGHTS_DIST = 'chebyshev'
RESNET_DIST = 'rotational'

dimensions = {
  'first': [2, 5],
  'second': [3, 15],
  'equal': [1, 1],
  'rotational': [math.pi, math.pi],
  'skew': [1, 1],
  'resnet': [1, 1],
  'chebyshev': [0.9, 0.9],
}

learning_rates = {
  'first': 0.5,
  'second': 0.1,
  'equal': 0.1,
  'rotational': 0.1,
  'skew': 0.1,
  'resnet': 0.1,
  'chebyshev': 0.1,
}

seeds = {
  'first': 1,
  'second': 0,
  'equal': 5,
  'rotational': 1,
  'skew': 5,
  'chebyshev': 5,
}
DIST = RESNET_DIST if WEIGHTS_DIST == 'resnet' else WEIGHTS_DIST
np.random.seed(seeds[DIST])

RAND_SD = dimensions[DIST][0]
RAND_DIST = 'uniform'
AXIS_SIZE = dimensions[DIST][1]

parameter_positions = {
  'first': [(0, 0, 0), (0, 0, 1)],
  'second': [(1, 0, 0), (1, 0, 1)],
  'equal': [(0, 0, 1), (1, 0, 1)],
  'rotational': [(0, 0, 0), (1, 0, 0)],
  'skew': [(0, 0, 1), (1, 0, 1)],
  'resnet': [(0, 0, 0), (0, 0, 1)],
  'monomial': [(0, 0, 0), (1, 0, 0)],
  'chebyshev': [(0, 0, 0), (1, 0, 0)],
}

sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

class TwoLayerNet:
  def __init__(self, w):
    self.w = w

  def forward(self, x):
    self.a1, self.a2 = forward(x, self.w)
    return self.a2

  def backward(self, x, y, lr):

    dw = [None] * 2

    error = self.a2 - y

    dw[1] = error @ self.a1.T
    dw[0] = self.w[1].T @ error * 2 * self.a1 * (1 - self.a1) @ x.T

    # Only update weights for non-fixed parameters
    pos = parameter_positions[WEIGHTS_DIST]
    for _pos in pos:
      self.w[_pos[0]][_pos[1], _pos[2]] -= lr * dw[_pos[0]][_pos[1], _pos[2]]
      self.w[_pos[0]][_pos[1], _pos[2]] = min(max(self.w[_pos[0]][_pos[1], _pos[2]], -AXIS_SIZE), AXIS_SIZE)

      # If the same parameter appears twice on the diagonal, fix them to be the same
      if _pos[1:3] == (0, 1):
        self.w[_pos[0]][1, 0] -= lr * dw[_pos[0]][1, 0]

        if WEIGHTS_DIST == 'skew':
          self.w[_pos[0]][1, 0] = -self.w[_pos[0]][1, 0]

        diff = self.w[_pos[0]][0, 1] - self.w[_pos[0]][1, 0]
        self.w[_pos[0]][0, 1] -= diff / 2
        self.w[_pos[0]][1, 0] += diff / 2

        if WEIGHTS_DIST == 'skew':
          self.w[_pos[0]][1, 0] = -self.w[_pos[0]][1, 0]

class FunctionalNet:
  def __init__(self, i, j):
    self.i = i
    self.j = j
    self.w = self.form_shell_weights(i, j)
    self._w = form_weights(i, j, [0]*6)
    self.derivs = None

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

  def get_loss(self, i, x, y):
    weights = ChebyshevNet.form_weights(i, self.j)
    w = list(map(lambda x: matrix(x), weights))

    w1 = w[0]
    z1 = w[0] @ x
    a1 = sigmoid(z1)
    z2 = 2*a1 - 1
    a2 = w[1] @ z2
    l = loss(a2, y)

    return w1, z1, a1, z2, a2, l

  def compare_grads(self, fd, d):
    print("Finite diff", np.sum(fd))
    print("Analytical", np.sum(d))
    print("Difference", np.sum(fd) - np.sum(d))

  def backward(self, x, y, lr):
    if not self.derivs:
      print("ERROR: Derivatives not defined")
      return

    d = {
      'i': None,
      'j': None
    }

    error = self.a2 - y

    # Derivative of functional matrix
    dw1 = self.derivs[0]
    dw2 = self.derivs[1]

    daj = dw2(self.j) @ (2*self.a1-1)
    d['j'] = error * daj


    dz1 = dw1(self.i) @ x
    da1 = self.a1 * (1 - self.a1) * dz1
    dz2 = 2 * da1
    da2 = self._w[1] @ dz2
    d['i'] = error * da2

    fd_a = 1e-8

    grads_a = np.array(self.get_loss(self.i, x, y))
    grads_b = np.array(self.get_loss(self.i + fd_a, x, y))

    (fd_w1, fd_z1, fd_a1, fd_z2, fd_a2, fd_l) = (grads_b - grads_a) / fd_a

    avg_dj = np.sum(d['j'].flatten()) / BATCH_SIZE
    avg_di = np.sum(d['i'].flatten()) / BATCH_SIZE

    self.compare_grads(fd_l, d['i'])

    self.j -= avg_dj * lr
    self.i -= avg_di * lr

    self.i = min(max(self.i, -AXIS_SIZE), AXIS_SIZE)
    self.j = min(max(self.j, -AXIS_SIZE), AXIS_SIZE)

    self.w = self.form_shell_weights(self.i, self.j)
    self._w = form_weights(self.i, self.j, [0]*6)

class RotationalNet(FunctionalNet):
  def __init__(self, i, j):
    super().__init__(i, j)
    self.derivs = [
      lambda a: np.array([
        [-np.sin(a), -np.cos(a)],
        [np.cos(a), -np.sin(a)]
      ]),
      lambda a: np.array([
        [-np.sin(a), -np.cos(a)],
        [np.cos(a), -np.sin(a)]
      ]),
    ]

  def forward(self, x):
    self.a1, self.a2 = RotationalNet._forward(x, self._w)
    return self.a2

  @staticmethod
  def _forward(x, w):
    a1 = sigmoid(w[0] @ x)
    a2 = w[1] @ a1
    return a1, a2

class FunctionalTanhNet(FunctionalNet):
  def __init__(self, i, j):
    super().__init__(i, j)

  def forward(self, x):
    self.a1, self.a2 = FunctionalTanhNet._forward(x, self._w)
    return self.a2

  @staticmethod
  def _forward(x, w):
    a1 = sigmoid(w[0] @ x)
    a2 = w[1] @ (2*a1 - 1)
    return a1, a2

class ChebyshevNet(FunctionalTanhNet):
  def __init__(self, i, j):
    super().__init__(i, j)

    d = lambda a: np.array([
      [0, 1],
      [4*a, 12*(a**2) - 3]
    ])
    self.derivs = [d, d]

  @staticmethod
  def form_weights(i, j):
    l = lambda a: np.array([1, a, 2*(a**2) - 1, 4*(a**3) - 3*a])
    return l(i), l(j)

class ResNet:
  def __init__(self, w):
    self.w = w

  def forward(self, x):
    self.a1, self.a2 = ResNet._forward(x, self.w)
    return self.a2

  @staticmethod
  def _forward(x, w):
    # a1 = sigmoid(w[0] @ x) + x
    # a2 = w[1] @ a1 + a1
    a1 = sigmoid(w[0] @ x + x)*2 - 1
    a2 = w[1] @ a1 + a1
    return a1, a2

  def backward(self, x, y, lr):

    dw = [None] * 2

    dw[1] = matrix([0] * 4)
    dw[0] = matrix([0] * 4)

def diag(a):
  return np.array([[a[0], a[1]], [a[1], a[2]]])

def forward_diag(a):
  return np.array([[a[0], a[1]], [a[2], a[0]]])

def skew_symmetric(i):
  return np.array([[0, i], [-i, 0]])

def matrix(a):
  return np.array([[a[0], a[1]], [a[2], a[3]]])

'''
  Distribute parameters i and j into 2x2 matrices, to form the weights.
'''
def form_weights(i, j, fixed, dist=WEIGHTS_DIST):
  weights = [[fixed[0], fixed[1], fixed[2]], [fixed[3], fixed[4], fixed[5]]]
  weights = list(map(lambda x: x/np.linalg.norm(x), weights))

  if dist == 'rotational':
    weights = [np.cos(i), -np.sin(i), np.sin(i)], [np.cos(j), -np.sin(j), np.sin(j)]
    return list(map(lambda x: forward_diag(x), weights))

  elif dist == 'chebyshev':
    weights = ChebyshevNet.form_weights(i, j)
    return list(map(lambda x: matrix(x), weights))

  elif dist == 'skew':
    return [skew_symmetric(i), skew_symmetric(j)]
  else:
    pos = parameter_positions[dist]
    weights[pos[0][0]][pos[0][1] + pos[0][2]] = i
    weights[pos[1][0]][pos[1][1] + pos[1][2]] = j

  return list(map(lambda x: diag(x), weights))

def forward(x, w, net=None):
  if net:
    return net._forward(x, w)

  a1 = sigmoid(w[0] @ x)
  if DIST in ['equal', 'skew']:
    a2 = w[1] @ (2*a1 - 1)
  else:
    a2 = w[1] @ a1

  return a1, a2

def loss(y_hat, Y):
  return sum(((y_hat - Y) ** 2).flatten()) / 2

def train(epochs, m, X, Y, lr):
  sgd_path = []
  losses = []
  for _ in range(epochs):
    xs = np.split(X, NUM_SAMPLES / BATCH_SIZE, axis=1)
    ys = np.split(Y, NUM_SAMPLES / BATCH_SIZE, axis=1)

    zipped = list(zip(xs, ys))

    np.random.shuffle(zipped)

    _loss = 0

    for x, y in zipped:
      y_hat = m.forward(x)

      _loss += loss(y_hat, y)

      pos = parameter_positions[WEIGHTS_DIST]
      path = (m.w[pos[0][0]][pos[0][1], [pos[0][2]]], m.w[pos[1][0]][pos[1][1], [pos[1][2]]], _loss)

      m.backward(x, y, lr)

    losses.append(_loss)

    sgd_path.append(path)

  return sgd_path, losses

def calc_loss(i, j, fixed, Y, net):
  y_hat = forward(X, form_weights(i, j, fixed, dist=DIST), net)[-1]
  return loss(y_hat, Y)

def create_landscape(axis, fixed, Y, net):
  return [[calc_loss(i, j, fixed, Y, net) for i in axis] for j in axis]

axis = np.arange(-AXIS_SIZE, AXIS_SIZE, AXIS_SIZE/100)

'''
  Plot 3D contours of the loss landscape
'''
def plot(i, j, fixed, Y, paths=None, net=None):
  fig = plt.figure()
  ax = fig.gca(projection='3d')

  Z = np.array(create_landscape(axis, fixed, Y, net))
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
  for loss_plt in losses:
    plt.plot(epoch_axis, loss_plt)
  # plt.yscale('log')
  plt.show()

def get_rand(shape, dist=RAND_DIST):
  if dist == 'normal':
    return np.random.normal(0, RAND_SD, shape)
  elif dist == 'uniform':
    return np.random.uniform(-RAND_SD, high=RAND_SD, size=shape)

nets = {
  'resnet': ResNet,
  'chebyshev': ChebyshevNet,
  'rotational': RotationalNet,
}

if __name__ == "__main__":
  rand = get_rand(12, 'uniform')
  fixed = rand[2:]

  Net = None if WEIGHTS_DIST not in nets else nets[WEIGHTS_DIST]

  X = get_rand((2, NUM_SAMPLES))
  Y = forward(X, form_weights(rand[0], rand[1], fixed, dist=DIST), net=Net)[-1]

  epochs = 100
  lr = learning_rates[DIST]

  num_paths = 1
  sgd_paths = []
  losses = []

  for _ in range(num_paths):
    rand_init = (np.random.rand(2) * 2 - 1) * AXIS_SIZE * 0.9

    Net = None if WEIGHTS_DIST not in nets else nets[WEIGHTS_DIST]

    if Net and WEIGHTS_DIST != 'resnet':
      model = Net(*rand_init)
    elif WEIGHTS_DIST == 'rotational':
      model = RotationalNet(*rand_init)
    else:
      if WEIGHTS_DIST == 'resnet':
        Net = ResNet
        model = Net(form_weights(*rand_init, fixed, dist=RESNET_DIST))
      else:
        model = TwoLayerNet(form_weights(*rand_init, fixed))

    path, _losses = train(epochs, model, X, Y, lr)
    sgd_paths.append(path)
    losses.append(_losses)

  # plot(rand[0], rand[1], fixed, Y, net=Net)
  plot(rand[0], rand[1], fixed, Y, sgd_paths, net=Net)
  # plot_losses(losses, epochs)
