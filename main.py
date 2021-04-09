import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

NUM_SAMPLES = 10
BATCH_SIZE = 10

WEIGHTS_DIST = 'chebyshev'
RESNET = False

ADD_NOISE = True
NOISE_SD = 0.05

LR_MIN = 0.03
LR_MAX = 0.4
LR_T0 = 100
LR_T_MULT = 2

LR_T = LR_T0
LR_T_CUR = 0

dimensions = {
  'first': [2, 5],
  'second': [3, 15],
  'equal': [1, 1],
  'rotational': [np.pi, np.pi],
  'skew': [1, 1],
  'chebyshev': [0.9, 0.9],
}

learning_rates = {
  'first': 0.5,
  'second': 0.1,
  'equal': 0.1,
  'rotational': 0.1,
  'skew': 0.1,
  'chebyshev': 0.1,
}

scaled = {
  'first': False,
  'second': False,
  'equal': True,
  'rotational': False,
  'skew': True,
  'chebyshev': True,
}

seeds = {
  'first': 1,
  'second': 0,
  'equal': 5,
  'rotational': 1,
  'skew': 5,
  'chebyshev': 3, #0, 3
}

SCALED = scaled[WEIGHTS_DIST]
np.random.seed(seeds[WEIGHTS_DIST])

RAND_SD = dimensions[WEIGHTS_DIST][0]
RAND_DIST = 'uniform'
AXIS_SIZE = dimensions[WEIGHTS_DIST][1]

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

class ClassicalNet:
  def __init__(self, w):
    self.w = w

  def forward(self, x):
    self.a1, self.a2 = forward(x, self.w)
    return self.a2

  def backward(self, x, y, lr):

    dw = [None] * 2

    error = self.a2 - y

    if SCALED:
      dw[1] = error @ (2*self.a1.T - 1)
    else:
      dw[1] = error @ self.a1.T

    w2 = self.w[1] + np.identity(2) if RESNET else self.w[1]

    dw[0] = w2.T @ error * self.a1 * (1 - self.a1) @ x.T

    # Only update weights for non-fixed parameters
    pos = parameter_positions[WEIGHTS_DIST]
    for _pos in pos:
      if self.w[_pos[0]][_pos[1], _pos[2]] < -AXIS_SIZE or self.w[_pos[0]][_pos[1], _pos[2]] > AXIS_SIZE:
        continue

      self.w[_pos[0]][_pos[1], _pos[2]] -= lr * dw[_pos[0]][_pos[1], _pos[2]]

      # If the same parameter appears twice on the diagonal, fix them to be the same
      if _pos[1:3] == (0, 1):
        self.w[_pos[0]][1, 0] -= lr * dw[_pos[0]][1, 0]

        if WEIGHTS_DIST == 'skew':
          self.w[_pos[0]][1, 0] *= -1

        diff = self.w[_pos[0]][0, 1] - self.w[_pos[0]][1, 0]
        self.w[_pos[0]][0, 1] -= diff / 2
        self.w[_pos[0]][1, 0] += diff / 2

        if WEIGHTS_DIST == 'skew':
          self.w[_pos[0]][1, 0] *= -1

class FunctionalNet:
  def __init__(self, i, j, scaled=False):
    self.i = i
    self.j = j
    self.scaled = scaled
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
    self.a1, self.a2 = forward(x, self._w, scaled=self.scaled)
    return self.a2

  def backward(self, x, y, lr):
    if not self.derivs:
      print("ERROR: Derivatives not defined")
      return

    if self.i < -AXIS_SIZE or self.i > AXIS_SIZE or self.j < -AXIS_SIZE or self.j > AXIS_SIZE:
      return

    d = {
      'i': None,
      'j': None
    }

    error = self.a2 - y

    # Derivative of functional matrix
    dw1 = self.derivs[0]
    dw2 = self.derivs[1]

    z1 = (2*self.a1-1) if self.scaled else self.a1

    daj = dw2(self.j) @ z1
    d['j'] = error * daj

    dz1 = dw1(self.i) @ x
    da1 = self.a1 * (1 - self.a1) * dz1
    dz2 = 2 * da1
    w2 = self._w[1] + np.identity(2) if RESNET else self._w[1]
    da2 = w2 @ dz2
    d['i'] = error * da2

    avg_dj = np.sum(d['j'].flatten()) / BATCH_SIZE
    avg_di = np.sum(d['i'].flatten()) / BATCH_SIZE

    self.j -= avg_dj * lr
    self.i -= avg_di * lr

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

class ChebyshevNet(FunctionalNet):
  def __init__(self, i, j):
    super().__init__(i, j, scaled=True)

    d = lambda a: np.array([
      [0, 1],
      [4*a, 12*(a**2) - 3]
    ])
    self.derivs = [d, d]

  @staticmethod
  def form_weights(i, j):
    l = lambda a: np.array([1, a, 2*(a**2) - 1, 4*(a**3) - 3*a])
    return l(i), l(j)


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

def forward(x, w, scaled=SCALED):
  i = np.identity(2) if RESNET else matrix([0]*4)
  a1 = sigmoid((w[0] + i) @ x)
  z1 = (2*a1-1) if scaled else a1
  a2 = (w[1] + i) @ z1
  return a1, a2

def loss(y_hat, Y):
  return sum(((y_hat - Y) ** 2).flatten()) / 2

def anneal_lr(lr):
  global LR_T, LR_T_CUR
  if LR_T_CUR == LR_T:
    # Reset
    LR_T_CUR = 0
    LR_T *= LR_T_MULT
  LR_T_CUR += 1
  return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + np.cos(np.pi * LR_T_CUR / LR_T))

def train(epochs, m, X, Y, lr):
  global LR_T, LR_T_CUR
  sgd_path = []
  losses = []
  lrs = []

  LR_T_CUR = 0
  LR_T = LR_T0

  for _ in range(epochs):
    xs = np.split(X, NUM_SAMPLES / BATCH_SIZE, axis=1)
    ys = np.split(Y, NUM_SAMPLES / BATCH_SIZE, axis=1)

    zipped = list(zip(xs, ys))

    np.random.shuffle(zipped)

    _loss = 0

    lr = anneal_lr(lr)

    for x, y in zipped:
      y_hat = m.forward(x)

      _loss += loss(y_hat, y)

      pos = parameter_positions[WEIGHTS_DIST]
      path = (m.w[pos[0][0]][pos[0][1], [pos[0][2]]], m.w[pos[1][0]][pos[1][1], [pos[1][2]]], _loss)

      m.backward(x, y, lr)

    losses.append(_loss)
    lrs.append(lr)
    sgd_path.append(path)

  return sgd_path, losses, lrs

def calc_loss(i, j, fixed, Y):
  y_hat = forward(X, form_weights(i, j, fixed))[-1]
  return loss(y_hat, Y)

def create_landscape(axis, fixed, Y):
  return [[calc_loss(i, j, fixed, Y) for i in axis] for j in axis]

'''
  Plot 3D contours of the loss landscape
'''
def plot(i, j, fixed, Y, paths=None):
  fig = plt.figure()
  ax = fig.gca(projection='3d')

  axis = np.arange(-AXIS_SIZE, AXIS_SIZE, AXIS_SIZE/100)
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
  for loss_plt in losses:
    plt.plot(epoch_axis, loss_plt)
  # plt.yscale('log')
  plt.show()

def plot_lrs(lrs, epochs, plot_log=True):
  epoch_axis = np.arange(1, epochs + 1)
  plt.plot(epoch_axis, lrs)
  if plot_log:
    plt.yscale('log')
    plt.ylim(1e-4, LR_MAX * 1.1)
  plt.show()

def get_rand(shape, dist=RAND_DIST):
  if dist == 'normal':
    return np.random.normal(0, RAND_SD, shape)
  elif dist == 'uniform':
    return np.random.uniform(-RAND_SD, high=RAND_SD, size=shape)

def add_noise(Y):
  noise = np.random.normal(0, NOISE_SD, np.shape(Y))
  return Y + noise

nets = {
  'chebyshev': ChebyshevNet,
  'rotational': RotationalNet,
}

if __name__ == "__main__":
  rand = get_rand(12, 'uniform')
  fixed = rand[2:]

  X = get_rand((2, NUM_SAMPLES))
  Y = forward(X, form_weights(rand[0], rand[1], fixed))[-1]

  Y_labels = add_noise(Y) if ADD_NOISE else Y

  epochs = 1000
  lr = learning_rates[WEIGHTS_DIST]

  num_paths = 5
  sgd_paths = []
  losses = []
  lrs = []

  Net = None if WEIGHTS_DIST not in nets else nets[WEIGHTS_DIST]

  for _ in range(num_paths):
    rand_init = (np.random.rand(2) * 2 - 1) * AXIS_SIZE * 0.9

    if Net:
      model = Net(*rand_init)
    else:
      model = ClassicalNet(form_weights(*rand_init, fixed))

    path, _losses, lrs = train(epochs, model, X, Y_labels, lr)
    sgd_paths.append(path)
    losses.append(_losses)

  # plot(rand[0], rand[1], fixed, Y)
  plot(rand[0], rand[1], fixed, Y, sgd_paths)
  # plot_losses(losses, epochs)
  plot_lrs(lrs, epochs, plot_log=False)
