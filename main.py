import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

NUM_SAMPLES = 10
BATCH_SIZE = 10

WEIGHTS_DIST = 'equal'
MODEL_FIXED_SAME = True

dimensions = {
  'first': [1, 1],
  'second': [3, 15],
  'equal': [1, 1],
  'rotational': [3, 5],
  'skew': [1, 1],
  'resnet': [3, 15],
  'monomial': [0.5, 0.5],
  'chebyshev': [0.9, 0.9],
}

seeds = {
  'first': 0,
  'second': 0,
  'equal': 5,
  'rotational': 0,
  'skew': 5,
  'resnet': 0,
  'monomial': 0,
  'chebyshev': 0,
}
np.random.seed(seeds[WEIGHTS_DIST])

RAND_SD = dimensions[WEIGHTS_DIST][0]
RAND_DIST = 'uniform'
AXIS_SIZE = dimensions[WEIGHTS_DIST][1]
ADD_NOISE = False
NOISE_STRENGTH = 5

ADD_ANNEALING = False
ANNEALING_STRENGTH = 100

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

    if ADD_NOISE:
      x = add_noise(x)

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

  def backward(self, x, y, lr):
    if not self.derivs:
      print("ERROR: Derivatives not defined")
      return

    dw = [None] * 2

    if ADD_NOISE:
      x = add_noise(x)

    error = self.a2 - y

    # Derivative of rotational matrix
    da1 = self.derivs[0]
    da2 = self.derivs[1]

    daj = da2(self.j) @ self.a1
    dw[1] = error * daj

    dai = da1(self.i) @ x
    dw[0] = self._w[1].T @ error * self.a1 * (1 - self.a1) @ dai.T

    avg_dw1 = np.sum(dw[1].flatten()) / len(x)
    avg_dw0 = np.sum(dw[0].flatten()) / len(x)

    self.j -= avg_dw1 * lr
    self.i -= avg_dw0 * lr

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

class FunctionalTanhNet(FunctionalNet):
  def __init__(self, i, j):
    super().__init__(i, j)

  def forward(self, x):
    self.a1, self.a2 = FunctionalTanhNet._forward(x, self._w)
    return self.a2

  @staticmethod
  def _forward(x, w):
    a1 = sigmoid(w[0] @ x) * 2 - 1
    a2 = w[1] @ a1
    return a1, a2

class MonomialNet(FunctionalTanhNet):
  def __init__(self, i, j):
    super().__init__(i, j)

  def backward(self, x, y, lr):
    pass

  @staticmethod
  def form_weights(i, j):
    # l = lambda a, b, c: np.array([1, a, b**2, c**3])
    l = lambda a, b, c, d: np.array([a, b**2, c**3, d**4])
    return l(i, i, i, i), l(j, j, j, j)

class ChebyshevNet(FunctionalTanhNet):
  def __init__(self, i, j):
    super().__init__(i, j)

    d = lambda a: np.array([
      [0, 1],
      [4*a, 12*(a**2) - 3]
    ])
    d2 = lambda a: np.array([
      [1, 4*a],
      [12*(a**2) - 3, 32*(d**3) - 16*d]
    ])
    self.derivs = [d, d]

  @staticmethod
  def form_weights(i, j):
    l = lambda a, b, c: np.array([1, a, 2*(b**2) - 1, 4*(c**3) - 3*c])
    l2 = lambda a, b, c, d: np.array([a, 2*(b**2) - 1, 4*(c**3) - 3*c, 8*(d**4) - 8*(d**2) + 1])
    # return l(i, i, i), l(j, j, j)
    return l2(i, i, i, i), l2(j, j, j, j)

class ResNet:
  def __init__(self, w):
    self.w = w
    self.is_3_layer = len(w) == 3

  def forward(self, x):
    if self.is_3_layer:
      self.a1, self.a2, self.a3 = ResNet._forward(x, self.w)
      return self.a3
    else:
      self.a1, self.a2 = ResNet._forward(x, self.w)
      return self.a2

  @staticmethod
  def _forward(x, w):
    a1 = sigmoid(w[0] @ x) + x
    a2 = w[1] @ a1 + a1
    if len(w) > 2:
      a3 = w[2] @ sigmoid(a2) + a2
      return a1, a2, a3
    return a1, a2

  def backward(self, x, y, lr):

    dw = [None] * 2

    if ADD_NOISE:
      x = add_noise(x)

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
def form_weights(i, j, fixed):
  weights = [[fixed[0], fixed[1], fixed[2]], [fixed[3], fixed[4], fixed[5]]]
  weights = list(map(lambda x: x/np.linalg.norm(x), weights))

  if WEIGHTS_DIST == 'rotational':
    weights = [np.cos(i), -np.sin(i), np.sin(i)], [np.cos(j), -np.sin(j), np.sin(j)]
    return list(map(lambda x: forward_diag(x), weights))

  elif WEIGHTS_DIST == 'monomial':
    weights = MonomialNet.form_weights(i, j)
    return list(map(lambda x: matrix(x), weights))

  elif WEIGHTS_DIST == 'chebyshev':
    weights = ChebyshevNet.form_weights(i, j)
    return list(map(lambda x: matrix(x), weights))

  elif WEIGHTS_DIST == 'skew':
    return [skew_symmetric(i), skew_symmetric(j)]
  else:
    pos = parameter_positions[WEIGHTS_DIST]
    weights[pos[0][0]][pos[0][1] + pos[0][2]] = i
    weights[pos[1][0]][pos[1][1] + pos[1][2]] = j

    if WEIGHTS_DIST in ['equal', 'second']:
      weights += [[fixed[6], fixed[7], fixed[8]]]

  return list(map(lambda x: diag(x), weights))

def add_noise(m):
  return m + np.random.normal(0, NOISE_STRENGTH, np.shape(m))

def forward(x, w, net=None):
  if net:
    return net._forward(x, w)
  a1 = sigmoid(w[0] @ x)*2 - 1
  a2 = w[1] @ a1
  if len(w) > 2:
    a3 = w[2] @ (2*sigmoid(a2)-1)
    return a1, a2, a3
  return a1, a2

def loss(y_hat, Y):
  return sum(((y_hat - Y) ** 2).flatten()) / 2

def anneal(lr, epoch):
  return lr * np.exp(-epoch/ANNEALING_STRENGTH)

def train(epochs, m, X, Y, lr):
  sgd_path = []
  losses = []
  for epoch in range(epochs):
    xs = np.split(X, NUM_SAMPLES / BATCH_SIZE, axis=1)
    ys = np.split(Y, NUM_SAMPLES / BATCH_SIZE, axis=1)

    zipped = list(zip(xs, ys))

    np.random.shuffle(zipped)

    _loss = 0

    lr = anneal(lr, epoch)

    for x, y in zipped:
      y_hat = m.forward(x)

      _loss += loss(y_hat, y)

      pos = parameter_positions[WEIGHTS_DIST]
      path = (m.w[pos[0][0]][pos[0][1], [pos[0][2]]], m.w[pos[1][0]][pos[1][1], [pos[1][2]]], _loss)

      m.backward(x, y, anneal(lr, epoch) if ADD_ANNEALING else lr)

    losses.append(_loss)

    sgd_path.append(path)

  return sgd_path, losses

def calc_loss(i, j, fixed, Y, net):
  y_hat = forward(X, form_weights(i, j, fixed), net)[-1]
  return loss(y_hat, Y)

def create_landscape(axis, fixed, Y, net):
  return [[calc_loss(i, j, fixed, Y, net) for i in axis] for j in axis]

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
def plot_3d(i, j, fixed, Y, paths=None, net=None):
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
  plt.plot(epoch_axis, losses)
  plt.show()

def get_rand(shape, dist=RAND_DIST):
  if dist == 'normal':
    return np.random.normal(0, RAND_SD, shape)
  elif dist == 'uniform':
    return np.random.uniform(-RAND_SD, high=RAND_SD, size=shape)

nets = {
  'resnet': ResNet,
  'monomial': MonomialNet,
  'chebyshev': ChebyshevNet,
}

if __name__ == "__main__":
  rand = get_rand(12, 'uniform')
  fixed = rand[2:]

  Net = None if WEIGHTS_DIST not in nets else nets[WEIGHTS_DIST]

  X = get_rand((2, NUM_SAMPLES))
  Y = forward(X, form_weights(rand[0], rand[1], fixed), net=Net)[-1]

  epochs = 10000
  lr = 0.1 if WEIGHTS_DIST in ['rotational', 'monomial', 'chebyshev', 'resnet'] else 0.7

  num_paths = 1
  sgd_paths = []

  # Model fixed weights set to same as Y, change to random to achieve non-convexity
  model_fixed = fixed if MODEL_FIXED_SAME else get_rand(12)

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
        model = Net(form_weights(*rand_init, model_fixed))
      else:
        model = TwoLayerNet(form_weights(*rand_init, model_fixed))

    path, losses = train(epochs, model, X, Y, lr)
    sgd_paths.append(path)

  # plot_losses(losses, epochs)
  # plot(model_fixed, Y, sgd_paths)
  # plot_3d(rand[0], rand[1], model_fixed, Y, net=Net)
  plot_3d(rand[0], rand[1], model_fixed, Y, sgd_paths, net=Net)
