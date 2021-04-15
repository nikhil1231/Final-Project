import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import itertools, functools, operator
from os.path import exists

PLOT_SURFACE = True
PLOT_SGD = True
PLOT_LOSSES = False
PLOT_LR = False

TEST_NET = False
FORCE_MAP_SGD = True

NUM_SAMPLES = 1000
BATCH_SIZE = 10

WEIGHTS_DIST = 'chebyshev'
RESNET = False
RESNET_LAST_ACTIVATE = True

ADD_NOISE = False
NOISE_SD = 0.05

LR_MIN = 0.0001
LR_MAX = 0.001
EPOCHS = 1000

# Parameter range, sample range, axis range
dimension_defaults = {
  'first': [2, 2, 5],
  'second': [1, 1, 5],
  'equal': [1, 1, 1],
  'rotational': [1, np.pi, 5],
  'skew': [1, 1, 1],
  'chebyshev': [0.9, 0.9, 0.9],
}

scaled_defaults = {
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
  'chebyshev': 6, #0, 3
}

view_angle_defaults = {
  'first': [50, 60],
  'second': [50, 130],
  'equal': [70, 210],
  'rotational': [70, 210],
  'skew': [70, 210],
  'chebyshev': [60, 30],
}

RAND_DIST = 'uniform'

parameter_positions = {
  'first': [(0, 0, 0), (0, 0, 1)],
  'second': [(1, 0, 0), (1, 0, 1)],
  'equal': [(0, 0, 1), (1, 0, 1)],
  'rotational': [(0, 0, 0), (1, 0, 0)],
  'skew': [(0, 0, 1), (1, 0, 1)],
  'resnet': [(0, 0, 0), (0, 0, 1)],
  'chebyshev': [(0, 0, 1), (1, 0, 1)],
}

'''
Values in the weight distribution to be frozen during unbound SGD.

In the form (matrix_i, row_i, col_i, value)
'''
test_bound_vars = {
  'chebyshev': [(0, 0, 0, 1), (1, 0, 0, 1)],
}

sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

class ClassicalNet:
  def __init__(self, w, dist, test_net, scaled, resnet, resnet_last_activate, axis_size, unbound=False):
    self.w = w
    self.dist = dist
    self.test_net = test_net
    self.scaled = scaled
    self.resnet = resnet
    self.resnet_last_activate = resnet_last_activate
    self.axis_size = axis_size
    self.unbound = unbound
    self.bind_vars()

  def forward(self, x):
    self.a1, self.a2 = forward(x, self.w, self.scaled, self.resnet, self.resnet_last_activate)
    return self.a2

  def backward(self, x, y, lr):

    dw = [None] * 2

    error = self.a2 - y

    if self.scaled:
      dw[1] = error @ (2*self.a1.T - 1)
    else:
      dw[1] = error @ self.a1.T

    w2 = self.w[1] + np.identity(2) if self.resnet else self.w[1]

    dw[0] = w2.T @ error * self.a1 * (1 - self.a1) @ x.T

    if self.unbound:
      self.w[0] -= lr * dw[0]
      self.w[1] -= lr * dw[1]
      self.bind_vars()
      return

    # Only update weights for non-fixed parameters
    pos = parameter_positions[self.dist]
    for _pos in pos:
      if self.w[_pos[0]][_pos[1], _pos[2]] < -self.axis_size or self.w[_pos[0]][_pos[1], _pos[2]] > self.axis_size:
        continue

      self.w[_pos[0]][_pos[1], _pos[2]] -= lr * dw[_pos[0]][_pos[1], _pos[2]]

      # If the same parameter appears twice on the diagonal, fix them to be the same
      if _pos[1:3] == (0, 1):
        self.w[_pos[0]][1, 0] -= lr * dw[_pos[0]][1, 0]

        if self.dist == 'skew':
          self.w[_pos[0]][1, 0] *= -1

        diff = self.w[_pos[0]][0, 1] - self.w[_pos[0]][1, 0]
        self.w[_pos[0]][0, 1] -= diff / 2
        self.w[_pos[0]][1, 0] += diff / 2

        if self.dist == 'skew':
          self.w[_pos[0]][1, 0] *= -1

  def get_parameters(self):
    pos = parameter_positions[self.dist]
    return self.w[pos[0][0]][pos[0][1], pos[0][2]], self.w[pos[1][0]][pos[1][1], pos[1][2]]

  def bind_vars(self):
    if self.dist in test_bound_vars and self.test_net:
      for var in test_bound_vars[self.dist]:
        self.w[var[0]][var[1], var[2]] = var[3]

class FunctionalNet:
  def __init__(self, i, j, dist, test_net, scaled, resnet, resnet_last_activate, batch_size, axis_size):
    self.i = i
    self.j = j
    self.dist = dist
    self.test_net = test_net
    self.scaled = scaled
    self.resnet = resnet
    self.resnet_last_activate = resnet_last_activate
    self.batch_size = batch_size
    self.axis_size = axis_size
    self.w = form_weights(i, j, [0]*6, dist)
    self.derivs = None

  def forward(self, x):
    self.a1, self.a2 = forward(x, self.w, self.scaled, self.resnet, self.resnet_last_activate)
    return self.a2

  def backward(self, x, y, lr):
    if not self.derivs:
      print("ERROR: Derivatives not defined")
      return

    if self.i < -self.axis_size or self.i > self.axis_size or self.j < -self.axis_size or self.j > self.axis_size:
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
    w2 = self.w[1] + np.identity(2) if self.resnet else self.w[1]
    da2 = w2 @ dz2
    d['i'] = error * da2

    avg_dj = np.sum(d['j'].flatten()) / self.batch_size
    avg_di = np.sum(d['i'].flatten()) / self.batch_size

    self.j -= avg_dj * lr
    self.i -= avg_di * lr

    self.w = form_weights(self.i, self.j, [0]*6, self.dist)

  def get_parameters(self):
    return self.i, self.j

class RotationalNet(FunctionalNet):
  def __init__(self, i, j, dist, test_net, scaled, resnet, resnet_last_activate, batch_size, axis_size):
    super().__init__(i, j, dist, test_net, scaled, resnet, resnet_last_activate, batch_size, axis_size)
    d = lambda a: np.array([
      [-np.sin(a), -np.cos(a)],
      [np.cos(a), -np.sin(a)]
    ])
    self.derivs = [d, d]

class ChebyshevNet(FunctionalNet):
  def __init__(self, i, j, dist, test_net, scaled, resnet, resnet_last_activate, batch_size, axis_size):
    super().__init__(i, j, dist, test_net, scaled, resnet, resnet_last_activate, batch_size, axis_size)

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
def form_weights(i, j, fixed, dist):
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

def forward(x, w, scaled, resnet, resnet_last_activate):
  if resnet:
    a1 = sigmoid(w[0] @ x) + x
    z1 = (a1 * 2/3) - 1/3 if scaled else a1
    if resnet_last_activate:
      a2 = sigmoid(w[1] @ z1) + z1
      a2 = (a2 * 2/3) - 1/3 if scaled else a2
    else:
      a2 = w[1] @ z1 + z1
  else:
    a1 = sigmoid(w[0] @ x)
    z1 = (2*a1-1) if scaled else a1
    a2 = w[1] @ z1
  return a1, a2

def loss(y_hat, Y):
  return sum(((y_hat - Y) ** 2).flatten()) / 2

def anneal_lr(epoch, lr_min, lr_max, max_epochs):
  return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / max_epochs))

def train(epochs, m, X, Y, fixed,
          lr_min, lr_max, max_epochs,
          num_samples, batch_size,
          test_net, force_map_sgd,
          scaled, resnet, resnet_last_active):
  sgd_path = []
  losses = []
  lrs = []

  for epoch in range(epochs):
    xs = np.split(X, num_samples / batch_size, axis=1)
    ys = np.split(Y, num_samples / batch_size, axis=1)

    zipped = list(zip(xs, ys))

    np.random.shuffle(zipped)

    _loss = 0

    lr = anneal_lr(epoch, lr_min, lr_max, max_epochs)

    for x, y in zipped:
      params = m.get_parameters()

      y_hat = m.forward(x)
      if test_net and force_map_sgd:
        y_hat = forward(x, form_weights(*params, fixed), scaled, resnet, resnet_last_active)[-1]

      _loss += loss(y_hat, y)

      path = (*params, _loss)

      m.backward(x, y, lr)

    losses.append(_loss)
    lrs.append(lr)
    sgd_path.append(path)

  return sgd_path, losses, lrs

def calc_loss(i, j, fixed, X, Y, dist, scaled, resnet, resnet_last_active):
  y_hat = forward(X, form_weights(i, j, fixed, dist), scaled, resnet, resnet_last_active)[-1]
  return loss(y_hat, Y)

def create_landscape(axis, fixed, X, Y, dist, scaled, resnet, resnet_last_active):
  return [[calc_loss(i, j, fixed, X, Y, dist, scaled, resnet, resnet_last_active) for i in axis] for j in axis]

'''
  Plot 3D contours of the loss landscape
'''
def plot(i, j, fixed, X, Y, dist, scaled, resnet, resnet_last_active, axis_size, save, filepath, paths=None):
  fig = plt.figure()
  ax = fig.gca(projection='3d')

  axis = np.arange(-axis_size, axis_size, axis_size/100)
  Z = np.array(create_landscape(axis, fixed, X, Y, dist, scaled, resnet, resnet_last_active))
  axis_x, axis_y = np.meshgrid(axis, axis)

  ax.plot_surface(axis_x, axis_y, Z, cmap=cm.terrain, linewidth=0, alpha=0.7)
  ax.scatter(i, j, 0, marker='x', s=150, color='black')
  if paths:
    for path in paths:
      ax.plot(*zip(*path), color='red')
      ax.scatter(*path[-1], marker='o', color='red')

  plt.xlim([-axis_size, axis_size])
  plt.ylim([-axis_size, axis_size])
  ax.axes.set_zlim3d([0, np.max(Z)])
  plt.xlabel('i')
  plt.ylabel('j')
  ax.set_zlabel('Loss')

  elevation = view_angle_defaults[dist][0]
  azimuth = view_angle_defaults[dist][1]

  ax.view_init(elev=elevation, azim=azimuth)

  if save:
    plt.savefig(filepath)
  else:
    plt.show()

def get_file_path(weights_dist, epochs, test_net, num_samples, batch_size, parameter_sd, sample_sd,
                  axis_size, test_sd, scaled, resnet, resnet_last_activate, lr_min, lr_max,
                  force_map_sgd, noise_sd, rand_dist):
  return f"figs/{weights_dist}_E{epochs}_T{test_net}_NS{num_samples}_BS{batch_size}_PSD{parameter_sd}\
_SSD{round(sample_sd, 2)}_AX{round(axis_size, 2)}_TSD{test_sd}_S{scaled}_RN{resnet}_RNL{resnet_last_activate}\
_LR{lr_min}-{lr_max}_FM{force_map_sgd}_N{noise_sd}_RD{rand_dist}.png"

'''
  Plot chart of loss over epochs
'''
def plot_losses(losses, epochs):
  epoch_axis = np.arange(1, epochs + 1)
  for loss_plt in losses:
    plt.plot(epoch_axis, loss_plt)
  # plt.yscale('log')
  plt.ylim(bottom=-0.5)
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.show()

def plot_lrs(lrs, epochs, plot_log=True):
  epoch_axis = np.arange(1, epochs + 1)
  plt.plot(epoch_axis, lrs)
  if plot_log:
    plt.yscale('log')
    plt.ylim(1e-4, LR_MAX * 1.1)
  plt.xlabel('Epochs')
  plt.ylabel('Learning rate (ηt)')
  plt.show()

def get_rand(shape, sd, dist):
  if dist == 'normal':
    return np.random.normal(0, sd, shape)
  elif dist == 'uniform':
    return np.random.uniform(-sd, high=sd, size=shape)

def add_noise(Y, noise_sd):
  noise = np.random.normal(0, noise_sd, np.shape(Y))
  return Y + noise

nets = {
  'chebyshev': ChebyshevNet,
  'rotational': RotationalNet,
}

def run(weights_dist=WEIGHTS_DIST,
        epochs=EPOCHS,
        test_net=TEST_NET,
        num_samples=NUM_SAMPLES,
        batch_size=BATCH_SIZE,
        parameter_sd=None,
        sample_sd=None,
        axis_size=None,
        test_sd=None,
        scaled=None,
        resnet=RESNET,
        resnet_last_activate=RESNET_LAST_ACTIVATE,
        lr_min=LR_MIN,
        lr_max=LR_MAX,
        force_map_sgd=FORCE_MAP_SGD,
        noise_sd=NOISE_SD,
        rand_dist=RAND_DIST,
        save_plot=False):

  if parameter_sd is None:
    parameter_sd = dimension_defaults[weights_dist][0]
  if sample_sd is None:
    sample_sd = dimension_defaults[weights_dist][1]
  if axis_size is None:
    axis_size = dimension_defaults[weights_dist][2]
  if scaled is None:
    scaled = scaled_defaults[weights_dist]

  fn = get_file_path(weights_dist, epochs, test_net, num_samples, batch_size, parameter_sd, sample_sd,
                    axis_size, test_sd, scaled, resnet, resnet_last_activate, lr_min, lr_max,
                    force_map_sgd, noise_sd, rand_dist)
  if save_plot and exists(fn):
    return

  np.random.seed(seeds[weights_dist])

  parameters = get_rand(2, parameter_sd, rand_dist)
  fixed = get_rand(6, parameter_sd, 'uniform')

  X = get_rand((2, num_samples), sample_sd, rand_dist)
  Y = forward(X, form_weights(*parameters, fixed, weights_dist), scaled, resnet, resnet_last_activate)[-1]

  num_paths = 5
  path_inits = (np.random.rand(num_paths, 2) * 2 - 1) * axis_size * 0.9
  sgd_paths = []
  losses = []
  lrs = []

  Y_labels = add_noise(Y, noise_sd) if ADD_NOISE else Y

  if PLOT_SGD:
    for path_init in path_inits:
      if TEST_NET:
        model = ClassicalNet([get_rand((2,2), test_sd, rand_dist) for _ in range(2)], weights_dist, test_net, scaled, resnet, resnet_last_activate, axis_size, unbound=True)
      elif weights_dist in nets:
        model = nets[weights_dist](*path_init, weights_dist, test_net, scaled, resnet, resnet_last_activate, batch_size, axis_size)
      else:
        model = ClassicalNet(form_weights(*path_init, fixed, weights_dist), weights_dist, test_net, scaled, resnet, resnet_last_activate, axis_size)

      path, _losses, lrs = train(epochs, model, X, Y, fixed,
                                lr_min, lr_max, epochs,
                                num_samples, batch_size,
                                test_net, force_map_sgd,
                                scaled, resnet, resnet_last_activate)
      sgd_paths.append(path)
      losses.append(_losses)

  if PLOT_SURFACE: plot(*parameters, fixed, X, Y_labels, weights_dist, scaled, resnet, resnet_last_activate, axis_size, save_plot, fn, sgd_paths if PLOT_SGD else None)
  if PLOT_LOSSES: plot_losses(losses, epochs)
  if PLOT_LR: plot_lrs(lrs, epochs, plot_log=False)

def grid_search(**kwargs):
  i = 0
  l = functools.reduce(operator.mul, map(len, kwargs.values()), 1)
  for e in itertools.product(*kwargs.values()):
    i += 1
    print(f"Running {i}/{l}", end='\r')
    run(**dict(zip(kwargs.keys(), e)), save_plot=True)

if __name__ == '__main__':
  grid_search(
    weights_dist=['first', 'chebyshev', 'rotational'],
    resnet=[True, False],
  )
