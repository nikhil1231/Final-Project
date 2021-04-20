import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import itertools, functools, operator
from os.path import exists

PLOT_SURFACE = True
PLOT_2D = True
PLOT_SGD = True
PLOT_LOSSES = False
PLOT_LR = False
PLOT_SCATTER = False

TEST_NET = False
FORCE_MAP_SGD = False

NUM_RUNS = 3
NUM_SAMPLES = 1000
BATCH_SIZE = 1

WEIGHTS_DIST = 'chebyshev'
RESNET = False
RESNET_LAST_ACTIVATE = False

ADD_NOISE = False
NOISE_SD = 0.05

LR_MIN = 0.001
LR_MAX = 0.03
EPOCHS = 10_000

COLORS = ['r', 'b', 'g']

# Parameter range, sample range, axis range
dimension_defaults = {
  'first': [1, 1, 1],
  'second': [1, 1, 5],
  'equal': [1, 1, 1],
  'rotational': [1, np.pi, 5],
  'skew': [1, 1, 1],
  'chebyshev': [0.9, 0.9, 0.9],
}

starting_positions = {
  'first': [-0.5, -0.6],
  'second': [-0.5, -0.6],
  'equal': [-0.5, -0.6],
  'rotational': [-0.5, -0.6],
  'skew': [-0.5, -0.6],
  'chebyshev': [-0.5, -0.6],
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

Structured to be visually intuitive, rather than the structure of the matrix.

Number represents the number of free params.

1 represents a randomised parameter, 0 is fixed.
'''
fixed_param_config = {
  'first': [
    [[1, 1], [0, 0]],
    [[1, 0], [0, 0]],
  ],
  'chebyshev': [
    [[0, 1], [0, 1]],
    [[1, 1], [1, 1]],
  ],
}

sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

class Net:
  def __init__(self, dist, test_net, scaled, resnet, resnet_last_activate, parameter_sd, axis_size):
    self.dist = dist
    self.test_net = test_net
    self.scaled = scaled
    self.resnet = resnet
    self.resnet_last_activate = resnet_last_activate
    self.parameter_sd = parameter_sd
    self.axis_size = axis_size

    if test_net:
      self.randomise_free_vars()

  def forward(self, x):
    self.a1, self.a2 = forward(x, self.w, self.scaled, self.resnet, self.resnet_last_activate)
    return self.a2

  def backward(self, x, y, lr):
    raise Exception("Backprop not implemented")

  def get_parameters(self):
    raise Exception("Get params not implemented")

  def randomise_free_vars(self):
    positions = fixed_param_config[self.dist]
    for e in itertools.product(*([[0, 1]] * 3)):
      if positions[e[1]][e[0]][e[2]]:
        self.w[e[0]][e[1], e[2]] = get_rand(1, self.parameter_sd, 'uniform')

class ClassicalNet(Net):
  def __init__(self, w, dist, test_net, scaled, resnet, resnet_last_activate, parameter_sd, axis_size):
    self.w = w
    super().__init__(dist, test_net, scaled, resnet, resnet_last_activate, parameter_sd, axis_size)

  def backward(self, x, y, lr):

    dw = [None] * 2

    error = self.a2 - y

    if self.scaled:
      dw[1] = error @ (2*self.a1.T - 1)
    else:
      dw[1] = error @ self.a1.T

    w2 = self.w[1] + np.identity(2) if self.resnet else self.w[1]

    dw[0] = w2.T @ error * self.a1 * (1 - self.a1) @ x.T

    dw[0] /= len(x[0])
    dw[1] /= len(x[0])

    if self.test_net:
      positions = fixed_param_config[self.dist]
      for e in itertools.product(*([[0, 1]] * 3)):
        if positions[e[1]][e[0]][e[2]]:
          self.w[e[0]][e[1], e[2]] -= lr * dw[e[0]][e[1], e[2]]
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

class FunctionalNet(Net):
  def __init__(self, i, j, dist, test_net, scaled, resnet, resnet_last_activate, parameter_sd, axis_size):
    self.i = i
    self.j = j
    self.w = form_weights(i, j, [0]*6, dist)
    super().__init__(dist, test_net, scaled, resnet, resnet_last_activate, parameter_sd, axis_size)
    self.derivs = None

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

    avg_dj = np.sum(d['j'].flatten()) / len(x[0])
    avg_di = np.sum(d['i'].flatten()) / len(x[0])

    self.j -= avg_dj * lr
    self.i -= avg_di * lr

    self.w = form_weights(self.i, self.j, [0]*6, self.dist)

  def get_parameters(self):
    return self.i, self.j

class RotationalNet(FunctionalNet):
  def __init__(self, i, j, dist, test_net, scaled, resnet, resnet_last_activate, parameter_sd, axis_size):
    super().__init__(i, j, dist, test_net, scaled, resnet, resnet_last_activate, parameter_sd, axis_size)
    d = lambda a: np.array([
      [-np.sin(a), -np.cos(a)],
      [np.cos(a), -np.sin(a)]
    ])
    self.derivs = [d, d]

class ChebyshevNet(FunctionalNet):
  def __init__(self, i, j, dist, test_net, scaled, resnet, resnet_last_activate, parameter_sd, axis_size):
    super().__init__(i, j, dist, test_net, scaled, resnet, resnet_last_activate, parameter_sd, axis_size)

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

def train(epochs, m, X, Y, fixed, dist,
          lr_min, lr_max, max_epochs,
          num_samples, batch_size,
          test_net, force_map_sgd,
          scaled, resnet, resnet_last_active):
  sgd_path = []
  losses = []
  lrs = []

  for epoch in range(epochs):
    idx = np.random.choice(np.arange(num_samples), batch_size, replace=False)

    batchx = X[:,idx]
    batchy = Y[:,idx]

    lr = anneal_lr(epoch, lr_min, lr_max, max_epochs)

    m.forward(batchx)

    # Update network, caclulate loss for plotting
    m.backward(batchx, batchy, lr)

    new_params = m.get_parameters()

    if test_net and force_map_sgd:
      y_hat = forward(X, form_weights(*new_params, fixed, dist), scaled, resnet, resnet_last_active)[-1]
    else:
      y_hat = m.forward(X)
    new_loss = loss(y_hat, Y)

    path = (*new_params, new_loss)
    sgd_path.append(path)
    losses.append(new_loss)
    lrs.append(lr)

  return sgd_path, losses, lrs

def calc_loss(i, j, fixed, X, Y, dist, scaled, resnet, resnet_last_active):
  y_hat = forward(X, form_weights(i, j, fixed, dist), scaled, resnet, resnet_last_active)[-1]
  return loss(y_hat, Y)

def create_landscape(axis, fixed, X, Y, dist, scaled, resnet, resnet_last_active):
  return [[calc_loss(i, j, fixed, X, Y, dist, scaled, resnet, resnet_last_active) for i in axis] for j in axis]

'''
  Plot 3D contours of the loss landscape
'''
def plot(i, j, fixed, X, Y, dist, scaled, resnet, resnet_last_active, axis_size, save, filepath, paths=None, plot_2d=False):
  fig = plt.figure()
  ax = fig.gca(projection='3d')

  axis = np.arange(-axis_size, axis_size, axis_size/100)
  landscape = create_landscape(axis, fixed, X, Y, dist, scaled, resnet, resnet_last_active)
  Z = np.array(landscape)
  axis_x, axis_y = np.meshgrid(axis, axis)

  ax.plot_surface(axis_x, axis_y, Z, cmap=cm.terrain, linewidth=0, alpha=0.7)
  ax.scatter(i, j, 0, marker='x', s=150, color='black')
  if paths:
    for i, path in enumerate(paths):
      ax.plot(*zip(*path), color=COLORS[i])
      ax.scatter(*path[-1], marker='o', color=COLORS[i])

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
    plt.savefig(f"{filepath}.png")
  else:
    plt.show()

  if plot_2d:
    plt.contour(axis, axis, landscape, levels=20, cmap=cm.terrain)
    if paths:
      for i, path in enumerate(paths):
        params = list(zip(*path))
        xs, ys = params[0], params[1]
        plt.plot(xs, ys, color=COLORS[i])
        plt.scatter(xs[-1], ys[-1], color=COLORS[i], marker='o')

    plt.xlim([-axis_size, axis_size])
    plt.ylim([-axis_size, axis_size])

    if save:
      plt.savefig(f"{filepath}_2D.png")
    else:
      plt.show()

def get_file_path(weights_dist, epochs, test_net, num_samples, batch_size, parameter_sd, sample_sd,
                  axis_size, test_sd, scaled, resnet, resnet_last_activate, lr_min, lr_max,
                  force_map_sgd, add_noise, noise_sd, rand_dist):
  return f"figs/{weights_dist}_SGD{PLOT_SGD}_E{epochs}_T{test_net}_NS{num_samples}_BS{batch_size}_PSD{parameter_sd}\
_SSD{round(sample_sd, 2)}_AX{round(axis_size, 2)}_TSD{test_sd}_S{scaled}_RN{resnet}_RNL{resnet_last_activate}\
_LR{lr_min}-{lr_max}_FM{force_map_sgd}_N{add_noise}_NSD{noise_sd}_RD{rand_dist}"

def plot_scatter(points, y=False):
  plt.scatter(*zip(points), color='r' if y else 'b')
  plt.show()

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

def noise(Y, noise_sd):
  n = np.random.normal(0, noise_sd, np.shape(Y))
  return Y + n

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
        add_noise=ADD_NOISE,
        noise_sd=NOISE_SD,
        rand_dist=RAND_DIST,
        sgd_same_point=False,
        save_plot=False):

  if parameter_sd is None:
    parameter_sd = dimension_defaults[weights_dist][0]
  if sample_sd is None:
    sample_sd = dimension_defaults[weights_dist][1]
  if axis_size is None:
    axis_size = dimension_defaults[weights_dist][2]
  if test_sd is None:
    test_sd = axis_size
  if scaled is None:
    scaled = scaled_defaults[weights_dist]

  fn = get_file_path(weights_dist, epochs, test_net, num_samples, batch_size, parameter_sd, sample_sd,
                    axis_size, test_sd, scaled, resnet, resnet_last_activate, lr_min, lr_max,
                    force_map_sgd, add_noise, noise_sd, rand_dist)
  if save_plot and exists("{fn}.png"):
    return

  np.random.seed(seeds[weights_dist])

  parameters = get_rand(2, parameter_sd, rand_dist)
  fixed = get_rand(6, parameter_sd, 'uniform')

  X = get_rand((2, num_samples), sample_sd, rand_dist)
  Y = forward(X, form_weights(*parameters, fixed, weights_dist), scaled, resnet, resnet_last_activate)[-1]

  path_inits = get_rand((NUM_RUNS, 2), axis_size * 0.9, 'uniform')
  sgd_paths = []
  losses = []
  lrs = []

  if PLOT_SCATTER:
    plot_scatter(X)
    plot_scatter(Y, True)

  Y = noise(Y, noise_sd) if add_noise else Y

  if PLOT_SGD:
    for path_init in path_inits:
      if sgd_same_point:
        path_init = starting_positions[weights_dist] # Start all from same point
      if test_net or weights_dist not in nets:
        model = ClassicalNet(form_weights(*path_init, fixed, weights_dist), weights_dist, test_net, scaled, resnet, resnet_last_activate, parameter_sd, axis_size)
      else:
        model = nets[weights_dist](*path_init, weights_dist, test_net, scaled, resnet, resnet_last_activate, parameter_sd, axis_size)
      path, _losses, lrs = train(epochs, model, X, Y, fixed, weights_dist,
                                lr_min, lr_max, epochs,
                                num_samples, batch_size,
                                test_net, force_map_sgd,
                                scaled, resnet, resnet_last_activate)
      sgd_paths.append(path)
      losses.append(_losses)

  if PLOT_SURFACE: plot(*parameters, fixed, X, Y, weights_dist, scaled, resnet, resnet_last_activate, axis_size, save_plot, fn, sgd_paths if PLOT_SGD else None, PLOT_2D)
  if PLOT_LOSSES: plot_losses(losses, epochs)
  if PLOT_LR: plot_lrs(lrs, epochs, plot_log=False)

  if test_net:
    if PLOT_SCATTER:
      # Use new parameters to generate new labels
      free_Y = model.forward(X)
      plot_scatter(free_Y, True)

    new_X = get_rand((2, num_samples), sample_sd, rand_dist)
    new_Y = forward(new_X, form_weights(*parameters, fixed, weights_dist), scaled, resnet, resnet_last_activate)[-1]

    y_hat = model.forward(new_X)

    loss_ = loss(y_hat, new_Y)

    print(loss_)


def grid_search(**kwargs):
  i = 0
  l = functools.reduce(operator.mul, map(len, kwargs.values()), 1)
  for e in itertools.product(*kwargs.values()):
    i += 1
    print(f"Running {i}/{l}", end='\r')
    run(**dict(zip(kwargs.keys(), e)), save_plot=True)

if __name__ == '__main__':
  # grid_search(
  #   weights_dist=['chebyshev'],
  #   add_noise=[True, False],
  # )
  run(weights_dist='chebyshev',
      add_noise=True,
      test_net=True,
      num_samples=1000)
