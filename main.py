import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import itertools, functools, operator
from concurrent.futures import ProcessPoolExecutor as Pool
import tqdm

PLOT_SURFACE = True
PLOT_2D = True
PLOT_SGD = True
PLOT_LOSSES = False
PLOT_LR = False
PLOT_SCATTER = False

CHECK_GRAD = False

TEST_NET = False
NUM_FREE_PARAMS = 8
FORCE_MAP_SGD = False

NUM_RUNS = 5
NUM_SAMPLES = 1000
BATCH_SIZE = 10

WEIGHTS_DIST = 'chebyshev'
RESNET = False

NOISE_SD = 0.0

LR_MIN = 0.001
LR_MAX = 0.03
EPOCHS = 15

COLORS = ['r', 'b', 'g', 'brown', 'm', 'darkviolet'] * 5

FOLDER = 'figs'

# Parameter range, sample range, axis range
dimension_defaults = {
  'first': [2, 2, 5],
  'rotational': [1, np.pi, 5],
  'chebyshev': [0.9, 0.9, 0.9],
}

starting_positions = {
  'first': [-3, -3],
  'rotational': [-1, 3],
  'chebyshev': [-0.5, -0.6],
}

scaled_defaults = {
  'first': False,
  'rotational': False,
  'chebyshev': True,
}

seeds = {
  'first': 1,
  'rotational': 1,
  'chebyshev': 6, #0, 3
}

view_angle_defaults = {
  'first': [65, 60],
  'rotational': [75, 210],
  'chebyshev': [60, 30],
}

RAND_DIST = 'uniform'

parameter_positions = {
  'first': [(0, 0, 0), (0, 0, 1)],
  'rotational': [],
  'chebyshev': [(0, 0, 1), (1, 0, 1)],
}

'''
Values in the weight distribution to be frozen during unbound SGD.

Structured to be visually intuitive, rather than the structure of the matrix.

Number represents the number of free params.

1 represents a randomised parameter, 0 is fixed.
'''
fixed_param_config = {
  '8': [
    [[1, 1], [1, 1]],
    [[1, 1], [1, 1]],
  ],
  'first3': [
    [[1, 1], [0, 0]],
    [[1, 0], [0, 0]],
  ],
  'chebyshev2': [
    [[0, 0], [0, 0]],
    [[0, 1], [0, 1]],
  ],
  'chebyshev6': [
    [[0, 1], [0, 1]],
    [[1, 1], [1, 1]],
  ],
}

sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

class Net:
  def __init__(self, i, j, dist, fixed, test_net, num_free_params, scaled, resnet, parameter_sd, axis_size):
    self.i = i
    self.j = j
    self.w = self.form_weights(i, j, fixed)
    self.dist = dist
    self.fixed = fixed
    self.test_net = test_net
    self.num_free_params = num_free_params
    self.scaled = scaled
    self.resnet = resnet
    self.parameter_sd = parameter_sd
    self.axis_size = axis_size

    if test_net:
      self.randomise_free_vars()

  def forward(self, x, grad=True):
    if grad:
      self.a1, self.a2, self.h2 = forward(x, self.w, self.scaled, self.resnet)
      return self.h2
    else:
      return forward(x, self.w, self.scaled, self.resnet)[-1]

  def backward(self, x, y, lr):

    if self.i < -self.axis_size or self.i > self.axis_size or self.j < -self.axis_size or self.j > self.axis_size:
      return

    # Derivative of functional matrix
    dw1i = self.derivs['w1']['i'](self.i)
    dw1j = self.derivs['w1']['j'](self.j)
    dw2i = self.derivs['w2']['i'](self.i)
    dw2j = self.derivs['w2']['j'](self.j)

    h1 = apply_scaling(self.a1 + x if self.resnet else self.a1, self.scaled, self.resnet)

    da1z1 = self.a1 * (1 - self.a1)
    dh1a1 = deriv_scaling(self.scaled, self.resnet)
    dz2h1 = self.w[1]
    da2z2 = self.a2 * (1 - self.a2)
    dh2a2 = 1
    dlh2 = self.h2 - y

    def dl_(dw1, dw2):
      dz1 = dw1 @ x

      da1 = da1z1 * dz1
      dh1 = dh1a1 * da1

      dz2 = dw2 @ h1 + self.w[1] @ dh1
      if self.resnet:
        da2 = da2z2 * dz2
        dh2 = dh2a2 * da2 + dh1
      else:
        da2 = dz2
        dh2 = dh2a2 * da2

      dl = dlh2 * dh2

      avg_d = np.mean(dl)

      return dw1, dz1, da1, dh1, dw2, dz2, da2, dh2, avg_d

    grads_i = dl_(dw1i, dw2i)
    grads_j = dl_(dw1j, dw2j)

    if CHECK_GRAD:
      self.check_grad(x, y, grads_i, grads_j)

    di = grads_i[-1]
    dj = grads_j[-1]

    self.i -= di * lr
    self.j -= dj * lr

    dw_free = [None] * 2
    dw_free[1] = dlh2 @ h1.T
    dw_free[0] = dz2h1.T @ dlh2 * da1z1 @ x.T

    new_w = self.form_weights(self.i, self.j, self.fixed)

    if self.test_net:
      positions = get_fixed_param_config(self.dist, self.num_free_params)
      for e in itertools.product(*([[0, 1]] * 3)):
        if positions[e[1]][e[0]][e[2]]:
          self.w[e[0]][e[1]][e[2]] -= lr * dw_free[e[0]][e[1]][e[2]]
        else:
          self.w[e[0]][e[1]][e[2]] = new_w[e[0]][e[1]][e[2]]
    else:
      self.w = new_w

  def get_parameters(self):
    return self.i, self.j

  def randomise_free_vars(self):
    if self.dist == 'rotational':
      return
    positions = get_fixed_param_config(self.dist, self.num_free_params)
    params_pos = parameter_positions[self.dist]
    for e in itertools.product(*([[0, 1]] * 3)):
      if positions[e[1]][e[0]][e[2]] and e not in params_pos:
        self.w[e[0]][e[1], e[2]] = get_rand(1, self.parameter_sd, 'uniform')

  def check_grad(self, x, y, grads_i=None, grads_j=None, eps=1e-6):
    if not grads_i or not grads_j:
      raise Exception("Grads required")
    i, j = self.get_parameters()
    v = self._get_forward_vals(x, y, i, j)
    # Generate augmented weights based on new params
    vi = self._get_forward_vals(x, y, i + eps, j)
    vj = self._get_forward_vals(x, y, i, j + eps)

    fdi = (vi - v) / eps
    fdj = (vj - v) / eps

    def print_diff(fd, grads, c):
      print(f"=== Grads {c} ===")
      for name, d, fd in zip([f"dw1{c}", f"dz1{c}", f"da1{c}", f"dh1{c}", f"dw2{c}", f"dz2{c}", f"da2{c}", f"dh2{c}", f"dl{c}"], fd, grads):
        l = loss(d, fd) if isinstance(d, np.ndarray) else (d - fd)**2
        print(name, l)

    print_diff(fdi, grads_i, 'i')
    print_diff(fdj, grads_j, 'j')

  def _get_forward_vals(self, x, y, *free_params):
    w_ = self.form_weights(*free_params, self.fixed)

    vals = _forward(x, w_, self.scaled, self.resnet)
    l = loss(vals[-1], y)

    return np.array([*vals, l])

class FirstNet(Net):
  def __init__(self, *args):
    super().__init__(*args)
    self.derivs = {
      'w1': {
        'i': lambda a: matrix([1, 0, 0, 0]),
        'j': lambda a: matrix([0, 1, 1, 0]),
      },
      'w2': {
        'i': lambda a: np.zeros((2, 2)),
        'j': lambda a: np.zeros((2, 2))
      }
    }

  @staticmethod
  def form_weights(i, j, fixed):
    return matrix([i, j, j, fixed[0]]), matrix(fixed[1:5])

class RotationalNet(Net):
  def __init__(self, *args):
    super().__init__(*args)
    self.derivs = {
      'w1': {
        'i': lambda a: matrix([-np.sin(a), -np.cos(a), np.cos(a), -np.sin(a)]),
        'j': lambda a: np.zeros((2, 2)),
      },
      'w2': {
        'i': lambda a: np.zeros((2, 2)),
        'j': lambda a: matrix([-np.sin(a), -np.cos(a), np.cos(a), -np.sin(a)])
      }
    }

  @staticmethod
  def form_weights(i, j, fixed):
    l = lambda a: matrix([np.cos(a), -np.sin(a), np.sin(a), np.cos(a)])
    return l(i), l(j)

class ChebyshevNet(Net):
  def __init__(self, *args):
    super().__init__(*args)

    self.derivs = {
      'w1': {
        'i': lambda a: matrix([0, 1, 4*a, 12*(a**2) - 3]),
        'j': lambda a: np.zeros((2, 2)),
      },
      'w2': {
        'i': lambda a: np.zeros((2, 2)),
        'j': lambda a: matrix([0, 1, 4*a, 12*(a**2) - 3])
      }
    }

  @staticmethod
  def form_weights(i, j, fixed):
    l = lambda a: matrix([1, a, 2*(a**2) - 1, 4*(a**3) - 3*a])
    return l(i), l(j)

  def get_parameters(self):
    return self.w[0][0, 1], self.w[1][0, 1]

def matrix(a):
  return np.array([[a[0], a[1]], [a[2], a[3]]])

def apply_scaling(m, scaled, resnet):
  if not scaled:
    return m
  if resnet:
    return (m * 2/3) - 1/3
  return 2*m-1

def deriv_scaling(scaled, resnet):
  return 1 if not scaled else 2/3 if resnet else 2

def _forward(x, w, scaled, resnet):
  zeros = np.zeros(np.shape(x))
  skip = x if resnet else zeros
  w1 = w[0]
  z1 = w1 @ x
  a1 = sigmoid(z1)
  h1 = apply_scaling(a1 + skip, scaled, resnet)

  w2 = w[1]
  z2 = w2 @ h1
  if resnet:
    a2 = sigmoid(z2)
    h2 = a2 + h1
  else:
    a2 = h2 = z2 + skip
  return w1, z1, a1, h1, w2, z2, a2, h2

def forward(*args):
  w1, z1, a1, h1, w2, z2, a2, h2 = _forward(*args)
  return a1, a2, h2

def loss(y_hat, Y):
  return 0.5 * sum(((y_hat - Y) ** 2).flatten()) / len(Y[0])

def anneal_lr(epoch, lr_min, lr_max, max_epochs):
  return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / max_epochs))

def train(epochs, m, X, Y, valid_X, valid_Y, fixed, dist,
          lr_min, lr_max,
          num_samples, batch_size,
          test_net, force_map_sgd,
          scaled, resnet):
  sgd_path = []
  losses = []
  valid_losses = []
  lrs = []

  for epoch in range(epochs):
    lr = anneal_lr(epoch + 1, lr_min, lr_max, epochs)
    lrs.append(lr)
    for _ in range(num_samples // batch_size):
      idx = np.random.choice(np.arange(num_samples), batch_size, replace=False)

      batchx = X[:,idx]
      batchy = Y[:,idx]

      m.forward(batchx)

      # Update network, caclulate loss for plotting
      m.backward(batchx, batchy, lr)

      new_params = m.get_parameters()

      if test_net and force_map_sgd:
        y_hat = forward(X, nets[dist].form_weights(*new_params, fixed), scaled, resnet)[-1]
      else:
        y_hat = m.forward(X, grad=False)
      new_loss = loss(y_hat, Y)

      valid_y_hat = m.forward(valid_X, grad=False)
      valid_loss = loss(valid_y_hat, valid_Y)

      path = (*new_params, new_loss)
      sgd_path.append(path)
      losses.append(new_loss)
      valid_losses.append(valid_loss)

  return sgd_path, losses, valid_losses, lrs

def calc_loss(i, j, fixed, X, Y, Net, *args):
  y_hat = forward(X, Net.form_weights(i, j, fixed), *args)[-1]
  return loss(y_hat, Y)

def create_landscape(axis, *args):
  return [[calc_loss(i, j, *args) for i in axis] for j in axis]

def get_fixed_param_config(dist, n):
  d = '8' if n == 8 else dist + str(n)
  return fixed_param_config[d]

def get_file_path(weights_dist, epochs, test_net, num_free_params, num_samples, batch_size, parameter_sd, sample_sd,
                  axis_size, test_sd, scaled, resnet, lr_min, lr_max,
                  force_map_sgd, noise_sd, rand_dist, subfolder):
  sf = f"/{subfolder}" if subfolder else ''
  return f"{FOLDER}{sf}/{weights_dist}_T{test_net}_NFP{num_free_params}_FM{force_map_sgd}_RN{resnet}_E{epochs}_NS{num_samples}_BS{batch_size}_PSD{parameter_sd}\
_SSD{round(sample_sd, 2)}_AX{round(axis_size, 2)}_TSD{test_sd}_S{scaled}\
_LR{lr_min}-{lr_max}_NSD{noise_sd}_RD{rand_dist}_SGD{PLOT_SGD}"

'''
  Plot 3D contours of the loss landscape
'''
def plot(i, j, fixed, X, Y, dist, Net, scaled, resnet, true_loss, axis_size, save, filepath, paths=None, plot_2d=False):
  fig = plt.figure()
  ax = fig.gca(projection='3d')

  ax_max = axis_size
  ax_min = -axis_size

  if paths:
    ax_max = max(ax_max, max(max((x, y) for path in paths for (x, y, _) in path)))
    ax_min = min(ax_min, min(min((x, y) for path in paths for (x, y, _) in path)))

  axis = np.arange(ax_min, ax_max, axis_size/10)
  landscape = create_landscape(axis, fixed, X, Y, Net, scaled, resnet)
  Z = np.array(landscape)
  axis_x, axis_y = np.meshgrid(axis, axis)

  ax.plot_surface(axis_x, axis_y, Z, cmap=cm.terrain, linewidth=0, alpha=0.6)
  ax.scatter(i, j, 0, marker='x', s=150, color='black')
  if true_loss:
    ax.scatter(i, j, true_loss, marker='+', s=150, color='black')
  if paths:
    for idx, path in enumerate(paths):
      ax.plot(*zip(*path), color=COLORS[idx])
      ax.scatter(*path[-1], marker='o', color=COLORS[idx])

  ax.axes.set_zlim3d([0, np.max(Z)])
  plt.xlabel('α')
  plt.ylabel('β')
  ax.set_zlabel('Loss')

  elevation = view_angle_defaults[dist][0]
  azimuth = view_angle_defaults[dist][1]

  ax.view_init(elev=elevation, azim=azimuth)

  if save:
    plt.savefig(f"{filepath}.png")
  else:
    plt.show()

  if plot_2d:
    plt.clf()
    plt.contour(axis, axis, landscape, levels=20, cmap=cm.terrain)
    plt.scatter(i, j, marker='x', s=150, color='black')
    if paths:
      for i, path in enumerate(paths):
        params = list(zip(*path))
        xs, ys = params[0], params[1]
        plt.plot(xs, ys, color=COLORS[i])
        plt.scatter(xs[-1], ys[-1], color=COLORS[i], marker='o')

    if save:
      plt.savefig(f"{filepath}_2D.png")
    else:
      plt.show()

def plot_scatter(scatters, y=False, filename=None):
  plt.clf()
  if y:
    for i, scatter in enumerate(scatters):
      plt.scatter(*zip(scatter), color=COLORS[i], s=2)
  else:
    plt.scatter(*zip(scatters), color='b')
  if filename:
    plt.savefig(filename)
  else:
    plt.show()

'''
  Plot chart of loss over epochs
'''
def plot_losses(losses, validation=None):
  epoch_axis = np.arange(1, len(losses[0]) + 1)
  for loss_plt, valid_plt in zip(losses, validation):
    plt.plot(epoch_axis, loss_plt, label='Training loss')
    plt.plot(epoch_axis, valid_plt, label='Validation loss')
  plt.yscale('log')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()

def plot_lrs(lrs, plot_log=True):
  epoch_axis = np.arange(1, len(lrs) + 1)
  plt.plot(epoch_axis, lrs)
  if plot_log:
    plt.yscale('log')
    plt.ylim(1e-4, LR_MAX * 1.1)
  plt.xlabel('Epochs')
  plt.ylabel('Learning rate (ηt)')
  plt.show()

def print_weights(ws, dist, nfp, true=False):
  col = lambda x, c: f"\\textcolor{{{c}}}{{{x}}}"
  params_pos = parameter_positions[dist]
  free_params = get_fixed_param_config(dist, nfp)

  def f(i, j, k):
    x = round(ws[i][j, k], 3)

    if (i, j, k) == params_pos[0]:
      return col(x, 'orange')
    elif (i, j, k) == params_pos[1]:
      return col(x, 'magenta')
    elif free_params[j][i][k] and not True:
      return col(x, 'cyan')
    return x

  for i in range(2):
    print('\\begin{bmatrix}')
    print(f"  {f(i, 0, 0)} & {f(i, 0, 1)} \\\\")
    print(f"  {f(i, 1, 0)} & {f(i, 1, 1)}")
    print('\\end{bmatrix}')

def get_rand(shape, sd, dist):
  if dist == 'normal':
    return np.random.normal(0, sd, shape)
  elif dist == 'uniform':
    return np.random.uniform(-sd, high=sd, size=shape)

def noise(Y, noise_sd):
  n = np.random.normal(0, noise_sd, np.shape(Y))
  return Y + n

nets = {
  'first': FirstNet,
  'chebyshev': ChebyshevNet,
  'rotational': RotationalNet,
}

def run(weights_dist=WEIGHTS_DIST,
        epochs=EPOCHS,
        test_net=TEST_NET,
        num_free_params=NUM_FREE_PARAMS,
        num_samples=NUM_SAMPLES,
        batch_size=BATCH_SIZE,
        num_runs=NUM_RUNS,
        parameter_sd=None,
        sample_sd=None,
        axis_size=None,
        test_sd=None,
        scaled=None,
        resnet=RESNET,
        lr_min=LR_MIN,
        lr_max=LR_MAX,
        force_map_sgd=FORCE_MAP_SGD,
        noise_sd=NOISE_SD,
        rand_dist=RAND_DIST,
        sgd_same_point=False,
        save_plot=False,
        subfolder=None,
        verbose=False):

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

  fn = get_file_path(weights_dist, epochs, test_net, num_free_params, num_samples, batch_size, parameter_sd, sample_sd,
                    axis_size, test_sd, scaled, resnet, lr_min, lr_max,
                    force_map_sgd, noise_sd, rand_dist, subfolder)

  np.random.seed(seeds[weights_dist])

  parameters = get_rand(2, parameter_sd, rand_dist)
  fixed = get_rand(6, parameter_sd, 'uniform')

  # Hard-code values for W_2
  # fixed[1:5] = [1, 0.001, 0.001, 0.002]

  Net = nets[weights_dist]
  true_params = Net.form_weights(*parameters, fixed)

  X = get_rand((2, num_samples), sample_sd, rand_dist)
  Y = forward(X, true_params, scaled, resnet)[-1]

  path_inits = [starting_positions[weights_dist]] * num_runs if sgd_same_point else get_rand((num_runs, 2), axis_size * 0.9, 'uniform')

  if PLOT_SCATTER:
    plot_scatter(X, filename=f"{fn}_SCAT-X.png")
    plot_scatter([Y], True, f"{fn}_SCAT-Y.png")

  Y_noise = noise(Y, noise_sd)
  true_loss = loss(Y, Y_noise)
  Y = Y_noise

  if verbose:
    print("True params")
    print_weights(true_params, weights_dist, num_free_params, true=True)
    print("Eigenvalues", np.linalg.eigvals(true_params[1]))

  valid_X = get_rand((2, num_samples), sample_sd, rand_dist)
  valid_Y = forward(valid_X, true_params, scaled, resnet)[-1]
  valid_Y = noise(valid_Y, noise_sd)

  if PLOT_SGD:
    sgd_paths, loss_runs, valid_loss_runs, lrs, scatters = run_sgd_multi(path_inits, Net, epochs, X, Y, valid_X, valid_Y,
                                                                        fixed, weights_dist, test_net, num_free_params,
                                                                        lr_min, lr_max, num_samples, batch_size,
                                                                        parameter_sd, axis_size,
                                                                        force_map_sgd, scaled, resnet, verbose)

  if PLOT_SURFACE: plot(*parameters, fixed, X, Y, weights_dist, Net, scaled, resnet, true_loss, axis_size, save_plot, fn, sgd_paths if PLOT_SGD else None, PLOT_2D)
  if PLOT_LOSSES: plot_losses(losses, validation=valid_loss_runs)
  if PLOT_LR and not save_plot: plot_lrs(lrs, plot_log=False)

  if test_net:
    if PLOT_SCATTER:
      # Use valid parameters to generate valid labels
      plot_scatter(scatters, True, f"{fn}_SCAT-Y_.png")

def run_sgd_multi(path_inits, *args):
  arg_set = [[i, p] + list(args) for i, p in enumerate(path_inits)]
  with Pool() as pool:
    runs = list(starmap_kwargs(pool, run_sgd, args_iter=arg_set, kw=False))
  return zip(*runs)

def run_sgd(i, path_init, Net, epochs, X, Y, valid_X, valid_Y, fixed, weights_dist, test_net, num_free_params,
            lr_min, lr_max,
            num_samples, batch_size,
            parameter_sd, axis_size,
            force_map_sgd, scaled, resnet, verbose):

  model = Net(*path_init, weights_dist, fixed, test_net, num_free_params, scaled, resnet, parameter_sd, axis_size)
  path, loss_run, valid_loss_run, lrs = train(epochs, model, X, Y, valid_X, valid_Y, fixed, weights_dist,
                                          lr_min, lr_max,
                                          num_samples, batch_size,
                                          test_net, force_map_sgd,
                                          scaled, resnet)

  if verbose:
    print(f"\nRun {i+1}")
    print("Params", model.get_parameters())
    print_weights(model.w, weights_dist, num_free_params)

    if test_net:
      print("Training loss, validation loss", (loss_run[-1], valid_loss_run[-1]))

  scatter = model.forward(X, grad=False)
  return path, loss_run, valid_loss_run, lrs, scatter

def starmap_kwargs(pool, f, args_iter=[], kwargs_iter={}, kw=True):
  if kw:
    args_for_starmap = zip(itertools.repeat(f), itertools.repeat(args_iter), kwargs_iter)
  else:
    args_for_starmap = zip(itertools.repeat(f), args_iter, itertools.repeat(kwargs_iter))
  return pool.map(apply_args, args_for_starmap)

def apply_args(f_args):
  f, args, kwargs = f_args
  return f(*args, **kwargs)

def grid_search(**kwargs):
  kwargs['save_plot'] = [True]
  arg_set = list(map(lambda args: dict(zip(kwargs.keys(), args)), itertools.product(*kwargs.values())))
  with Pool() as pool:
    starmap_kwargs(pool, run, kwargs_iter=arg_set)

if __name__ == '__main__':
  # first, rotational, chebyshev
  # run(weights_dist='chebyshev', noise_sd=0.2, num_runs=5, test_net=True, num_free_params=2)

  grid_search(weights_dist=['chebyshev'],
              lr_max=[0.1],
              # noise_sd=[0.2],
              test_net=[True],
              num_free_params=[2, 6],
              subfolder=['unstructured'])
