import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()

    self.l1 = nn.Linear(2, 2)
    self.activation = nn.ReLU()

  def forward(self, x):
    x = self.l1(x)
    x = self.activation(x)
    return x

rand = np.random.normal(0, 1, (2, 2))

a = [rand[0,0], rand[0,1], rand[1,1]]

sax = lambda a, x: np.matmul([a[0], a[1]], x)

X = np.random.normal(0, 1, (1, 10))
Y = sax(a, X)

print(Y)

lin = np.linspace(-3, 3, 300)

def plot(loss_func):
  landscape = []
  for i in lin:
    landscape.append([])
    for j in lin:
      landscape[-1].append(loss_func([i, j, a[2]], X, Y))

  plt.contour(lin, lin, landscape, levels=10)
  plt.show()

# model = Model()
model = torch.nn.Sequential(nn.Linear(2, 1, bias=False))
MSELoss = nn.MSELoss()

model.eval()

def loss2(a, X, Y):
  model[0].weight[0] = a[0]
  preds = model(torch.tensor(X))
  loss = MSELoss(preds, torch.tensor(Y))
  return loss

print(loss2(a, X, Y))

loss = lambda a, X, Y: sum(((sax(a, X) - Y) ** 2).flatten())
# loss = lambda