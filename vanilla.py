import numpy as np
import matplotlib.pyplot as plt

rand = np.random.normal(0, 1, (2, 2))

a = [rand[0,0], rand[0,1], rand[1,1]]

sax = lambda a, x: np.matmul([[a[0], a[1]], [a[1], a[2]]], x)

X = np.random.normal(0, 1, (2, 10))
Y = sax(a, X)

lin = np.linspace(-3, 3, 300)

loss = lambda a, X, Y: sum(((sax(a, X) - Y) ** 2).flatten())


landscape = []
for i in lin:
  landscape.append([])
  for j in lin:
    landscape[-1].append(loss([i, j, a[2]], X, Y))

plt.contour(lin, lin, landscape, levels=10)
plt.show()
