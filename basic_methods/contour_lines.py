# Program which prints contour_lines of a function

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

x = np.linspace(-6, 6, 100)

y = np.linspace(-6, 6, 100)

X, Y = np.meshgrid(x, y)


def my_function(X, y):
	return 8 * X + 12 * y + X ** 2 - 2 * Y ** 2


Z = my_function(X, y)
# Create contour lines or level curves using matplotlib.pyplot module

fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)

ax.clabel(CS, inline=True, fontsize=10)
ax.set_title('Countour lines')
plt.show()
