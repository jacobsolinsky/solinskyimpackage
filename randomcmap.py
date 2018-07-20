from numpy.random import shuffle
from numpy import linspace
from matplotlib.pyplot.cm.colors import ListedColormap
from matplotlib.pyplot.cm import vals
vals = linspace(0, 1, 10000)
shuffle(vals)
cmap = ListedColormap(jet(vals))
