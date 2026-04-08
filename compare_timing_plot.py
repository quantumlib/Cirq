import numpy as np
from matplotlib import pyplot as plt

t0 = np.loadtxt("timing0.txt")
t1 = np.loadtxt("timing1.txt")

lo = np.min([t0, t1])
hi = np.max([t0, t1])
edges = np.linspace(lo, hi, 26)

fig = plt.gcf()
fig.clf()
ax = fig.subplots(2, 1)
ax[0].hist(t0, edges)
ax[0].text(
    0.95,
    0.85,
    "before",
    horizontalalignment="right",
    transform=ax[0].transAxes,
)
ax[1].hist(t1, edges)
ax[1].text(
    0.95,
    0.85,
    "after",
    horizontalalignment="right",
    transform=ax[1].transAxes,
)
ax[1].set_xlabel("timeit get_initial_mapping (s)")
plt.setp(ax, xlim=(lo, hi))
