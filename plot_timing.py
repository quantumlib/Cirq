# 2026-04-09 very hackish it takes too much time to clean this
import collections

import numpy as np
from matplotlib import pyplot as plt
import cirq

data = np.loadtxt("timing.txt")

t1 = collections.defaultdict(list)
t6 = collections.defaultdict(list)

for nc, t1i, t6i in data:
    t1[nc].append(t1i)
    t6[nc].append(t6i)

nchanges = np.array(sorted(t1.keys()))

fig, axs = plt.subplots(2, 3)
for ax, nc in zip(axs.flat, nchanges):
    cirq.integrated_histogram(t1[nc], ax=ax, color="blue")
    cirq.integrated_histogram(t6[nc], ax=ax, color="red", linestyle="--", lw=2)
    ax.set_title(f"changed-files-count = {int(nc)}")
    if ax is next(axs.flat):
        ax.legend(ax.lines[0:3:2], ["t1cpu", "t6cpu"])
plt.setp(axs[:, 1:], ylabel="")
plt.setp(axs[-1], xlabel="pytest run time (s)")
