This directory contains convenience functions for visualizations.

The public functions are:

* class Heatmap: plot a per-qubit metric as a heatmap.

More to be added later.

# Demo plots

The heatmap figure is produced with the example codes below.

![heatmap image](https://github.com/quantumlib/Cirq/blob/master/cirq/google/vis/figures/bristlecone_heatmap_example.png)

```python
import matplotlib.pyplot as plt
import numpy as np
import cirq.google.vis.heatmap as vis_heatmap

value_map = {
    (qubit.row, qubit.col): np.random.random()
    for qubit in cirq.google.known_devices.Bristlecone.qubits
}

heatmap = vis_heatmap.Heatmap(value_map)
fig, ax = plt.subplots(figsize=(9, 9))
heatmap.plot(ax)
plt.show()
```
