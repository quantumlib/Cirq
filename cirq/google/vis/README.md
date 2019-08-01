This directory contains convenience functions for visualizations.

The public functions are:

* heatmap(): Plot a per-qubit metric as a heatmap.

More to be added later.

# Demo plots

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
