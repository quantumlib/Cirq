"""The script that produces bristlecone_heatmap_example.png."""

import numpy as np
import matplotlib.pyplot as plt
import cirq


def main():
    value_map = {(qubit.row, qubit.col): np.random.random()
                 for qubit in cirq.google.known_devices.Bristlecone.qubits}

    heatmap = cirq.Heatmap(value_map)
    fig, ax = plt.subplots(figsize=(9, 9))
    heatmap.plot(ax)
    fig.show(warn=False)


if __name__ == '__main__':
    main()
