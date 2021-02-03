"""The script that produces bristlecone_heatmap_example.png."""

import numpy as np
import cirq


def main():
    value_map = {
        (qubit.row, qubit.col): np.random.random() for qubit in cirq.google.Bristlecone.qubits
    }

    heatmap = cirq.Heatmap(value_map)
    heatmap.plot()


if __name__ == '__main__':
    main()
