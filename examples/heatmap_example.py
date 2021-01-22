"""The script that produces bristlecone_heatmap_example.png."""
from typing import Tuple, cast

import numpy as np
import cirq


def bristlecone():
    value_map = {
        (qubit.row, qubit.col): np.random.random() for qubit in cirq.google.Bristlecone.qubits
    }

    heatmap = cirq.Heatmap(value_map)
    file_path = "examples/bristlecone_heatmap_example.png"
    fig, _, _ = heatmap.plot()
    fig.figure.savefig(file_path)


def sycamore_edges():
    # TODO(https://github.com/quantumlib/Cirq/issues/3696): replace this when we have a proper
    # method for it
    return {
        pair
        for gate_defs in cirq.google.Sycamore.gate_definitions.values()
        for gate_def in gate_defs
        if gate_def.number_of_qubits == 2
        for pair in gate_def.target_set
        if len(pair) == 2
    }


def two_qubit_interaction_heatmap():
    title = 'Two Qubit Sycamore Gate Xeb Cycle Total Error'
    value_map = {qubit_pair: np.random.random() for qubit_pair in sycamore_edges()}
    heatmap = cirq.InterHeatmap(value_map, title)
    file_path = "examples/qubitinteraction_heatmap_example.png"
    fig, _, _ = heatmap.plot()
    fig.figure.savefig(file_path)


if __name__ == '__main__':
    # coverage: ignore
    bristlecone()
    two_qubit_interaction_heatmap()
