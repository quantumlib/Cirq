# pylint: disable=wrong-or-nonexistent-copyright-notice
"""Example heatmaps from the cirq.vis.heatmap package."""

import numpy as np

import cirq
import cirq_google


def single_qubit_heatmap():
    """Demo of cirq.Heatmap.
    Demonstrates how cirq.Heatmap can be used to generate a heatmap of the qubit fidelities.
    """
    value_map = {(qubit,): np.random.random() for qubit in cirq_google.Bristlecone.qubits}

    heatmap = cirq.Heatmap(value_map)
    # This is going to produce an image similar to examples/single_qubit_heatmap_example.png
    heatmap.plot()


def _sycamore_qubit_pairs():
    # TODO(https://github.com/quantumlib/Cirq/issues/3696): replace this when we have a proper
    # method for it
    return set(
        tuple(sorted(pair))
        for gate_defs in cirq_google.Sycamore.gate_definitions.values()
        for gate_def in gate_defs
        if gate_def.number_of_qubits == 2
        for pair in gate_def.target_set
        if len(pair) == 2
    )


def two_qubit_interaction_heatmap():
    """Demo of cirq.InteractionHeatmap.
    Demonstrates how cirq.Heatmap can be used to generate a two qubit interaction heatmap.
    """

    # normally one would get these from cirq_google.engine
    s = np.random.RandomState(1234)
    random_characterization_data = {qubit_pair: s.rand() for qubit_pair in _sycamore_qubit_pairs()}

    heatmap = cirq.TwoQubitInteractionHeatmap(
        value_map=random_characterization_data,
        title='Two Qubit Sycamore Gate XEB Cycle Total Error',
    )
    # this is going to plot something similar to examples/two_qubit_interaction_heatmap_example.png
    heatmap.plot()


def main():
    # coverage: ignore
    single_qubit_heatmap()
    two_qubit_interaction_heatmap()


if __name__ == '__main__':
    main()
