"""Estimate fidelity of large random quantum circuit from observed bitstrings.

Fidelity estimator defined here is used in cross-entropy benchmarking and
works under the assumption that the evaluated circuit is sufficiently
random, see https://arxiv.org/abs/1608.00263.
"""

from typing import cast, List, Sequence, Tuple

import numpy as np

from cirq.circuits import Circuit
from cirq.ops import Qid
from cirq.sim import Simulator, WaveFunctionTrialResult


def compute_linear_xeb_fidelity(circuit: Circuit, qubit_order: Sequence[Qid],
                                bitstrings: Sequence[int]) -> float:
    """Computes fidelity estimate from one circuit using linear XEB estimator.

    Args:
        circuit: Random quantum circuit which has been executed on quantum
            processor under test
        qubit_order: Qubit order used to construct bitstrings from measurements
        bitstrings: Results of terminal all-qubit measurements performed after
            each circuit execution
    Returns:
        Estimate of circuit fidelity.
    Raises:
        ValueError: Circuit is inconsistent with qubit order or one of the
            bitstrings is inconsistent with the number of qubits.
    """
    dim = 2**len(qubit_order)

    if set(qubit_order) != circuit.all_qubits():
        raise ValueError(
            f'Inconsistent qubits: circuit has {circuit.all_qubits()}, '
            f'qubit order is {qubit_order}')
    for bitstring in bitstrings:
        if bitstring < 0 or dim <= bitstring:
            raise ValueError(
                f'Bitstring {bitstring} could not have been observed '
                f'on {len(qubit_order)} qubits.')

    simulator = Simulator()
    result = cast(WaveFunctionTrialResult,
                  simulator.simulate(circuit, qubit_order=qubit_order))
    output_probabilities = np.abs(result.final_state)**2
    assert 1 - 1e-4 < np.sum(output_probabilities) < 1 + 1e-4
    fidelity_estimate = dim * np.mean(output_probabilities[bitstrings]) - 1
    return fidelity_estimate
