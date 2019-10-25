"""Implements direct fidelity estimation.

Direct Fidelity Estimation from Few Pauli Measurements
https://arxiv.org/abs/1104.4695

Practical characterization of quantum devices without tomography
https://arxiv.org/abs/1104.3835
"""

import itertools
from typing import cast
from typing import List
from typing import Optional
from typing import Tuple
import numpy as np
import cirq


def build_circuit():
    # Builds an arbitrary circuit to test. The circuit is non Clifford to show
    # the use of simulators.
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.Z(qubits[0])**0.25,  # T-Gate, non Clifford.
        cirq.X(qubits[1])**0.123,
        cirq.X(qubits[2])**0.456)
    return circuit, qubits


def compute_characteristic_function(circuit: cirq.Circuit,
                                    P_i: Tuple[cirq.Gate, ...],
                                    qubits: List[cirq.Qid],
                                    noise: Optional[cirq.NoiseModel]):
    n_qubits = len(P_i)
    d = 2**n_qubits

    simulator = cirq.DensityMatrixSimulator()
    # rho or sigma in https://arxiv.org/pdf/1104.3835.pdf
    density_matrix = cast(cirq.DensityMatrixTrialResult,
                          simulator.simulate(circuit)).final_density_matrix

    pauli_string = cirq.PauliString(dict(zip(qubits, P_i)))
    qubit_map = dict(zip(qubits, range(n_qubits)))
    # rho_i or sigma_i in https://arxiv.org/pdf/1104.3835.pdf
    trace = pauli_string.expectation_from_density_matrix(
        density_matrix, qubit_map)
    assert np.isclose(trace.imag, 0.0, atol=1e-6)
    trace = trace.real

    prob = trace * trace / d  # Pr(i) in https://arxiv.org/pdf/1104.3835.pdf

    return trace, prob


def direct_fidelity_estimation(circuit: cirq.Circuit, qubits: List[cirq.Qid],
                               noise: cirq.NoiseModel, n_trials: int):
    # n_trials is upper-case N in https://arxiv.org/pdf/1104.3835.pdf

    # Number of qubits, lower-case n in https://arxiv.org/pdf/1104.3835.pdf
    n_qubits = len(qubits)

    # Computes for every \hat{P_i} of https://arxiv.org/pdf/1104.3835.pdf,
    # estimate rho_i and Pr(i). We then collect tuples (rho_i, Pr(i), \hat{Pi})
    # inside the variable 'pauli_traces'.
    pauli_traces = []
    for P_i in itertools.product([cirq.I, cirq.X, cirq.Y, cirq.Z],
                                 repeat=n_qubits):
        rho_i, Pr_i = compute_characteristic_function(circuit,
                                                      P_i,
                                                      qubits,
                                                      noise=None)
        pauli_traces.append({'P_i': P_i, 'rho_i': rho_i, 'Pr_i': Pr_i})

    assert len(pauli_traces) == 4**n_qubits

    p = [x['Pr_i'] for x in pauli_traces]
    assert np.isclose(sum(p), 1.0, atol=1e-6)

    # The package np.random.choice() is quite sensitive to probabilities not
    # summing up to 1.0. Even an absolute difference below 1e-6 (as checked just
    # above) does bother it, so we re-normalize the probs.
    inv_sum_p = 1 / sum(p)
    norm_p = [x * inv_sum_p for x in p]

    fidelity = 0.0
    for _ in range(n_trials):
        # Randomly sample as per probability.
        i = np.random.choice(range(4**n_qubits), p=norm_p)

        Pr_i = pauli_traces[i]['Pr_i']
        P_i = pauli_traces[i]['P_i']
        rho_i = pauli_traces[i]['rho_i']

        sigma_i, _ = compute_characteristic_function(circuit, P_i, qubits,
                                                     noise)

        fidelity += Pr_i * sigma_i / rho_i

    return fidelity / n_trials


def main():
    circuit, qubits = build_circuit()
    circuit.append(cirq.measure(*qubits, key='y'))

    noise = cirq.ConstantQubitNoiseModel(cirq.depolarize(0.1))

    estimated_fidelity = direct_fidelity_estimation(circuit,
                                                    qubits,
                                                    noise,
                                                    n_trials=10)
    print(estimated_fidelity)


if __name__ == '__main__':
    main()
