"""Implements direct fidelity estimation.

Fidelity between the desired pure state rho and the actual state sigma is
defined as:
F(rho, sigma) = Tr (rho sigma)

It is a unit-less measurement between 0.0 and 1.0. The following two papers
independently described a faster way to estimate its value:

Direct Fidelity Estimation from Few Pauli Measurements
https://arxiv.org/abs/1104.4695

Practical characterization of quantum devices without tomography
https://arxiv.org/abs/1104.3835

This code implements the algorithm proposed for an example circuit (defined in
the function build_circuit()) and a noise (defines in the variable noise).
"""

import argparse
import asyncio
import itertools
from typing import cast
from typing import List
from typing import Optional
from typing import Tuple
import sys
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
    print('Circuit used:')
    print(circuit)
    return circuit, qubits


def compute_characteristic_function(circuit: cirq.Circuit,
                                    P_i: Tuple[cirq.Gate, ...],
                                    qubits: List[cirq.Qid],
                                    density_matrix: np.ndarray):
    n_qubits = len(P_i)
    d = 2**n_qubits

    pauli_string = cirq.PauliString(dict(zip(qubits, P_i)))
    qubit_map = dict(zip(qubits, range(n_qubits)))
    # rho_i or sigma_i in https://arxiv.org/abs/1104.3835
    trace = pauli_string.expectation_from_density_matrix(
        density_matrix, qubit_map)
    assert np.isclose(trace.imag, 0.0, atol=1e-6)
    trace = trace.real

    prob = trace * trace / d  # Pr(i) in https://arxiv.org/abs/1104.3835

    return trace, prob


async def estimate_characteristic_function(
        circuit: cirq.Circuit, P_i: Tuple[cirq.Gate, ...],
        qubits: List[cirq.Qid], simulator: cirq.DensityMatrixSimulator,
        samples_per_term: int):
    """
    Estimates the characteristic function using a (noisy) circuit simulator by
    sampling the results.

    Args:
        circuit: The circuit to run the simulation on.
        P_i: The Pauli matrix.
        qubits: The list of qubits.
        simulator: The (noisy) simulator.
        samples_per_term: An integer greater than 0, the number of samples.

    Returns:
        The estimated characteristic function.
    """
    pauli_string = cirq.PauliString(dict(zip(qubits, P_i)))

    p = cirq.PauliSumCollector(circuit=circuit,
                               observable=pauli_string,
                               samples_per_term=samples_per_term)

    await p.collect_async(sampler=simulator)

    sigma_i = p.estimated_energy()
    assert np.isclose(sigma_i.imag, 0.0, atol=1e-6)
    sigma_i = sigma_i.real

    return sigma_i


def direct_fidelity_estimation(circuit: cirq.Circuit, qubits: List[cirq.Qid],
                               noise: cirq.NoiseModel, n_trials: int,
                               samples_per_term: int):
    """
    Implementation of direct fidelity estimation, as per 'Direct Fidelity
    Estimation from Few Pauli Measurements' https://arxiv.org/abs/1104.4695 and
    'Practical characterization of quantum devices without tomography'
    https://arxiv.org/abs/1104.3835.

    Args:
        circuit: The circuit to run the simulation on.
        qubits: The list of qubits.
        noise: The noise model when doing a simulation.
        n_trial: The total number of Pauli measurements.
        samples_per_term: is set to 0, we use the 'noise' parameter above and
            simulate noise in the circuit. If greater than 0, we ignore the
            'noise' parameter above and instead run an estimation of the
            characteristic function.

    Returns:
        The estimated fidelity.
    """
    # n_trials is upper-case N in https://arxiv.org/abs/1104.3835

    # Number of qubits, lower-case n in https://arxiv.org/abs/1104.3835
    n_qubits = len(qubits)

    # Computes for every \hat{P_i} of https://arxiv.org/abs/1104.3835
    # estimate rho_i and Pr(i). We then collect tuples (rho_i, Pr(i), \hat{Pi})
    # inside the variable 'pauli_traces'.
    pauli_traces = []

    simulator = cirq.DensityMatrixSimulator()
    # rho in https://arxiv.org/abs/1104.3835
    clean_density_matrix = cast(
        cirq.DensityMatrixTrialResult,
        simulator.simulate(circuit)).final_density_matrix

    # TODO(#2639): Sample the Pauli states more efficiently when the circuit
    # consists of Clifford gates only, as described on page 4 of:
    # https://arxiv.org/abs/1104.4695
    for P_i in itertools.product([cirq.I, cirq.X, cirq.Y, cirq.Z],
                                 repeat=n_qubits):
        rho_i, Pr_i = compute_characteristic_function(circuit, P_i, qubits,
                                                      clean_density_matrix)
        pauli_traces.append({'P_i': P_i, 'rho_i': rho_i, 'Pr_i': Pr_i})

    assert len(pauli_traces) == 4**n_qubits

    p = np.asarray([x['Pr_i'] for x in pauli_traces])
    assert np.isclose(np.sum(p), 1.0, atol=1e-6)

    # The package np.random.choice() is quite sensitive to probabilities not
    # summing up to 1.0. Even an absolute difference below 1e-6 (as checked just
    # above) does bother it, so we re-normalize the probs.
    p /= np.sum(p)

    simulator = cirq.DensityMatrixSimulator(noise=noise)
    fidelity = 0.0

    if samples_per_term == 0:
        # sigma in https://arxiv.org/abs/1104.3835
        noisy_density_matrix = cast(
            cirq.DensityMatrixTrialResult,
            simulator.simulate(circuit)).final_density_matrix

    for _ in range(n_trials):
        # Randomly sample as per probability.
        i = np.random.choice(len(pauli_traces), p=p)

        Pr_i = pauli_traces[i]['Pr_i']
        P_i = pauli_traces[i]['P_i']
        rho_i = pauli_traces[i]['rho_i']

        if samples_per_term > 0:
            sigma_i = asyncio.get_event_loop().run_until_complete(
                estimate_characteristic_function(circuit, P_i, qubits,
                                                 simulator, samples_per_term))
        else:
            sigma_i, _ = compute_characteristic_function(
                circuit, P_i, qubits, noisy_density_matrix)

        fidelity += Pr_i * sigma_i / rho_i

    return fidelity / n_trials


def parse_arguments(args):
    """Helper function that parses the given arguments."""
    parser = argparse.ArgumentParser('Direct fidelity estimation.')

    parser.add_argument('--n_trials',
                        default=10,
                        type=int,
                        help='Number of trials to run.')

    parser.add_argument('--samples_per_term',
                        default=0,
                        type=int,
                        help='Number of samples per trial or 0 if no sampling.')

    return vars(parser.parse_args(args))


def main(*, n_trials: int, samples_per_term: int):
    circuit, qubits = build_circuit()
    circuit.append(cirq.measure(*qubits, key='y'))

    noise = cirq.ConstantQubitNoiseModel(cirq.depolarize(0.1))
    print('Noise model: %s' % (noise))

    estimated_fidelity = direct_fidelity_estimation(
        circuit,
        qubits,
        noise,
        n_trials=n_trials,
        samples_per_term=samples_per_term)
    print('Estimated fidelity: %f' % (estimated_fidelity))


if __name__ == '__main__':
    main(**parse_arguments(sys.argv[1:]))
