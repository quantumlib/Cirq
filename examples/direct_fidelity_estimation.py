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


def build_circuit() -> Tuple[cirq.Circuit, List[cirq.Qid]]:
    # Builds an arbitrary circuit to test. Do not include a measurement gate.
    # The circuit need not be Clifford, but if it is, simulations will be
    # faster.
    qubits: List[cirq.Qid] = cast(List[cirq.Qid], cirq.LineQubit.range(3))
    circuit: cirq.Circuit = cirq.Circuit(cirq.CNOT(qubits[0], qubits[2]),
                                         cirq.Z(qubits[0]), cirq.H(qubits[2]),
                                         cirq.CNOT(qubits[2], qubits[1]),
                                         cirq.X(qubits[0]), cirq.X(qubits[1]),
                                         cirq.CNOT(qubits[0], qubits[2]))
    print('Circuit used:')
    print(circuit)
    return circuit, qubits


def compute_characteristic_function(circuit: cirq.Circuit,
                                    pauli_string: cirq.PauliString,
                                    qubits: List[cirq.Qid],
                                    density_matrix: np.ndarray):
    n_qubits = len(qubits)
    d = 2**n_qubits

    qubit_map = dict(zip(qubits, range(n_qubits)))
    # rho_i or sigma_i in https://arxiv.org/abs/1104.3835
    trace = pauli_string.expectation_from_density_matrix(
        density_matrix, qubit_map)
    assert np.isclose(trace.imag, 0.0, atol=1e-6)
    trace = trace.real

    prob = trace * trace / d  # Pr(i) in https://arxiv.org/abs/1104.3835

    return trace, prob


async def estimate_characteristic_function(circuit: cirq.Circuit,
                                           pauli_string: cirq.PauliString,
                                           qubits: List[cirq.Qid],
                                           sampler: cirq.Sampler,
                                           samples_per_term: int):
    """
    Estimates the characteristic function using a (noisy) circuit simulator by
    sampling the results.

    Args:
        circuit: The circuit to run the simulation on.
        pauli_string: The Pauli string.
        qubits: The list of qubits.
        sampler: Either a noisy simulator or an engine.
        samples_per_term: An integer greater than 0, the number of samples.

    Returns:
        The estimated characteristic function.
    """
    p = cirq.PauliSumCollector(circuit=circuit,
                               observable=pauli_string,
                               samples_per_term=samples_per_term)

    await p.collect_async(sampler=sampler)

    sigma_i = p.estimated_energy()
    assert np.isclose(sigma_i.imag, 0.0, atol=1e-6)
    sigma_i = sigma_i.real

    return sigma_i


def _estimate_pauli_traces_clifford(n_qubits: int,
                                    clifford_state: cirq.CliffordState,
                                    n_clifford_trials: int):
    """
    Estimates the Pauli traces in case the circuit is Clifford. When we have a
    Clifford circuit, there are 2**n Pauli traces that have probability 1/2**n
    and all the other traces have probability 0. In addition, there is a fast
    way to compute find out what the traces are. See the documentation of
    cirq.CliffordState for more detail. This function uses the speedup to sample
    the Pauli states with non-zero probability.

    Args:
        n_qubits: An integer that is the number of qubits.
        clifford_state: The basis of the Pauli states with non-zero probability.
        n_clifford_trials: An integer that is the number of Pauli states to
            sample.

    Returns:
        A list of Pauli states (represented as tuples of Pauli string, rho_i,
            and probability.
    """

    # When the circuit consists of Clifford gates only, we can sample the
    # Pauli states more efficiently as described on page 4 of:
    # https://arxiv.org/abs/1104.4695

    d = 2**n_qubits

    # The stabilizers_basis variable only contains basis vectors. For
    # example, if we have n=3 qubits, then we should have 2**n=8 Pauli
    # states that we can sample, but the basis will still have 3 entries. We
    # must flip a coin for each, whether or not to include them.
    stabilizer_basis = clifford_state.stabilizers()

    pauli_traces = []
    for _ in range(n_clifford_trials):
        # Build the Pauli string as a random sample of the basis elements.
        dense_pauli_string = cirq.DensePauliString.eye(n_qubits)
        for stabilizer in stabilizer_basis:
            if np.random.randint(2) == 1:
                dense_pauli_string *= stabilizer

        # The code below is equivalent to calling
        # clifford_state.wave_function() and then calling
        # compute_characteristic_function() on the results (albeit with a
        # wave function instead of a density matrix). It is, however,
        # unncessary to do so. Instead we directly obtain the scalar rho_i.
        rho_i = dense_pauli_string.coefficient

        assert np.isclose(rho_i.imag, 0.0, atol=1e-6)
        rho_i = rho_i.real

        dense_pauli_string *= rho_i

        assert np.isclose(abs(rho_i), 1.0, atol=1e-6)
        Pr_i = 1.0 / d

        pauli_traces.append({
            'P_i': dense_pauli_string.sparse(),
            'rho_i': rho_i,
            'Pr_i': Pr_i
        })
    return pauli_traces


def _estimate_pauli_traces_general(qubits: List[cirq.Qid],
                                   circuit: cirq.Circuit):
    """
    Estimates the Pauli traces in case the circuit is not Clifford. In this case
    we cannot use the speedup implemented in the function
    _estimate_pauli_traces_clifford() above, and so do a slow, density matrix
    simulation.

    Args:
        qubits: The list of qubits.
        circuit: The (non Clifford) circuit.

    Returns:
        A list of Pauli states (represented as tuples of Pauli string, rho_i,
            and probability.
    """

    n_qubits = len(qubits)

    dense_simulator = cirq.DensityMatrixSimulator()
    # rho in https://arxiv.org/abs/1104.3835
    clean_density_matrix = cast(
        cirq.DensityMatrixTrialResult,
        dense_simulator.simulate(circuit)).final_density_matrix

    pauli_traces = []
    for P_i in itertools.product([cirq.I, cirq.X, cirq.Y, cirq.Z],
                                 repeat=n_qubits):
        pauli_string = cirq.PauliString(dict(zip(qubits, P_i)))
        rho_i, Pr_i = compute_characteristic_function(circuit, pauli_string,
                                                      qubits,
                                                      clean_density_matrix)
        pauli_traces.append({'P_i': pauli_string, 'rho_i': rho_i, 'Pr_i': Pr_i})
    return pauli_traces


def direct_fidelity_estimation(circuit: cirq.Circuit, qubits: List[cirq.Qid],
                               sampler: cirq.Sampler, n_trials: int,
                               n_clifford_trials: int, samples_per_term: int):
    """
    Implementation of direct fidelity estimation, as per 'Direct Fidelity
    Estimation from Few Pauli Measurements' https://arxiv.org/abs/1104.4695 and
    'Practical characterization of quantum devices without tomography'
    https://arxiv.org/abs/1104.3835.

    Args:
        circuit: The circuit to run the simulation on.
        qubits: The list of qubits.
        sampler: Either a noisy simulator or an engine.
        n_trial: The total number of Pauli measurements.
        n_clifford_trials: In case the circuit is Clifford, we specify the
            number of trials to estimate the noise-free pauli traces.
        samples_per_term: if set to 0, we use the 'sampler' parameter above as
            a noise (must be of type cirq.DensityMatrixSimulator) and
            simulate noise in the circuit. If greater than 0, we instead use the
            'sampler' parameter directly to estimate the characteristic
            function.
    Returns:
        The estimated fidelity.
    """
    # n_trials is upper-case N in https://arxiv.org/abs/1104.3835

    # Number of qubits, lower-case n in https://arxiv.org/abs/1104.3835
    n_qubits = len(qubits)
    d = 2**n_qubits

    clifford_circuit = True
    clifford_state: Optional[cirq.CliffordState] = None
    try:
        clifford_state = cirq.CliffordState(
            qubit_map={qubits[i]: i for i in range(len(qubits))})
        for gate in circuit.all_operations():
            clifford_state.apply_unitary(gate)
    except ValueError:
        clifford_circuit = False

    # Computes for every \hat{P_i} of https://arxiv.org/abs/1104.3835
    # estimate rho_i and Pr(i). We then collect tuples (rho_i, Pr(i), \hat{Pi})
    # inside the variable 'pauli_traces'.
    if clifford_circuit:
        print('Circuit is Clifford')
        assert clifford_state is not None
        pauli_traces = _estimate_pauli_traces_clifford(
            n_qubits, cast(cirq.CliffordState, clifford_state),
            n_clifford_trials)
    else:
        print('Circuit is not Clifford')
        pauli_traces = _estimate_pauli_traces_general(qubits, circuit)

    p = np.asarray([x['Pr_i'] for x in pauli_traces])

    if not clifford_circuit:
        # For Clifford circuits, we do a Monte Carlo simulations, and thus there
        # is no guarantee that it adds up to 1.0 (but it should to the limit).
        assert np.isclose(np.sum(p), 1.0, atol=1e-6)

    # The package np.random.choice() is quite sensitive to probabilities not
    # summing up to 1.0. Even an absolute difference below 1e-6 (as checked just
    # above) does bother it, so we re-normalize the probs.
    p /= np.sum(p)

    fidelity = 0.0

    if samples_per_term == 0:
        # sigma in https://arxiv.org/abs/1104.3835
        if not isinstance(sampler, cirq.DensityMatrixSimulator):
            raise TypeError('sampler is not a cirq.DensityMatrixSimulator '
                            'but samples_per_term is zero.')
        noisy_simulator = cast(cirq.DensityMatrixSimulator, sampler)
        noisy_density_matrix = cast(
            cirq.DensityMatrixTrialResult,
            noisy_simulator.simulate(circuit)).final_density_matrix

    for _ in range(n_trials):
        # Randomly sample as per probability.
        i = np.random.choice(len(pauli_traces), p=p)

        Pr_i = pauli_traces[i]['Pr_i']
        measure_pauli_string: cirq.PauliString = pauli_traces[i]['P_i']
        rho_i = pauli_traces[i]['rho_i']

        if samples_per_term > 0:
            sigma_i = asyncio.get_event_loop().run_until_complete(
                estimate_characteristic_function(circuit, measure_pauli_string,
                                                 qubits, sampler,
                                                 samples_per_term))
        else:
            sigma_i, _ = compute_characteristic_function(
                circuit, measure_pauli_string, qubits, noisy_density_matrix)

        fidelity += Pr_i * sigma_i / rho_i

    return fidelity / n_trials * d


def parse_arguments(args):
    """Helper function that parses the given arguments."""
    parser = argparse.ArgumentParser('Direct fidelity estimation.')

    parser.add_argument('--n_trials',
                        default=10,
                        type=int,
                        help='Number of trials to run.')

    # TODO(#2802): Offer some guidance on how to set this flag. Maybe have an
    # option to do an exhaustive sample and do numerical studies to know which
    # choice is the best.
    parser.add_argument('--n_clifford_trials',
                        default=3,
                        type=int,
                        help='Number of trials for Clifford circuits. This is '
                        'in effect when the circuit is Clifford. In this '
                        'case, we randomly sample the Pauli traces with '
                        'non-zero probabilities. The higher the number, '
                        'the more accurate the overall fidelity '
                        'estimation, at the cost of extra computing and '
                        'measurements.')

    parser.add_argument('--samples_per_term',
                        default=0,
                        type=int,
                        help='Number of samples per trial or 0 if no sampling.')

    return vars(parser.parse_args(args))


def main(*, n_trials: int, n_clifford_trials: int, samples_per_term: int):
    circuit, qubits = build_circuit()

    noise = cirq.ConstantQubitNoiseModel(cirq.depolarize(0.1))
    print('Noise model: %s' % (noise))
    noisy_simulator = cirq.DensityMatrixSimulator(noise=noise)

    estimated_fidelity = direct_fidelity_estimation(
        circuit,
        qubits,
        noisy_simulator,
        n_trials=n_trials,
        n_clifford_trials=n_clifford_trials,
        samples_per_term=samples_per_term)
    print('Estimated fidelity: %f' % (estimated_fidelity))


if __name__ == '__main__':
    main(**parse_arguments(sys.argv[1:]))
