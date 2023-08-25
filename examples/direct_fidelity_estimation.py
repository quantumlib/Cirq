# pylint: disable=wrong-or-nonexistent-copyright-notice
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
import itertools
import math
import random
import sys
from dataclasses import dataclass
from typing import cast, List, Optional, Tuple

import cirq
import duet
import numpy as np
from cirq.sim import clifford


def build_circuit() -> Tuple[cirq.Circuit, List[cirq.Qid]]:
    # Builds an arbitrary circuit to test. Do not include a measurement gate.
    # The circuit need not be Clifford, but if it is, simulations will be
    # faster.
    qubits: List[cirq.Qid] = cast(List[cirq.Qid], cirq.LineQubit.range(3))
    circuit: cirq.Circuit = cirq.Circuit(
        cirq.CNOT(qubits[0], qubits[2]),
        cirq.Z(qubits[0]),
        cirq.H(qubits[2]),
        cirq.CNOT(qubits[2], qubits[1]),
        cirq.X(qubits[0]),
        cirq.X(qubits[1]),
        cirq.CNOT(qubits[0], qubits[2]),
    )
    print('Circuit used:')
    print(circuit)
    return circuit, qubits


def compute_characteristic_function(
    pauli_string: cirq.PauliString, qubits: List[cirq.Qid], density_matrix: np.ndarray
):
    n_qubits = len(qubits)
    d = 2**n_qubits

    qubit_map = dict(zip(qubits, range(n_qubits)))
    # rho_i or sigma_i in https://arxiv.org/abs/1104.3835
    trace = pauli_string.expectation_from_density_matrix(density_matrix, qubit_map)
    assert np.isclose(trace.imag, 0.0, atol=1e-6)
    trace = trace.real

    prob = trace * trace / d  # Pr(i) in https://arxiv.org/abs/1104.3835

    return trace, prob


async def estimate_characteristic_function(
    circuit: cirq.Circuit,
    pauli_string: cirq.PauliString,
    sampler: cirq.Sampler,
    samples_per_term: int,
) -> float:
    """Estimates the characteristic function using a (noisy) circuit simulator.

    Args:
        circuit: The circuit to run the simulation on.
        pauli_string: The Pauli string.
        sampler: Either a noisy simulator or an engine.
        samples_per_term: An integer greater than 0, the number of samples.

    Returns:
        The estimated characteristic function.
    """
    p = cirq.PauliSumCollector(
        circuit=circuit, observable=pauli_string, samples_per_term=samples_per_term
    )

    await p.collect_async(sampler=sampler)

    sigma_i = p.estimated_energy()
    assert np.isclose(sigma_i.imag, 0.0, atol=1e-6)
    sigma_i = sigma_i.real

    return sigma_i


def _randomly_sample_from_stabilizer_bases(
    stabilizer_basis: List[cirq.DensePauliString], n_measured_operators: int, n_qubits: int
):
    """Randomly creates Pauli states by including or not the given stabilizer basis.

    Args:
        stabilizer_basis: A list of Pauli strings that is the stabilizer basis
            to sample from.
        n_measured_operators: The total number of Pauli measurements, or None to
            explore each Pauli state once.
        n_qubits: An integer that is the number of qubits.

    Returns:
        A list of Pauli strings that is the Pauli states built.
    """
    dense_pauli_strings = []
    for _ in range(n_measured_operators):
        # Build the Pauli string as a random sample of the basis elements.
        dense_pauli_string = cirq.DensePauliString.eye(n_qubits)
        for stabilizer in stabilizer_basis:
            if np.random.randint(2) == 1:
                dense_pauli_string *= stabilizer
        dense_pauli_strings.append(dense_pauli_string)
    return dense_pauli_strings


def _enumerate_all_from_stabilizer_bases(
    stabilizer_basis: List[cirq.DensePauliString], n_qubits: int
):
    """Return the list of Pauli state that are spanned by the given Pauli basis.

    Args:
        stabilizer_basis: A list of Pauli strings that is the stabilizer basis
            to build all the Pauli strings.
        n_qubits: An integer that is the number of qubits.

    Returns:
        A list of Pauli strings that is the Pauli states built.
    """
    dense_pauli_strings = []
    for coefficients in itertools.product([False, True], repeat=n_qubits):
        dense_pauli_string = cirq.DensePauliString.eye(n_qubits)
        for (keep, stabilizer) in zip(coefficients, stabilizer_basis):
            if keep:
                dense_pauli_string *= stabilizer
        dense_pauli_strings.append(dense_pauli_string)
    return dense_pauli_strings


@dataclass
class PauliTrace:
    """Holder of a description fo Pauli states.

    This class contains the Pauli states as described on page 2 of: https://arxiv.org/abs/1104.3835
    """

    # Pauli string.
    P_i: cirq.PauliString
    # Coefficient of the ideal pure state expanded in the Pauli basis scaled by
    # sqrt(dim H), formally defined at bottom of left column of page 2.
    rho_i: float
    # A probability (between 0.0 and 1.0) that is the relevance distribution,
    # formally defined at top of right column of page 2.
    Pr_i: float


def _estimate_pauli_traces_clifford(
    n_qubits: int,
    stabilizer_basis: List[cirq.DensePauliString],
    n_measured_operators: Optional[int],
) -> List[PauliTrace]:
    """Estimates the Pauli traces in case the circuit is Clifford.

    When we have a Clifford circuit, there are 2**n Pauli traces that have probability 1/2**n
    and all the other traces have probability 0. In addition, there is a fast
    way to compute find out what the traces are. See the documentation of
    cirq.CliffordTableau for more detail. This function uses the speedup to sample
    the Pauli states with non-zero probability.

    Args:
        n_qubits: An integer that is the number of qubits.
        stabilizer_basis: The basis of the Pauli states with non-zero probability.
        n_measured_operators: The total number of Pauli measurements, or None to
            explore each Pauli state once.

    Returns:
        A list of Pauli states (represented as tuples of Pauli string, rho_i,
            and probability.
    """

    # When the circuit consists of Clifford gates only, we can sample the
    # Pauli states more efficiently as described on page 4 of:
    # https://arxiv.org/abs/1104.4695

    d = 2**n_qubits

    if n_measured_operators is not None:
        dense_pauli_strings = _randomly_sample_from_stabilizer_bases(
            stabilizer_basis, n_measured_operators, n_qubits
        )
        assert len(dense_pauli_strings) == n_measured_operators
    else:
        dense_pauli_strings = _enumerate_all_from_stabilizer_bases(stabilizer_basis, n_qubits)
        assert len(dense_pauli_strings) == 2**n_qubits

    pauli_traces: List[PauliTrace] = []
    for dense_pauli_string in dense_pauli_strings:
        # The code below is equivalent to getting the state vectors from the
        # Clifford tableau and then calling compute_characteristic_function()
        # on the results (albeit with a  wave function instead of a density
        # matrix). It is, however, unnecessary to do so. Instead we directly
        # obtain the scalar rho_i.
        rho_i = dense_pauli_string.coefficient

        assert np.isclose(rho_i.imag, 0.0, atol=1e-6)
        rho_i = rho_i.real

        dense_pauli_string *= rho_i

        assert np.isclose(abs(rho_i), 1.0, atol=1e-6)
        Pr_i = 1.0 / d

        pauli_traces.append(PauliTrace(P_i=dense_pauli_string.sparse(), rho_i=rho_i, Pr_i=Pr_i))
    return pauli_traces


def _estimate_pauli_traces_general(
    qubits: List[cirq.Qid], circuit: cirq.Circuit, n_measured_operators: Optional[int]
) -> List[PauliTrace]:
    """Estimates the Pauli traces in case the circuit is not Clifford.

    In this case we cannot use the speedup implemented in the function
    _estimate_pauli_traces_clifford() above, and so do a slow, density matrix
    simulation.

    Args:
        qubits: The list of qubits.
        circuit: The (non Clifford) circuit.
        n_measured_operators: The total number of Pauli measurements, or None to
            explore each Pauli state once.

    Returns:
        A list of Pauli states (represented as tuples of Pauli string, rho_i,
            and probability.
    """

    n_qubits = len(qubits)

    dense_simulator = cirq.DensityMatrixSimulator()
    # rho in https://arxiv.org/abs/1104.3835
    clean_density_matrix = dense_simulator.simulate(circuit).final_density_matrix

    all_operators = itertools.product([cirq.I, cirq.X, cirq.Y, cirq.Z], repeat=n_qubits)
    if n_measured_operators is not None:
        dense_operators = random.sample(tuple(all_operators), n_measured_operators)
    else:
        dense_operators = list(all_operators)

    pauli_traces: List[PauliTrace] = []
    for P_i in dense_operators:
        pauli_string: cirq.PauliString[cirq.Qid] = cirq.PauliString(dict(zip(qubits, P_i)))
        rho_i, Pr_i = compute_characteristic_function(pauli_string, qubits, clean_density_matrix)
        pauli_traces.append(PauliTrace(P_i=pauli_string, rho_i=rho_i, Pr_i=Pr_i))
    return pauli_traces


def _estimate_std_devs_clifford(fidelity: float, n: int) -> Tuple[Optional[float], float]:
    """Estimates the standard deviation of the measurement for Clifford circuits.

    Args:
        fidelity: The measured fidelity
        n: the number of measurements

    Returns:
        The standard deviation (estimated from the fidelity and a maximum bound
        on the variance regardless of what the true fidelity is)
    """

    # We use the Bhatia Davis inequality to estimate the variance of the
    # fidelity. This gives that:
    # Var[\hat{F}] <= (1 - F) . F / N
    # StdDev[\hat{F}] <= \sqrt{(1 - F) . F / N}
    #
    # By further using the fact that 0 <= F <= 1 we get:
    # StdDev[\hat{F}] <= \frac{1}{2 \sqrt{N}}

    # Because of the noisiness of the simulation, the estimated fidelity can be
    # outside the [0, 1] range. If that is the case, we just do not use it to
    # compute the estimate.
    in_range = fidelity >= 0 and fidelity <= 1.0
    std_dev_estimate = math.sqrt((1.0 - fidelity) * fidelity / n) if in_range else None

    std_dev_bound = 0.5 / math.sqrt(n)
    return std_dev_estimate, std_dev_bound


@dataclass
class Result:
    """Contains the results of a trial, either by simulator or actual run."""

    # The Pauli trace that was measured
    pauli_trace: PauliTrace
    # Coefficient of the measured/simulated pure state expanded in the Pauli
    # basis scaled by sqrt(dim H), formally defined at bottom of left column of
    # second page of https://arxiv.org/abs/1104.3835
    sigma_i: float


@dataclass
class DFEIntermediateResult:
    """A container for debug and run data returned from `direct_fidelity_estimation`.

    This is useful when running a long-computation on an actual computer, which is expensive.
    This allows theses runs to be more easily debugged offline.
    """

    # If the circuit is Clifford, the Clifford tableau from which we can extract
    # a list of Pauli strings for a basis of the stabilizers.
    clifford_tableau: Optional[cirq.CliffordTableau]
    # The list of Pauli traces we can sample from.
    pauli_traces: List[PauliTrace]
    # Measurement results from sampling the circuit.
    trial_results: List[Result]
    # Standard deviations (estimate based on fidelity and bound)
    std_dev_estimate: Optional[float]
    std_dev_bound: Optional[float]


def direct_fidelity_estimation(
    circuit: cirq.Circuit,
    qubits: List[cirq.Qid],
    sampler: cirq.Sampler,
    n_measured_operators: Optional[int],
    samples_per_term: int,
):
    """Perform a direct fidelity estimation.

    This implementation of direct fidelity estimation, is that of 'Direct Fidelity
    Estimation from Few Pauli Measurements' https://arxiv.org/abs/1104.4695 and
    'Practical characterization of quantum devices without tomography'
    https://arxiv.org/abs/1104.3835.

    Args:
        circuit: The circuit to run the simulation on.
        qubits: The list of qubits.
        sampler: Either a noisy simulator or an engine.
        n_measured_operators: The total number of Pauli measurements, or None to
            explore each Pauli state once.
        samples_per_term: if set to 0, we use the 'sampler' parameter above as
            a noise (must be of type cirq.DensityMatrixSimulator) and
            simulate noise in the circuit. If greater than 0, we instead use the
            'sampler' parameter directly to estimate the characteristic
            function.

    Returns:
        The estimated fidelity and a log of the run.

    Raises:
        TypeError: If the circuit is not made up entirely of Clifford gates.
    """
    # n_measured_operators is upper-case N in https://arxiv.org/abs/1104.3835

    # Number of qubits, lower-case n in https://arxiv.org/abs/1104.3835
    n_qubits = len(qubits)

    clifford_circuit = True
    clifford_tableau = cirq.CliffordTableau(n_qubits)
    try:
        for gate in circuit.all_operations():
            tableau_args = clifford.CliffordTableauSimulationState(
                tableau=clifford_tableau, qubits=qubits
            )
            cirq.act_on(gate, tableau_args)
    except TypeError:
        clifford_circuit = False

    # Computes for every \hat{P_i} of https://arxiv.org/abs/1104.3835
    # estimate rho_i and Pr(i). We then collect tuples (rho_i, Pr(i), \hat{Pi})
    # inside the variable 'pauli_traces'.
    if clifford_circuit:
        # The stabilizers_basis variable only contains basis vectors. For
        # example, if we have n=3 qubits, then we should have 2**n=8 Pauli
        # states that we can sample, but the basis will still have 3 entries. We
        # must flip a coin for each, whether or not to include them.
        stabilizer_basis: List[cirq.DensePauliString] = clifford_tableau.stabilizers()

        pauli_traces = _estimate_pauli_traces_clifford(
            n_qubits, stabilizer_basis, n_measured_operators
        )
    else:
        pauli_traces = _estimate_pauli_traces_general(qubits, circuit, n_measured_operators)

    p = np.asarray([x.Pr_i for x in pauli_traces])

    if n_measured_operators is None:
        # Since we enumerate all the possible traces, the probs should add to 1.
        assert np.isclose(np.sum(p), 1.0, atol=1e-6)
    p /= np.sum(p)

    fidelity = 0.0

    if samples_per_term == 0:
        # sigma in https://arxiv.org/abs/1104.3835
        if not isinstance(sampler, cirq.DensityMatrixSimulator):
            raise TypeError(
                'sampler is not a cirq.DensityMatrixSimulator but samples_per_term is zero.'
            )
        noisy_density_matrix = sampler.simulate(circuit).final_density_matrix

    if clifford_circuit and n_measured_operators is None:
        # In case the circuit is Clifford and we compute an exhaustive list of
        # Pauli traces, instead of sampling we can simply enumerate them because
        # they all have the same probability.
        measured_pauli_traces = pauli_traces
    else:
        # Otherwise, randomly sample as per probability.
        measured_pauli_traces = np.random.choice(
            pauli_traces, size=len(pauli_traces), p=p  # type: ignore[arg-type]
        ).tolist()

    trial_results: List[Result] = []
    for pauli_trace in measured_pauli_traces:
        measure_pauli_string: cirq.PauliString = pauli_trace.P_i
        rho_i = pauli_trace.rho_i

        if samples_per_term > 0:
            sigma_i = duet.run(
                estimate_characteristic_function,
                circuit,
                measure_pauli_string,
                sampler,
                samples_per_term,
            )
        else:
            sigma_i, _ = compute_characteristic_function(
                measure_pauli_string, qubits, noisy_density_matrix
            )

        trial_results.append(Result(pauli_trace=pauli_trace, sigma_i=sigma_i))

        fidelity += sigma_i / rho_i

    estimated_fidelity = fidelity / len(pauli_traces)

    std_dev_estimate: Optional[float]
    std_dev_bound: Optional[float]
    if clifford_circuit:
        std_dev_estimate, std_dev_bound = _estimate_std_devs_clifford(
            estimated_fidelity, len(measured_pauli_traces)
        )
    else:
        std_dev_estimate, std_dev_bound = None, None

    dfe_intermediate_result = DFEIntermediateResult(
        clifford_tableau=clifford_tableau if clifford_circuit else None,
        pauli_traces=pauli_traces,
        trial_results=trial_results,
        std_dev_estimate=std_dev_estimate,
        std_dev_bound=std_dev_bound,
    )

    return estimated_fidelity, dfe_intermediate_result


def parse_arguments(args):
    """Helper function that parses the given arguments."""
    parser = argparse.ArgumentParser('Direct fidelity estimation.')

    # TODO: Offer some guidance on how to set this flag. Maybe have an
    # option to do an exhaustive sample and do numerical studies to know which
    # choice is the best.
    # Github issue: https://github.com/quantumlib/Cirq/issues/2802
    parser.add_argument(
        '--n_measured_operators',
        default=10,
        type=int,
        help='Numbers of measured operators (Pauli strings). '
        'If the circuit is Clifford, these operators are '
        'computed by sampling for the basis of stabilizers. If '
        'the circuit is not Clifford, this is a random sample '
        'all the possible operators. If the value of this '
        'parameter is None, we enumerate all the operators '
        'which is 2**n_qubit for Clifford circuits and '
        '4**n_qubits otherwise.',
    )

    parser.add_argument(
        '--samples_per_term',
        default=0,
        type=int,
        help='Number of samples per trial or 0 if no sampling.',
    )

    return vars(parser.parse_args(args))


def main(*, n_measured_operators: Optional[int], samples_per_term: int):
    circuit, qubits = build_circuit()

    noise = cirq.ConstantQubitNoiseModel(cirq.depolarize(0.1))
    print(f'Noise model: {noise}')
    noisy_simulator = cirq.DensityMatrixSimulator(noise=noise)

    estimated_fidelity, _ = direct_fidelity_estimation(
        circuit,
        qubits,
        noisy_simulator,
        n_measured_operators=n_measured_operators,
        samples_per_term=samples_per_term,
    )
    print(f'Estimated fidelity: {estimated_fidelity:f}')


if __name__ == '__main__':
    main(**parse_arguments(sys.argv[1:]))
