# Copyright 2021 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Estimation of fidelity associated with experimental circuit executions."""
from typing import (
    Callable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    cast,
)

import numpy as np

from cirq.circuits import Circuit
from cirq.ops import QubitOrder, QubitOrderOrList
from cirq.sim import final_state_vector
from cirq.value import state_vector_to_probabilities


def linear_xeb_fidelity_from_probabilities(
    hilbert_space_dimension: int,
    probabilities: Sequence[float],
) -> float:
    """Linear XEB fidelity estimator.

    Estimates fidelity from ideal probabilities of observed bitstrings.

    This estimator makes two assumptions. First, it assumes that the circuit
    used in experiment is sufficiently scrambling that its output probabilities
    follow the Porter-Thomas distribution. This assumption holds for typical
    instances of random quantum circuits of sufficient depth. Second, it assumes
    that the circuit uses enough qubits so that the Porter-Thomas distribution
    can be approximated with the exponential distribution.

    In practice the validity of these assumptions can be confirmed by plotting
    a histogram of output probabilities and comparing it to the exponential
    distribution.

    The mean of this estimator is the true fidelity f and the variance is

        (1 + 2f - f^2) / M

    where f is the fidelity and M the number of observations, equal to
    len(probabilities). This is better than logarithmic XEB (see below)
    when fidelity is f < 0.32. Since this estimator is unbiased, the
    variance is equal to the mean squared error of the estimator.

    The estimator is intended for use with xeb_fidelity() below.

    Args:
        hilbert_space_dimension: Dimension of the Hilbert space on which
           the channel whose fidelity is being estimated is defined.
        probabilities: Ideal probabilities of bitstrings observed in
            experiment.
    Returns:
        Estimate of fidelity associated with an experimental realization
        of a quantum circuit.
    """
    return hilbert_space_dimension * np.mean(probabilities) - 1


def log_xeb_fidelity_from_probabilities(
    hilbert_space_dimension: int,
    probabilities: Sequence[float],
) -> float:
    """Logarithmic XEB fidelity estimator.

    Estimates fidelity from ideal probabilities of observed bitstrings.

    See `linear_xeb_fidelity_from_probabilities` for the assumptions made
    by this estimator.

    The mean of this estimator is the true fidelity f and the variance is

        (pi^2/6 - f^2) / M

    where f is the fidelity and M the number of observations, equal to
    len(probabilities). This is better than linear XEB (see above) when
    fidelity is f > 0.32. Since this estimator is unbiased, the variance
    is equal to the mean squared error of the estimator.

    The estimator is intended for use with xeb_fidelity() below.

    Args:
        hilbert_space_dimension: Dimension of the Hilbert space on which
           the channel whose fidelity is being estimated is defined.
        probabilities: Ideal probabilities of bitstrings observed in
            experiment.
    Returns:
        Estimate of fidelity associated with an experimental realization
        of a quantum circuit.
    """
    return np.log(hilbert_space_dimension) + np.euler_gamma + np.mean(np.log(probabilities))


def hog_score_xeb_fidelity_from_probabilities(
    hilbert_space_dimension: int,
    probabilities: Sequence[float],
) -> float:
    """XEB fidelity estimator based on normalized HOG score.

    Estimates fidelity from ideal probabilities of observed bitstrings.

    See `linear_xeb_fidelity_from_probabilities` for the assumptions made
    by this estimator.

    The mean of this estimator is the true fidelity f and the variance is

        (1/log(2)^2 - f^2) / M

    where f is the fidelity and M the number of observations, equal to
    len(probabilities). This is always worse than log XEB (see above).
    Since this estimator is unbiased, the variance is equal to the mean
    squared error of the estimator.

    The estimator is intended for use with xeb_fidelity() below. It is
    based on the HOG problem defined in https://arxiv.org/abs/1612.05903.

    Args:
        hilbert_space_dimension: Dimension of the Hilbert space on which
           the channel whose fidelity is being estimated is defined.
        probabilities: Ideal probabilities of bitstrings observed in
            experiment.
    Returns:
        Estimate of fidelity associated with an experimental realization
        of a quantum circuit.
    """
    score = np.mean(probabilities > np.log(2) / hilbert_space_dimension)
    return (2 * score - 1) / np.log(2)


def xeb_fidelity(
    circuit: Circuit,
    bitstrings: Sequence[int],
    qubit_order: QubitOrderOrList = QubitOrder.DEFAULT,
    amplitudes: Optional[Mapping[int, complex]] = None,
    estimator: Callable[[int, Sequence[float]], float] = linear_xeb_fidelity_from_probabilities,
) -> float:
    """Estimates XEB fidelity from one circuit using user-supplied estimator.

    Fidelity quantifies the similarity of two quantum states. Here, we estimate
    the fidelity between the theoretically predicted output state of circuit and
    the state produced in its experimental realization. Note that we don't know
    the latter state. Nevertheless, we can estimate the fidelity between the two
    states from the knowledge of the bitstrings observed in the experiment.

    In order to make the estimate more robust one should average the estimates
    over many random circuits. The API supports per-circuit fidelity estimation
    to enable users to examine the properties of estimate distribution over
    many circuits.

    See https://arxiv.org/abs/1608.00263 for more details.

    Args:
        circuit: Random quantum circuit which has been executed on quantum
            processor under test.
        bitstrings: Results of terminal all-qubit measurements performed after
            each circuit execution as integer array where each integer is
            formed from measured qubit values according to `qubit_order` from
            most to least significant qubit, i.e. in the order consistent with
            `cirq.final_state_vector`.
        qubit_order: Qubit order used to construct bitstrings enumerating
            qubits starting with the most significant qubit.
        amplitudes: Optional mapping from bitstring to output amplitude.
            If provided, simulation is skipped. Useful for large circuits
            when an offline simulation had already been performed.
        estimator: Fidelity estimator to use, see above. Defaults to the
            linear XEB fidelity estimator.
    Returns:
        Estimate of fidelity associated with an experimental realization of
        circuit which yielded measurements in bitstrings.
    Raises:
        ValueError: Circuit is inconsistent with qubit order or one of the
            bitstrings is inconsistent with the number of qubits.
    """
    dim = np.prod(circuit.qid_shape(), dtype=np.int64)

    if isinstance(bitstrings, tuple):
        bitstrings = list(bitstrings)

    for bitstring in bitstrings:
        if not 0 <= bitstring < dim:
            raise ValueError(
                f'Bitstring {bitstring} could not have been observed '
                f'on {len(circuit.qid_shape())} qubits.'
            )

    if amplitudes is None:
        output_state = final_state_vector(circuit, qubit_order=qubit_order)
        output_probabilities = state_vector_to_probabilities(output_state)
        bitstring_probabilities = output_probabilities[bitstrings]
    else:
        bitstring_probabilities = np.abs([amplitudes[bitstring] for bitstring in bitstrings]) ** 2
    return estimator(dim, bitstring_probabilities)


def linear_xeb_fidelity(
    circuit: Circuit,
    bitstrings: Sequence[int],
    qubit_order: QubitOrderOrList = QubitOrder.DEFAULT,
    amplitudes: Optional[Mapping[int, complex]] = None,
) -> float:
    """Estimates XEB fidelity from one circuit using linear estimator."""
    return xeb_fidelity(
        circuit,
        bitstrings,
        qubit_order,
        amplitudes,
        estimator=linear_xeb_fidelity_from_probabilities,
    )


def log_xeb_fidelity(
    circuit: Circuit,
    bitstrings: Sequence[int],
    qubit_order: QubitOrderOrList = QubitOrder.DEFAULT,
    amplitudes: Optional[Mapping[int, complex]] = None,
) -> float:
    """Estimates XEB fidelity from one circuit using logarithmic estimator."""
    return xeb_fidelity(
        circuit, bitstrings, qubit_order, amplitudes, estimator=log_xeb_fidelity_from_probabilities
    )


def least_squares_xeb_fidelity_from_expectations(
    measured_expectations: Sequence[float],
    exact_expectations: Sequence[float],
    uniform_expectations: Sequence[float],
) -> Tuple[float, List[float]]:
    """Least squares fidelity estimator.

    An XEB experiment collects data from the execution of random circuits
    subject to noise. The effect of applying a random circuit with unitary U is
    modeled as U followed by a depolarizing channel. The result is that the
    initial state |𝜓⟩ is mapped to a density matrix ρ_U as follows:

        |𝜓⟩ → ρ_U = f |𝜓_U⟩⟨𝜓_U| + (1 - f) I / D

    where |𝜓_U⟩ = U|𝜓⟩, D is the dimension of the Hilbert space, I / D is the
    maximally mixed state, and f is the fidelity with which the circuit is
    applied. Let O_U be an observable that is diagonal in the computational
    basis. Then the expectation of O_U on ρ_U is given by

        Tr(ρ_U O_U) = f ⟨𝜓_U|O_U|𝜓_U⟩ + (1 - f) Tr(O_U / D).

    This equation shows how f can be estimated, since Tr(ρ_U O_U) can be
    estimated from experimental data, and ⟨𝜓_U|O_U|𝜓_U⟩ and Tr(O_U / D) can be
    computed numerically.

    Let e_U = ⟨𝜓_U|O_U|𝜓_U⟩, u_U = Tr(O_U / D), and m_U denote the experimental
    estimate of Tr(ρ_U O_U). Then we estimate f by performing least squares
    minimization of the quantity

        f (e_U - u_U) - (m_U - u_U)

    over different random circuits (giving different U). The solution to the
    least squares problem is given by

        f = (∑_U (m_U - u_U) * (e_U - u_U)) / (∑_U (e_U - u_U)^2).

    Args:
        measured_expectations: A sequence of the m_U, the experimental estimates
            of the observable, one for each circuit U.
        exact_expectations: A sequence of the e_U, the exact value of the
            observable. The order should match the order of the
            `measured_expectations` argument.
        uniform_expectations: A sequence of the u_U, the expectation of the
            observable on a uniformly random bitstring. The order should match
            the order in the other arguments.

    Returns:
        A tuple of two values. The first value is the estimated fidelity.
        The second value is a list of the residuals

            f (e_U - u_U) - (m_U - u_U)

        of the least squares minimization.

    Raises:
        ValueError: The lengths of the input sequences are not all the same.
    """
    if not (len(measured_expectations) == len(exact_expectations) == len(uniform_expectations)):
        raise ValueError(
            'The lengths of measured_expectations, '
            'exact_expectations, and uniform_expectations must '
            'all be the same. Got lengths '
            f'{len(measured_expectations)}, '
            f'{len(exact_expectations)}, and '
            f'{len(uniform_expectations)}.'
        )
    numerator = 0.0
    denominator = 0.0
    for m, e, u in zip(measured_expectations, exact_expectations, uniform_expectations):
        numerator += (m - u) * (e - u)
        denominator += (e - u) ** 2
    fidelity = numerator / denominator
    residuals = [
        fidelity * (e - u) - (m - u)
        for m, e, u in zip(measured_expectations, exact_expectations, uniform_expectations)
    ]
    return fidelity, residuals


def least_squares_xeb_fidelity_from_probabilities(
    hilbert_space_dimension: int,
    observed_probabilities: Sequence[Sequence[float]],
    all_probabilities: Sequence[Sequence[float]],
    observable_from_probability: Optional[Callable[[float], float]] = None,
    normalize_probabilities: bool = True,
) -> Tuple[float, List[float]]:
    """Least squares fidelity estimator with observable based on probabilities.

    Using the notation from the docstring of
    `least_squares_xeb_fidelity_from_expectations`, this function computes the
    least squares fidelity estimate when the observable O_U has eigenvalue
    corresponding to the computational basis state |z⟩ given by g(p(z)), where
    p(z) = |⟨z|𝜓_U⟩|^2 and g is a function that can be specified. By default,
    g is the identity function, but other choices, such as the logarithm, are
    useful. By default, the probability p(z) is actually multiplied by the
    Hilbert space dimension D, so that the observable is actually g(D * p(z)).
    This behavior can be disabled by setting `normalize_probabilities` to
    False.

    Args:
        hilbert_space_dimension: Dimension of the Hilbert space on which
           the channel whose fidelity is being estimated is defined.
        observed_probabilities: Ideal probabilities of bitstrings observed in
            experiments. A list of lists, where each inner list contains the
            probabilities for a single circuit.
        all_probabilities: Ideal probabilities of all possible bitstrings.
            A list of lists, where each inner list contains the probabilities
            for a single circuit, and should have length equal to the Hilbert
            space dimension. The order of the lists should correspond to that
            of `observed_probabilities`.
        observable_from_probability: Function that computes the observable from
            a given probability.
        normalize_probabilities: Whether to multiply the probabilities by the
            Hilbert space dimension before computing the observable.

    Returns:
        A tuple of two values. The first value is the estimated fidelity.
        The second value is a list of the residuals

            f (e_U - u_U) - (m_U - u_U)

        of the least squares minimization.
    """
    if not isinstance(observable_from_probability, np.ufunc):
        if observable_from_probability is None:
            observable_from_probability = lambda p: p
        else:
            observable_from_probability = np.frompyfunc(observable_from_probability, 1, 1)
    observable_from_probability = cast(Callable, observable_from_probability)
    measured_expectations = []
    exact_expectations = []
    uniform_expectations = []
    prefactor = hilbert_space_dimension if normalize_probabilities else 1.0
    for observed_probs, all_probs in zip(observed_probabilities, all_probabilities):
        observed_probs = np.array(observed_probs)
        all_probs = np.array(all_probs)
        observable = observable_from_probability(prefactor * cast(np.ndarray, all_probs))
        measured_expectations.append(
            np.mean(observable_from_probability(prefactor * cast(np.ndarray, observed_probs)))
        )
        exact_expectations.append(np.sum(all_probs * observable))
        uniform_expectations.append(np.sum(observable) / hilbert_space_dimension)
    return least_squares_xeb_fidelity_from_expectations(
        measured_expectations, exact_expectations, uniform_expectations
    )
