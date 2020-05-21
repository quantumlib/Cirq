# Copyright 2019 The Cirq Developers
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

from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from cirq.circuits import Circuit
from cirq.ops import QubitOrder, QubitOrderOrList
from cirq.sim import final_state_vector


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
    return (np.log(hilbert_space_dimension) + np.euler_gamma +
            np.mean(np.log(probabilities)))


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
        estimator: Callable[[int, Sequence[float]],
                            float] = linear_xeb_fidelity_from_probabilities,
) -> float:
    """Estimates XEB fidelity from one circuit using user-supplied estimator.

    Fidelity quantifies the similarity of two quantum states. Here, we estimate
    the fidelity between the theoretically predicted output state of circuit and
    the state producted in its experimental realization. Note that we don't know
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
            qubits starting with the most sigificant qubit.
        amplitudes: Optional mapping from bitstring to output amplitude.
            If provided, simulation is skipped. Useful for large circuits
            when an offline simulation had already been peformed.
        estimator: Fidelity estimator to use, see above. Defaults to the
            linear XEB fidelity estimator.
    Returns:
        Estimate of fidelity associated with an experimental realization of
        circuit which yielded measurements in bitstrings.
    Raises:
        ValueError: Circuit is inconsistent with qubit order or one of the
            bitstrings is inconsistent with the number of qubits.
    """
    dim = np.product(circuit.qid_shape())

    if isinstance(bitstrings, tuple):
        bitstrings = list(bitstrings)

    for bitstring in bitstrings:
        if not 0 <= bitstring < dim:
            raise ValueError(
                f'Bitstring {bitstring} could not have been observed '
                f'on {len(circuit.qid_shape())} qubits.')

    if amplitudes is None:
        output_state = final_state_vector(circuit, qubit_order=qubit_order)
        output_probabilities = np.abs(output_state)**2
        bitstring_probabilities = output_probabilities[bitstrings]
    else:
        bitstring_probabilities = np.abs(
            [amplitudes[bitstring] for bitstring in bitstrings])**2
    return estimator(dim, bitstring_probabilities)


def linear_xeb_fidelity(
        circuit: Circuit,
        bitstrings: Sequence[int],
        qubit_order: QubitOrderOrList = QubitOrder.DEFAULT,
        amplitudes: Optional[Mapping[int, complex]] = None,
) -> float:
    """Estimates XEB fidelity from one circuit using linear estimator."""
    return xeb_fidelity(circuit,
                        bitstrings,
                        qubit_order,
                        amplitudes,
                        estimator=linear_xeb_fidelity_from_probabilities)


def log_xeb_fidelity(
        circuit: Circuit,
        bitstrings: Sequence[int],
        qubit_order: QubitOrderOrList = QubitOrder.DEFAULT,
        amplitudes: Optional[Mapping[int, complex]] = None,
) -> float:
    """Estimates XEB fidelity from one circuit using logarithmic estimator."""
    return xeb_fidelity(circuit,
                        bitstrings,
                        qubit_order,
                        amplitudes,
                        estimator=log_xeb_fidelity_from_probabilities)
