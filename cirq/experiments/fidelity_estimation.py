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

from typing import Sequence

import numpy as np

from cirq.circuits import Circuit
from cirq.ops import QubitOrder, QubitOrderOrList
from cirq.sim import final_wavefunction


def linear_xeb_fidelity(
        circuit: Circuit,
        bitstrings: Sequence[int],
        qubit_order: QubitOrderOrList = QubitOrder.DEFAULT,
) -> float:
    """Computes fidelity estimate from one circuit using linear XEB estimator.

    Fidelity quantifies the similarity of two quantum states. Here, we estimate
    the fidelity between the theoretically predicted output state of circuit and
    the state producted in its experimental realization. Note that we don't know
    the latter state. Nevertheless, we can estimate the fidelity between the two
    states from the knowledge of the bitstrings observed in the experiment.

    This estimation procedure makes two assumptions. First, it assumes that the
    circuit is sufficiently scrambling that its output probabilities follow the
    Porter-Thomas distribution. This assumption holds for typical instances of
    random quantum circuits of sufficient depth. Second, it assumes that the
    circuit uses enough qubits so that the Porter-Thomas distribution can be
    approximated with the exponential distribution.

    In practice the validity of these assumptions can be confirmed by plotting
    a histogram of output probabilities and comparing it to the exponential
    distribution.

    In order to make the estimate more robust one should average the estimates
    over many random circuits. The API supports per-circuit fidelity estimation
    to enable users to examine the properties of estimate distribution over
    many circuits.

    See https://arxiv.org/abs/1608.00263 for more details.

    Args:
        circuit: Random quantum circuit which has been executed on quantum
            processor under test
        bitstrings: Results of terminal all-qubit measurements performed after
            each circuit execution as integer array where each integer is
            formed from measured qubit values according to `qubit_order` from
            most to least significant qubit, i.e. in the order consistent with
            `cirq.final_wavefunction`.
        qubit_order: Qubit order used to construct bitstrings enumerating
            qubits starting with the most sigificant qubit
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

    output_state = final_wavefunction(circuit, qubit_order=qubit_order)
    output_probabilities = np.abs(output_state)**2
    return dim * np.mean(output_probabilities[bitstrings]) - 1
