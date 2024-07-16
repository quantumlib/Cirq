# Copyright 2024 The Cirq Developers
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

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import cirq
import numpy as np
import numpy.typing as npt
from cirq.transformers import transformer_api


@transformer_api.transformer
class RandomizedMeasurements:
    def __init__(self, *, num_unitaries: int = 1, subsystem: Sequence[int] | None = None):
        """Class structure for performing and analyzing a general randomized measurement protocol.
        For more details on the randomized measurement toolbox see https://arxiv.org/abs/2203.11374

        Args:
            num_unitaries: Number of random pre-measurement unitaries
            subsystem: The specific subsystem (e.g qubit index) to measure in random basis
        """
        self.num_unitaries = num_unitaries
        self.subsystem = subsystem

    def __call__(
        self,
        circuit: 'cirq.AbstractCircuit',
        rng: np.random.Generator | None = None,
        *,
        context: transformer_api.TransformerContext | None = None,
    ):
        """Apply the transformer to the given circuit. Given an input circuit returns
        a list of circuits with the pre-measurement unitaries. If no arguments are specified,
        it will default to computing the entropy of the entire circuit.

        Args:
            circuit: The circuit to add randomized measurements to.
            rng: Random number generator.
            context: Not used; to satisfy transformer API.

        Returns:
            List of circuits with pre-measurement unitaries and measurements added
        """
        if rng is None:
            rng = np.random.default_rng()

        qubits = sorted(circuit.all_qubits())
        num_qubits = len(qubits)

        pre_measurement_unitaries_list = self._generate_unitaries_list(rng, num_qubits)
        circuit_list = []

        for unitaries in pre_measurement_unitaries_list:
            pre_measurement_moment = self.unitaries_to_moment(unitaries, qubits)
            temp_circuit = cirq.Circuit.from_moments(
                *circuit.moments, pre_measurement_moment, cirq.measure_each(*qubits)
            )

            circuit_list.append(temp_circuit)

        return circuit_list

    def _generate_unitaries_list(
        self, rng: np.random.Generator, num_qubits: int
    ) -> npt.NDArray[Any]:
        """Generates a list of pre-measurement unitaries."""

        pauli_strings = rng.choice(["X", "Y", "Z"], size=(self.num_unitaries, num_qubits))

        if self.subsystem is not None:
            for i in range(pauli_strings.shape[1]):
                if i not in self.subsystem:
                    pauli_strings[:, i] = np.array(["Z"] * self.num_unitaries)

        return pauli_strings

    def unitaries_to_moment(
        self, unitaries: Sequence[Literal["X", "Y", "Z"]], qubits: Sequence[Any]
    ) -> 'cirq.Moment':
        """Outputs the cirq moment associated with the pre-measurement rotations.
        Args:
            unitaries: List of pre-measurement unitaries
            qubits: List of qubits

        Returns: The cirq moment associated with the pre-measurement rotations
        """
        op_list: list[cirq.Operation] = []
        for idx, pauli in enumerate(unitaries):
            op_list.append(_pauli_basis_rotation(pauli).on(qubits[idx]))

        return cirq.Moment.from_ops(*op_list)


def _pauli_basis_rotation(basis: Literal["X", "Y", "Z"]) -> 'cirq.Gate':
    """Given a measurement basis returns the associated rotation.
    Args:
        basis: Measurement basis
    Returns: The cirq gate for associated with measurement basis
    """
    if basis == "X":
        return cirq.Ry(rads=-np.pi / 2)
    elif basis == "Y":
        return cirq.Rx(rads=np.pi / 2)
    elif basis == "Z":
        return cirq.I
