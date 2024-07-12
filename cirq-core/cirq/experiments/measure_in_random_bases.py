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


class RandomizedMeasurements:
    def __init__(
        self,
        num_qubits: int,
        num_unitaries: int,
        subsystem: Sequence[str | int] | None = None,
        qubit_mapping: Mapping[int, str | int] | None = None,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        """Class structure for performing and analyzing a general randomized measurement protocol.
        For more details on the randomized measurement toolbox see https://arxiv.org/abs/2203.11374
        Args:
            num_qubits: Number of qubits in the circuit
            num_unitaries: Number of random pre-measurement unitaries
            subsystem: The specific subsystem measured in random basis
            qubit_mapping: The mapping between the measurement bitstring index and qubit specifier
            rng: Random generator
        """
        self.num_qubits = num_qubits
        self.num_unitaries = num_unitaries
        self.subsystem = subsystem

        self.qubit_mapping = (
            qubit_mapping if qubit_mapping else {i: i for i in range(self.num_qubits)}
        )

        self.rng = rng

        self.pre_measurement_unitaries_list = self._generate_unitaries_list()

    def _generate_unitaries_list(self) -> npt.NDArray[Any]:
        """Generates a list of pre-measurement unitaries."""

        pauli_strings = self.rng.choice(["X", "Y", "Z"], size=(self.num_unitaries, self.num_qubits))

        if self.subsystem is not None:
            for i in range(pauli_strings.shape[1]):
                if self.qubit_mapping[i] not in self.subsystem:
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


def append_randomized_measurements(
    circuit: 'cirq.AbstractCircuit',
    randomized_measurements_generator: RandomizedMeasurements | None = None,
    *,
    subsystem: tuple[int] | None = None,
    qubits: Sequence | None = None,
    num_unitaries: int | None = None,
    rng: np.random.Generator = np.random.default_rng(),
) -> Sequence['cirq.Circuit']:
    """Given an input circuit returns a list of circuits with the pre-measurement unitaries.
    If no arguments are specified, it will default to computing the entropy of the entire
    circuit.

    Args:
        circuit: The input circuit
        randomized_measurements_generator: RandomizedMeasurements class to use for
        generating random measurements.
        subsystem: The specific subsystem measured in random basis.
        qubits: A sequence of qubits to measure in random basis.
        num_unitaries: The number of random pre-measurement unitaries to append.
        rng: Random number genergate
    Returns:
        List of circuits with pre-measurement unitaries and measurements added
    """
    qubits = qubits or list(circuit.all_qubits())

    if randomized_measurements_generator is None:
        randomized_measurements_generator = RandomizedMeasurements(
            len(qubits),
            num_unitaries if num_unitaries else len(qubits),
            subsystem=subsystem,
            rng=rng,
        )

    circuit_list = []

    for unitaries in randomized_measurements_generator.pre_measurement_unitaries_list:
        pre_measurement_moment = randomized_measurements_generator.unitaries_to_moment(
            unitaries, qubits
        )

        temp_circuit = cirq.Circuit.from_moments(
            *circuit.moments, pre_measurement_moment, cirq.measure_each(*qubits)
        )

        circuit_list.append(temp_circuit)

    return circuit_list
