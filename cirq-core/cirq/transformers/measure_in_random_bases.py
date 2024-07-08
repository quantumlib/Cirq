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


import random
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Optional, List, Tuple

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
        rng: np.random.Generator | None = None,
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

        self.rng = rng if rng else np.random.default_rng()

        self.pre_measurement_unitaries_list = self._generate_unitaries_list()

    def _generate_unitaries_list(self) -> Sequence[Sequence[Any]]:
        """Generates a list of pre-measurement unitaries."""

        pauli_strings = self.rng.choice(["X", "Y", "Z"], size=(self.num_unitaries, self.num_qubits))

        if self.subsystem is not None:
            for i in range(pauli_strings.shape[1]):
                if self.qubit_mapping[i] not in self.subsystem:
                    pauli_strings[:, i] = np.array(["Z"] * self.num_unitaries)

        return pauli_strings

    def unitaries_to_moment(
        self, unitaries: Sequence[Literal["X", "Y", "Z"]], qubits: Sequence[Any]
    ) -> cirq.Moment:
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

    def process_measurements(self, bitstrings: npt.NDArray[np.int8]):
        """Processes the measurement results.

        To be over-written by the child class

        Args:
            bitstrings: A list of 0s and 1s resulting from the circuit measurement
        """


@staticmethod
def _pauli_basis_rotation(basis: Literal["X", "Y", "Z"]) -> cirq.Gate:
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


@staticmethod
def _random_cue_gates(qubit_locs: List[Tuple[int, int]], clifford: bool = False) -> cirq.Circuit:
    qubits = [cirq.GridQubit(qubit_locs[i][0], qubit_locs[i][1]) for i in range(len(qubit_locs))]
    circuit = cirq.Circuit()

    phi_vals = np.linspace(0, 2 * np.pi, 1001)
    chi_vals = np.linspace(0, 2 * np.pi, 1001)
    theta_vals = np.linspace(0, 1, 1001)
    theta_vals = np.arccos(np.sqrt(theta_vals)) * 2
    nums = np.linspace(-1, 1, 1001)

    for qubit in qubits:
        if clifford:
            phi = random.choice(phi_vals)
            chi = random.choice(chi_vals)
            theta = random.choice(theta_vals)
            circuit.append(cirq.Z(qubit) ** ((phi - chi) / np.pi))
            circuit.append(cirq.X(qubit) ** (theta / np.pi))
            circuit.append(cirq.Z(qubit) ** ((phi + chi) / np.pi))
        else:
            gate = cirq.PhasedXZGate(
                x_exponent=random.choice(nums),
                z_exponent=random.choice(nums),
                axis_phase_exponent=random.choice(nums),
            )
            circuit.append(gate(qubit))

    return circuit


def append_randomized_measurements(
    circuit: 'cirq.AbstractCircuit', *, context: Optional['cirq.TransformerContext'] = None
) -> Sequence['cirq.Circuit']:
    """Given an input circuit returns a list of circuits with the pre-measurement unitaries.

    Args:
        circuit: Input circuit
        context: `cirq.TransformerContext` storing common configurable options
            for transformers. The default has `deep=True` to ensure
            measurements at all levels are dephased.

    Returns: List of circuits with pre-measurement unitaries and measurements added
    """
    qubits = circuit.all_qubits()

    randomized_measurement_circuits = RandomizedMeasurements(len(qubits), len(qubits))

    circuit_list = []

    for unitaries in randomized_measurement_circuits.pre_measurement_unitaries_list:
        pre_measurement_moment = randomized_measurement_circuits.unitaries_to_moment(
            unitaries, list(qubits)
        )

        temp_circuit = cirq.Circuit.from_moments(
            *circuit.moments, pre_measurement_moment, cirq.measure_each(*qubits)
        )

        circuit_list.append(temp_circuit)

    return circuit_list
