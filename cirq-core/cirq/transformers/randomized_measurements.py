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

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

import cirq
from cirq.ops import SingleQubitCliffordGate
from cirq.transformers import transformer_api


@transformer_api.transformer
class RandomizedMeasurements:
    """A transformer that appends a moment of random rotations from a given unitary ensemble (pauli,
    clifford, cue)"""

    def __init__(self, subsystem: Sequence[int] | None = None):
        """Class structure for performing and analyzing a general randomized measurement protocol.
        For more details on the randomized measurement toolbox see https://arxiv.org/abs/2203.11374

        Args:
            subsystem: The specific subsystem (e.g qubit index) to measure in random basis
            rest of the qubits are measured in the computational basis
        """
        self.subsystem = subsystem

    def __call__(
        self,
        circuit: cirq.AbstractCircuit,
        unitary_ensemble: str = "pauli",
        rng: np.random.Generator | None = None,
        *,
        context: transformer_api.TransformerContext | None = None,
    ) -> cirq.Circuit:
        """Apply the transformer to the given circuit. Given an input circuit returns
        a new circuit with the pre-measurement unitaries and measurements gates added.
        to the qubits in the subsystem provided.If no subsystem is specified in the
        construction of this class it defaults to measuring all the qubits in the
        randomized bases.

        Args:
            circuit: The circuit to add randomized measurements to.
            unitary_ensemble: Choice of unitary ensemble (pauli/clifford/cue(circular
                unitary ensemble))
            context: Not used; to satisfy transformer API.
            rng: Random number generator.

        Returns:
            A circuit with pre-measurement unitaries and measurements added
        """

        all_qubits = sorted(circuit.all_qubits())
        if self.subsystem is None:
            subsystem_qubits = all_qubits
        else:
            subsystem_qubits = [all_qubits[s] for s in self.subsystem]
        if rng is None:
            rng = np.random.default_rng()

        pre_measurement_moment = self.random_single_qubit_unitary_moment(
            unitary_ensemble, subsystem_qubits, rng
        )

        return cirq.Circuit.from_moments(
            *circuit.moments, pre_measurement_moment, cirq.M(*subsystem_qubits, key="m")
        )

    def random_single_qubit_unitary_moment(
        self, unitary_ensemble: str, qubits: Sequence[Any], rng: np.random.Generator
    ) -> cirq.Moment:
        """Outputs the cirq moment associated with the pre-measurement rotations.

        Args:
            unitary_ensemble: clifford, pauli, cue
            qubits: List of qubits
            rng: Random number generator to be used in sampling.

        Returns:
            The cirq moment associated with the pre-measurement rotations

        Raises:
            ValueError: When unitary_ensemble is not one of "cue", "pauli" or "clifford"
        """

        if unitary_ensemble.lower() == "pauli":
            unitaries = [_pauli_basis_rotation(rng) for _ in range(len(qubits))]

        elif unitary_ensemble.lower() == "clifford":
            unitaries = [_single_qubit_clifford(rng) for _ in range(len(qubits))]

        elif unitary_ensemble.lower() == "cue":
            unitaries = [_single_qubit_cue(rng) for _ in range(len(qubits))]

        else:
            raise ValueError("Only pauli, clifford and cue unitaries are available")

        op_list: list[cirq.Operation] = []

        for idx, unitary in enumerate(unitaries):
            op_list.append(unitary.on(qubits[idx]))

        return cirq.Moment.from_ops(*op_list)


def _pauli_basis_rotation(rng: np.random.Generator) -> cirq.Gate:
    """Randomly generate a Pauli basis rotation.

    Args:
        rng: Random number generator

    Returns:
        cirq gate
    """
    basis_idx = rng.choice(np.arange(3))

    if basis_idx == 0:
        gate: cirq.Gate = cirq.Ry(rads=-np.pi / 2)
    elif basis_idx == 1:
        gate = cirq.Rx(rads=np.pi / 2)
    else:
        gate = cirq.I
    return gate


def _single_qubit_clifford(rng: np.random.Generator) -> cirq.Gate:
    """Randomly generate a single-qubit Clifford rotation.

    Args:
        rng: Random number generator

    Returns:
        cirq gate
    """

    # there are 24 distinct single-qubit Clifford gates
    clifford_idx = rng.choice(np.arange(24))

    return SingleQubitCliffordGate.to_phased_xz_gate(
        SingleQubitCliffordGate.all_single_qubit_cliffords[clifford_idx]
    )


def _single_qubit_cue(rng: np.random.Generator) -> cirq.Gate:
    """Randomly generate a CUE gate.

    Args:
        rng: Random number generator

    Returns:
        cirq gate
    """

    # phasedxz parameters are distinct between -1 and +1
    x_exponent, z_exponent, axis_phase_exponent = 1 - 2 * rng.random(size=3)

    return cirq.PhasedXZGate(
        x_exponent=x_exponent, z_exponent=z_exponent, axis_phase_exponent=axis_phase_exponent
    )
