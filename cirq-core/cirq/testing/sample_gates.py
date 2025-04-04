# Copyright 2023 The Cirq Developers
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
import dataclasses

import numpy as np

import cirq
from cirq import ops, protocols, qis


def _matrix_for_phasing_state(num_qubits, phase_state, phase):
    matrix = qis.eye_tensor((2,) * num_qubits, dtype=np.complex128)
    matrix = matrix.reshape((2**num_qubits, 2**num_qubits))
    matrix[phase_state, phase_state] = phase
    print(num_qubits, phase_state, phase)
    print(matrix)
    return matrix


@dataclasses.dataclass(frozen=True)
class PhaseUsingCleanAncilla(ops.Gate):
    r"""Phases the state $|phase\_state>$ by $\exp(1j * \pi * \theta)$ using one clean ancilla."""

    theta: float
    phase_state: int = 1
    target_bitsize: int = 1
    ancilla_bitsize: int = 1

    def _num_qubits_(self):
        return self.target_bitsize

    def _decompose_with_context_(self, qubits, *, context: protocols.DecompositionContext):
        anc = context.qubit_manager.qalloc(self.ancilla_bitsize)
        cv = [int(x) for x in f'{self.phase_state:0{self.target_bitsize}b}']
        cnot_ladder = [cirq.CNOT(anc[i - 1], anc[i]) for i in range(1, self.ancilla_bitsize)]

        yield ops.X(anc[0]).controlled_by(*qubits, control_values=cv)
        yield [cnot_ladder, ops.Z(anc[-1]) ** self.theta, reversed(cnot_ladder)]
        yield ops.X(anc[0]).controlled_by(*qubits, control_values=cv)

    def narrow_unitary(self) -> np.ndarray:
        """Narrowed unitary corresponding to the unitary effect applied on target qubits."""
        phase = np.exp(1j * np.pi * self.theta)
        return _matrix_for_phasing_state(self.target_bitsize, self.phase_state, phase)


@dataclasses.dataclass(frozen=True)
class PhaseUsingDirtyAncilla(ops.Gate):
    r"""Phases the state $|phase\_state>$ by -1 using one dirty ancilla."""

    phase_state: int = 1
    target_bitsize: int = 1
    ancilla_bitsize: int = 1

    def _num_qubits_(self):
        return self.target_bitsize

    def _decompose_with_context_(self, qubits, *, context: protocols.DecompositionContext):
        anc = context.qubit_manager.qalloc(self.ancilla_bitsize)
        cv = [int(x) for x in f'{self.phase_state:0{self.target_bitsize}b}']
        cnot_ladder = [cirq.CNOT(anc[i - 1], anc[i]) for i in range(1, self.ancilla_bitsize)]
        yield ops.X(anc[0]).controlled_by(*qubits, control_values=cv)
        yield [cnot_ladder, ops.Z(anc[-1]), reversed(cnot_ladder)]
        yield ops.X(anc[0]).controlled_by(*qubits, control_values=cv)
        yield [cnot_ladder, ops.Z(anc[-1]), reversed(cnot_ladder)]

    def narrow_unitary(self) -> np.ndarray:
        """Narrowed unitary corresponding to the unitary effect applied on target qubits."""
        return _matrix_for_phasing_state(self.target_bitsize, self.phase_state, -1)
