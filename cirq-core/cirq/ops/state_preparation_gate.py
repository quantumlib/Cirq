# Copyright 2018 The Cirq Developers
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

"""Quantum gates defined by a matrix."""

from typing import Any, Sequence, Dict, List, Tuple, TYPE_CHECKING

import numpy as np

from cirq import protocols
from cirq.ops import raw_types

if TYPE_CHECKING:
    import cirq


class PrepareState(raw_types.Gate):
    """A unitary qubit gate which resets all qubits to the |0> state
    and then prepares the target state."""

    def __init__(self, target_state: np.ndarray, name: str = None) -> None:
        """Initializes a matrix gate.

        Args:
            target_state: The state vector that this gate should prepare.
        """
        if len(target_state.shape) != 1:
            raise ValueError('`target_state` must be a square 1d numpy array.')

        n = int(np.round(np.log2(target_state.shape[0] or 1)))
        if 2 ** n != target_state.shape[0]:
            raise ValueError(
                f'Matrix width ({target_state.shape[0]}) is not a power of 2 and '
                f'qid_shape is not specified.'
            )

        self._state = target_state
        self._num_qubits = n
        self._name = name if name is not None else "StatePreparation"
        self._qid_shape = (2,) * n

    def _json_dict_(self) -> Dict[str, Any]:
        """Converts the gate object into a serializable dictionary"""
        return {
            'cirq_type': self.__class__.__name__,
            'matrix': self._state.tolist(),
        }

    def _num_qubits_(self):
        return self._num_qubits

    @staticmethod
    def _has_unitary_() -> bool:
        """Checks and returns if the gate has a unitary representation.
        It doesn't, since the resetting of the channels is a non-unitary operations, it involves measurement."""
        return False

    @classmethod
    def _from_json_dict_(cls, target_state, **_kwargs):
        """Recreates the gate object from it's serialized form

        Args:
            target_state: the state to prepare using this gate
        """
        return cls(target_state=np.array(target_state))

    def _qid_shape_(self) -> Tuple[int, ...]:
        return self._qid_shape

    def _circuit_diagram_info_(
        self, _args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        """Returns the information required to draw out the circuit diagram for this gate."""
        symbols = (
            [self._name]
            if self._num_qubits == 1
            else [f'{self._name}[{i+1}]' for i in range(0, self._num_qubits)]
        )
        return protocols.CircuitDiagramInfo(wire_symbols=symbols)

    def _decompose_(self, qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        """Decompose the n-qubit diagonal gates into a Reset channel and a Matrix Gate."""
        decomposed_circ: List[Any] = [cirq.reset(qubit) for qubit in qubits]
        matrix = np.zeros(shape=(2 ** self._num_qubits, 2 ** self._num_qubits), dtype=np.complex)
        for idx, val in enumerate(self._state):
            matrix[idx][0] = val
        decomposed_circ.append(cirq.MatrixGate(matrix))
        return decomposed_circ
