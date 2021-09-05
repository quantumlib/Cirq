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
from cirq.ops.common_channels import ResetChannel
from cirq.ops.matrix_gates import MatrixGate
from cirq._compat import proper_repr

if TYPE_CHECKING:
    import cirq


class PrepareState(raw_types.Gate):
    """A unitary qubit gate which resets all qubits to the |0> state
    and then prepares the target state."""

    def __init__(self, target_state: np.ndarray, name: str = "StatePreparation") -> None:
        """Initializes a matrix gate.

        Args:
            target_state: The state vector that this gate should prepare.
            name: the name of the gate

        Raises:
            ValueError: if the array is not 1D, or does not have 2**n elements for some n.
        """
        if len(target_state.shape) != 1:
            raise ValueError('`target_state` must be a 1d numpy array.')

        n = int(np.round(np.log2(target_state.shape[0] or 1)))
        if 2 ** n != target_state.shape[0]:
            raise ValueError(f'Matrix width ({target_state.shape[0]}) is not a power of 2')

        self._state = target_state.astype(np.complex) / np.linalg.norm(target_state)
        self._num_qubits = n
        self._name = name
        self._qid_shape = (2,) * n

    def _json_dict_(self) -> Dict[str, Any]:
        """Converts the gate object into a serializable dictionary"""
        return {
            'cirq_type': self.__class__.__name__,
            'target_state': self._state.tolist(),
        }

    def _num_qubits_(self):
        return self._num_qubits

    @staticmethod
    def _has_unitary_() -> bool:
        """Checks and returns if the gate has a unitary representation.
        It doesn't, since the resetting of the channels is a non-unitary operations,
        it involves measurement."""
        return False

    @classmethod
    def _from_json_dict_(cls, target_state, **_kwargs):
        """Recreates the gate object from it's serialized form

        Args:
            target_state: the state to prepare using this gate
            _kwargs: other keyword arguments, ignored
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

    def _get_unitary_transform(self):
        initial_basis = np.eye(2 ** self._num_qubits, dtype=np.complex)
        final_basis = [self._state]
        for vector in initial_basis:
            for new_basis_vector in final_basis:
                vector -= np.conj(np.dot(new_basis_vector, vector)) * new_basis_vector
            if not np.allclose(vector, 0):
                vector /= np.linalg.norm(vector)
                final_basis.append(vector)
        final_basis = np.stack(final_basis[: initial_basis.shape[0]], axis=1)
        return final_basis

    def _decompose_(self, qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        """Decompose the n-qubit diagonal gates into a Reset channel and a Matrix Gate."""
        decomposed_circ: List[Any] = [ResetChannel(qubit.dimension).on(qubit) for qubit in qubits]
        final_basis = self._get_unitary_transform()
        decomposed_circ.append(MatrixGate(final_basis).on(*qubits))
        return decomposed_circ

    def __repr__(self) -> str:
        return f'cirq.PrepareState({proper_repr(self._state)})'

    @property
    def state(self):
        return self._state

    def __eq__(self, other):
        return np.allclose(self._state, other.state)
