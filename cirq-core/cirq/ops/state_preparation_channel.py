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

"""Quantum gates to prepare a given target state."""

from typing import Any, Dict, Tuple, Iterable, TYPE_CHECKING

import numpy as np

from cirq import protocols
from cirq.ops import raw_types
from cirq._compat import proper_repr

if TYPE_CHECKING:
    import cirq


class StatePreparationChannel(raw_types.Gate):
    """A channel which prepares any state provided as the state vector on it's target qubits."""

    def __init__(self, target_state: np.ndarray, *, name: str = "StatePreparation") -> None:
        """Initializes a State Preparation channel.

        Args:
            target_state: The state vector that this gate should prepare.
            name: the name of the gate, used when printing it in the circuit diagram

        Raises:
            ValueError: if the array is not 1D, or does not have 2**n elements for some integer n.
        """
        if len(target_state.shape) != 1:
            raise ValueError('`target_state` must be a 1d numpy array.')

        n = int(np.round(np.log2(target_state.shape[0] or 1)))
        if 2 ** n != target_state.shape[0]:
            raise ValueError(f'Matrix width ({target_state.shape[0]}) is not a power of 2')

        self._state = target_state.astype(np.complex128) / np.linalg.norm(target_state)
        self._num_qubits = n
        self._name = name
        self._qid_shape = (2,) * n

    def _has_unitary_(self) -> bool:
        """Checks and returns if the gate has a unitary representation.
        It doesn't, since the resetting of the channels is a non-unitary operations,
        it involves measurement."""
        return False

    def _json_dict_(self) -> Dict[str, Any]:
        """Converts the gate object into a serializable dictionary"""
        return {
            'cirq_type': self.__class__.__name__,
            'target_state': self._state.tolist(),
            'name': self._name,
        }

    @classmethod
    def _from_json_dict_(
        cls, target_state: np.ndarray, name: str, **kwargs
    ) -> 'StatePreparationChannel':
        """Recreates the channel object from it's serialized form

        Args:
            target_state: the state to prepare using this channel
            name: the name of the gate for printing in circuit diagrams
            kwargs: other keyword arguments, ignored
        """
        return cls(target_state=np.array(target_state), name=name)

    def _num_qubits_(self) -> int:
        return self._num_qubits

    def _qid_shape_(self) -> Tuple[int, ...]:
        return self._qid_shape

    def _circuit_diagram_info_(
        self, _args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        """Returns the information required to draw out the circuit diagram for this channel."""
        symbols = (
            [self._name]
            if self._num_qubits == 1
            else [f'{self._name}[{i+1}]' for i in range(0, self._num_qubits)]
        )
        return protocols.CircuitDiagramInfo(wire_symbols=symbols)

    def _has_kraus_(self) -> bool:
        return True

    def _kraus_(self) -> Iterable[np.ndarray]:
        """Returns the Kraus operator for this gate

        The Kraus Operator is |Psi><i| for all |i>, where |Psi> is the target state.
        This allows is to take any input state to the target state.
        The operator satisfies the completeness relation Sum(E^ E) = I.
        """
        operator = np.zeros(shape=(2 ** self._num_qubits,) * 3, dtype=np.complex128)
        for i in range(len(operator)):
            operator[i, :, i] = self._state
        return operator

    def __repr__(self) -> str:
        return (
            f'cirq.StatePreparationChannel('
            f'target_state={proper_repr(self.state)}, name="{self._name}")'
        )

    def __str__(self) -> str:
        return f'StatePreparationChannel({self.state})'

    def _approx_eq_(self, other: Any, atol) -> bool:
        if not isinstance(other, StatePreparationChannel):
            return False
        return np.allclose(self.state, other.state, rtol=0, atol=atol)

    def __eq__(self, other) -> bool:
        if not isinstance(other, StatePreparationChannel):
            return False
        return np.array_equal(self.state, other.state)

    @property
    def state(self) -> np.ndarray:
        return self._state
