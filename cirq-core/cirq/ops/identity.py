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
"""IdentityGate."""

from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING, Sequence, Union

import numpy as np
import sympy

from cirq import protocols, value
from cirq._doc import document
from cirq.type_workarounds import NotImplementedType
from cirq.ops import raw_types

if TYPE_CHECKING:
    import cirq


@value.value_equality
class IdentityGate(raw_types.Gate):
    """A Gate that perform no operation on qubits.

    The unitary matrix of this gate is a diagonal matrix with all 1s on the
    diagonal and all 0s off the diagonal in any basis.

    `cirq.I` is the single qubit identity gate.
    """

    def __init__(
        self, num_qubits: Optional[int] = None, qid_shape: Optional[Tuple[int, ...]] = None
    ) -> None:
        """Inits IdentityGate.

        Args:
            num_qubits: The number of qubits for the identity gate.
            qid_shape: Specifies the dimension of each qid the measurement
                applies to.  The default is 2 for every qubit.

        Raises:
            ValueError: If the length of qid_shape doesn't equal num_qubits, or
                neither `num_qubits` or `qid_shape` is supplied.

        """
        if qid_shape is None:
            if num_qubits is None:
                raise ValueError('Specify either the num_qubits or qid_shape argument.')
            qid_shape = (2,) * num_qubits
        elif num_qubits is None:
            num_qubits = len(qid_shape)
        self._qid_shape = qid_shape
        if len(self._qid_shape) != num_qubits:
            raise ValueError('len(qid_shape) != num_qubits')

    def _act_on_(self, sim_state: 'cirq.SimulationStateBase', qubits: Sequence['cirq.Qid']):
        return True

    def _qid_shape_(self) -> Tuple[int, ...]:
        return self._qid_shape

    def num_qubits(self) -> int:
        return len(self._qid_shape)

    def __pow__(self, power: Any) -> Any:
        if isinstance(power, (int, float, complex, sympy.Basic)):
            return self
        return NotImplemented

    def _commutes_(self, other: Any, *, atol: float = 1e-8) -> Union[bool, NotImplementedType]:
        """The identity gate commutes with all other gates."""
        if not isinstance(other, raw_types.Gate):
            return NotImplemented
        return True

    def _has_unitary_(self) -> bool:
        return True

    def _unitary_(self) -> np.ndarray:
        return np.identity(np.prod(self._qid_shape, dtype=np.int64).item())

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs') -> Optional[np.ndarray]:
        return args.target_tensor

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        if not all(d == 2 for d in self._qid_shape):
            return NotImplemented
        return value.LinearDict({'I' * self.num_qubits(): 1.0})

    def __repr__(self) -> str:
        if self._qid_shape == (2,):
            return 'cirq.I'
        if all(e == 2 for e in self._qid_shape):
            return f'cirq.IdentityGate({len(self._qid_shape)})'
        return f'cirq.IdentityGate(qid_shape={self._qid_shape!r})'

    def _decompose_(self, qubits) -> 'cirq.OP_TREE':
        return []

    def __str__(self) -> str:
        if self.num_qubits() == 1:
            return 'I'
        return f'I({self.num_qubits()})'

    def _value_equality_values_(self) -> Any:
        return self._qid_shape

    def _trace_distance_bound_(self) -> float:
        return 0.0

    def _json_dict_(self) -> Dict[str, Any]:
        other = {}
        if not all(d == 2 for d in self._qid_shape):
            other['qid_shape'] = self._qid_shape
        return {'num_qubits': len(self._qid_shape), **other}

    def _mul_with_qubits(self, qubits: Tuple['cirq.Qid', ...], other):
        if isinstance(other, raw_types.Operation):
            return other
        if isinstance(other, (complex, float, int)):
            from cirq.ops.pauli_string import PauliString

            return PauliString(coefficient=other)
        return NotImplemented

    _rmul_with_qubits = _mul_with_qubits

    def _circuit_diagram_info_(self, args) -> Tuple[str, ...]:
        return ('I',) * self.num_qubits()

    def _qasm_(self, args: 'cirq.QasmArgs', qubits: Tuple['cirq.Qid', ...]) -> Optional[str]:
        args.validate_version('2.0')
        return ''.join([args.format('id {0};\n', qubit) for qubit in qubits])

    @classmethod
    def _from_json_dict_(cls, num_qubits, qid_shape=None, **kwargs):
        return cls(num_qubits=num_qubits, qid_shape=None if qid_shape is None else tuple(qid_shape))


I = IdentityGate(num_qubits=1)
document(
    I,
    """The one qubit identity gate.

    Matrix:
    ```
        [[1, 0],
         [0, 1]]
    ```
    """,
)


def identity_each(*qubits: 'cirq.Qid') -> 'cirq.Operation':
    """Returns a single IdentityGate applied to all the given qubits.

    Args:
        *qubits: The qubits that the identity gate will apply to.

    Returns:
        An identity operation on the given qubits.

    Raises:
        ValueError: If the qubits are not instances of Qid.
    """
    for qubit in qubits:
        if not isinstance(qubit, raw_types.Qid):
            raise ValueError(f'Not a cirq.Qid: {qubit!r}.')
    return IdentityGate(qid_shape=protocols.qid_shape(qubits)).on(*qubits)
