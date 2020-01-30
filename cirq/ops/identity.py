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
"""IdentityGate and IdentityOperation."""

from typing import (Any, Iterable, List, Optional, Sequence, Tuple, Union,
                    TYPE_CHECKING)

import numpy as np
import sympy

from cirq import protocols, value
from cirq._compat import deprecated
from cirq._doc import document
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

    def __init__(self,
                 num_qubits: Optional[int] = None,
                 qid_shape: Tuple[int, ...] = None):
        """
        Args:
            num_qubits:
            qid_shape: Specifies the dimension of each qid the measurement
                applies to.  The default is 2 for every qubit.

        Raises:
            ValueError: If the length of qid_shape doesn't equal num_qubits.
        """
        if qid_shape is None:
            if num_qubits is None:
                raise ValueError(
                    'Specify either the num_qubits or qid_shape argument.')
            qid_shape = (2,) * num_qubits
        elif num_qubits is None:
            num_qubits = len(qid_shape)
        self._qid_shape = qid_shape
        if len(self._qid_shape) != num_qubits:
            raise ValueError('len(qid_shape) != num_qubits')

    def _qid_shape_(self) -> Tuple[int, ...]:
        return self._qid_shape

    def num_qubits(self) -> int:
        return len(self._qid_shape)

    def on_each(self, *targets: Union['cirq.Qid', Iterable[Any]]
               ) -> List['cirq.Operation']:
        """Returns a list of operations that applies the single qubit identity
        to each of the targets.

        Args:
            *targets: The qubits to apply this gate to.

        Returns:
            Operations applying this gate to the target qubits.

        Raises:
            ValueError if targets are not instances of Qid or List[Qid] or
            the gate from which this is applied is not a single qubit identity
            gate.
        """
        if len(self._qid_shape) != 1:
            raise ValueError(
                'IdentityGate only supports on_each when it is a one qubit '
                'gate.')
        operations: List['cirq.Operation'] = []
        for target in targets:
            if isinstance(target, Iterable) and not isinstance(target, str):
                operations.extend(self.on_each(*target))
            elif isinstance(target, raw_types.Qid):
                operations.append(self.on(target))
            else:
                raise ValueError(
                    'Gate was called with type different than Qid. Type: {}'.
                    format(type(target)))
        return operations

    def __pow__(self, power: Any) -> Any:
        if isinstance(power, (int, float, complex, sympy.Basic)):
            return self
        return NotImplemented

    def _unitary_(self):
        return np.identity(np.prod(self._qid_shape, dtype=int))

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs'
                       ) -> Optional[np.ndarray]:
        return args.target_tensor

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        if not all(d == 2 for d in self._qid_shape):
            return NotImplemented
        return value.LinearDict({'I' * self.num_qubits(): 1.0})

    def __repr__(self):
        if self._qid_shape == (2,):
            return 'cirq.I'
        if all(e == 2 for e in self._qid_shape):
            return f'cirq.IdentityGate({len(self._qid_shape)})'
        return f'cirq.IdentityGate(qid_shape={self._qid_shape!r})'

    def _decompose_(self, qubits):
        return []

    def __str__(self):
        if self.num_qubits() == 1:
            return 'I'
        return 'I({})'.format(self.num_qubits())

    def _value_equality_values_(self):
        return self._qid_shape

    def _trace_distance_bound_(self):
        return 0.0

    def _json_dict_(self):
        other = {}
        if not all(d == 2 for d in self._qid_shape):
            other['qid_shape'] = self._qid_shape
        return {
            'cirq_type': self.__class__.__name__,
            'num_qubits': len(self._qid_shape),
            **other,
        }

    def _mul_with_qubits(self, qubits: Tuple['cirq.Qid', ...], other):
        if isinstance(other, raw_types.Operation):
            return other
        if isinstance(other, (complex, float, int)):
            from cirq.ops.pauli_string import PauliString
            return PauliString(coefficient=other)
        return NotImplemented

    _rmul_with_qubits = _mul_with_qubits

    def _circuit_diagram_info_(self, args):
        return ('I',) * self.num_qubits()

    def _qasm_(self, args: 'cirq.QasmArgs',
               qubits: Tuple['cirq.Qid', ...]) -> Optional[str]:
        args.validate_version('2.0')
        return ''.join([args.format('id {0};\n', qubit) for qubit in qubits])

    @classmethod
    def _from_json_dict_(cls, num_qubits, qid_shape=None, **kwargs):
        return cls(num_qubits=num_qubits,
                   qid_shape=None if qid_shape is None else tuple(qid_shape))


class IdentityOperation(raw_types.Operation):
    """An application of the identity gate to a sequence of qubits."""

    @deprecated(deadline='v0.8',
                fix='Use cirq.IdentityGate or cirq.identity_each instead.',
                name='IdentityOperation')
    def __new__(cls, qubits: Sequence['cirq.Qid']):
        return IdentityGate(qid_shape=protocols.qid_shape(qubits)).on(*qubits)

    @property
    def qubits(self) -> Tuple['cirq.Qid', ...]:
        raise NotImplementedError('deprecated')

    def with_qubits(self, *new_qubits: 'cirq.Qid') -> 'cirq.Operation':
        raise NotImplementedError('deprecated')


I = IdentityGate(num_qubits=1)
document(
    I, """The one qubit identity gate.

    Matrix:

        [[1, 0],
         [0, 1]]
    """)


def identity_each(*qubits: 'cirq.Qid') -> 'cirq.Operation':
    """Returns a single IdentityGate applied to all the given qubits.

    Args:
        *qubits: The qubits that the identity gate will apply to.

    Returns:
        An identity operation on the given qubits.

    Raises:
        ValueError if the qubits are not instances of Qid.
    """
    for qubit in qubits:
        if not isinstance(qubit, raw_types.Qid):
            raise ValueError(f'Not a cirq.Qid: {qubit!r}.')
    return IdentityGate(qid_shape=protocols.qid_shape(qubits)).on(*qubits)
