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

from typing import (Any, Iterable, FrozenSet, List, Optional, Sequence, Tuple,
                    Union)

import numpy as np
import sympy

from cirq import protocols, value
from cirq.ops import raw_types


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

    def on(self, *qubits: raw_types.Qid) -> raw_types.Operation:
        """Returns an application of this gate to the given qubits.

        Args:
            *qubits: The collection of qubits to potentially apply the gate to.
        """
        return IdentityOperation(list(qubits))

    def on_each(self, *targets: Union[raw_types.Qid, Iterable[Any]]
               ) -> List[raw_types.Operation]:
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
        operations: List[raw_types.Operation] = []
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
        other = ''
        if not all(d == 2 for d in self._qid_shape):
            other = ', {!r}'.format(self._qid_shape)
        return 'cirq.IdentityGate({!r}{})'.format(self.num_qubits(), other)

    def __str__(self):
        if (self.num_qubits() == 1):
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

    @classmethod
    def _from_json_dict_(cls, num_qubits, qid_shape=None, **kwargs):
        return cls(num_qubits=num_qubits,
                   qid_shape=None if qid_shape is None else tuple(qid_shape))


@value.value_equality(approximate=True)
class IdentityOperation(raw_types.Operation):
    """An application of the identity gate to a sequence of qubits."""

    def __init__(self, qubits: Sequence[raw_types.Qid]) -> None:
        """
        Args:
            qubits: The qubits to operate on.
        """
        if isinstance(qubits, raw_types.Qid):
            qubits = [qubits]
        if len(qubits) == 0:
            raise ValueError(
                'Applied an identity gate to an empty set of qubits.')

        if any(not isinstance(qubit, raw_types.Qid) for qubit in qubits):
            raise ValueError(
                'Gave non-Qid objects to IdentityOperation: {!r}'.format(
                    qubits))
        self._qubits = tuple(qubits)

    @property
    def qubits(self) -> Tuple[raw_types.Qid, ...]:
        """The qubits targeted by the operation."""
        return self._qubits

    def with_qubits(self, *new_qubits: raw_types.Qid) -> 'raw_types.Operation':
        return IdentityOperation(new_qubits)

    def __repr__(self):
        # Abbreviate when possible.
        if len(self.qubits) == 1:
            return f'cirq.I.on({self._qubits[0]!r})'

        return 'cirq.IdentityOperation(qubits={!r})'.format(list(self._qubits))

    def __str__(self):
        return 'I({})'.format(', '.join(str(e) for e in self._qubits))

    def _value_equality_values_(self) -> FrozenSet[raw_types.Qid]:
        return frozenset(self._qubits)

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        return value.LinearDict({'I' * len(self._qubits): 1.0})

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs'
                       ) -> Optional[np.ndarray]:
        return args.target_tensor

    def _circuit_diagram_info_(self, args: 'protocols.CircuitDiagramInfoArgs'
                              ) -> 'protocols.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(wire_symbols=('I',) *
                                            len(self._qubits),
                                            connected=True)

    def _qasm_(self, args: 'protocols.QasmArgs') -> Optional[str]:
        args.validate_version('2.0')
        return ''.join(
            [args.format('id {0};\n', qubit) for qubit in self._qubits])

    def __mul__(self, other):
        if isinstance(other, raw_types.Operation):
            return other
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, power):
        if isinstance(power, (int, float, complex, sympy.Basic)):
            return self
        return NotImplemented

    def _json_dict_(self):
        return {
            'cirq_type': self.__class__.__name__,
            'qubits': self._qubits,
        }

    @classmethod
    def _from_json_dict_(cls, qubits, **kwargs):
        return cls(qubits=qubits)


# The one qubit identity gate.
#
# Matrix:
#
#     [[1, 0],
#      [0, 1]]
I = IdentityGate(num_qubits=1)


def identity_each(*qubits: raw_types.Qid) -> raw_types.Operation:
    """Returns a single IdentityGate applied to all the given qubits.

    Args:
        *qubits: The qubits that the identity gate will apply to.

    Returns:
        An identity operation on the given qubits.

    Raises:
        ValueError if the qubits are not instances of Qid.
    """
    if not all(isinstance(qubit, raw_types.Qid) for qubit in qubits):
        raise ValueError('identity() was called with type different than Qid.')

    qid_shape = protocols.qid_shape(qubits)
    return IdentityGate(len(qubits), qid_shape).on(*qubits)
