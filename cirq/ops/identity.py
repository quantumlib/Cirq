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

from typing import (
    FrozenSet, Optional, Sequence, Tuple
)

import numpy as np
import sympy

from cirq import protocols, value
from cirq.ops import raw_types, gate_features
from cirq.type_workarounds import NotImplementedType


@value.value_equality
class IdentityGate(raw_types.Gate):
    """A Gate that perform no operation on qubits.

    The unitary matrix of this gate is a diagonal matrix with all 1s on the
    diagonal and all 0s off the diagonal in any basis.

    `cirq.I` is the single qubit identity gate.
    """

    def __init__(self, num_qubits):
        self._num_qubits = num_qubits

    def num_qubits(self) -> int:
        return self._num_qubits

    def _unitary_(self):
        return np.identity(2 ** self.num_qubits())

    def _apply_unitary_(
        self, args: protocols.ApplyUnitaryArgs) -> Optional[np.ndarray]:
        return args.target_tensor

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        return value.LinearDict({'I' * self.num_qubits(): 1.0})

    def __repr__(self):
        if self.num_qubits() == 1:
            return 'cirq.I'
        return 'cirq.IdentityGate({!r})'.format(self.num_qubits())

    def __str__(self):
        if (self.num_qubits() == 1):
            return 'I'
        else:
            return 'I({})'.format(self.num_qubits())

    def _circuit_diagram_info_(self,
        args: protocols.CircuitDiagramInfoArgs) -> protocols.CircuitDiagramInfo:
        return protocols.CircuitDiagramInfo(
            wire_symbols=('I',) * self.num_qubits(), connected=True)

    def _value_equality_values_(self):
        return self.num_qubits(),

    def on(self, *qubits: raw_types.Qid) -> raw_types.Operation:
        """Returns an application of this gate to the given qubits.

        Args:
            *qubits: The collection of qubits to potentially apply the gate to.
        """
        return IdentityOperation(list(qubits))

    def __pow__(self, power):
        if isinstance(power, (int, float, complex, sympy.Basic)):
            return IdentityOperation
        return NotImplemented


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
                'Gave non-Qid objects to IdentityOperation: {!r}'.format(qubits))
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

    def _unitary_(self):
        return np.identity(2 ** len(self._qubits))

    def _apply_unitary_(self, args: protocols.ApplyUnitaryArgs
                       ) -> Optional[np.ndarray]:
        return args.target_tensor

    def _circuit_diagram_info_(self, args: protocols.CircuitDiagramInfoArgs
                              ) -> protocols.CircuitDiagramInfo:
        return protocols.CircuitDiagramInfo(wire_symbols=('I',) *
                                            len(self._qubits),
                                            connected=True)

    def _trace_distance_bound_(self): 
        return 0

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


# The one qubit identity gate.
#
# Matrix:
#
#     [[1, 0],
#      [0, 1]]
I = IdentityGate(num_qubits=1)
