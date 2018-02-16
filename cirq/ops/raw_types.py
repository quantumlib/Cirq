# Copyright 2018 Google LLC
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

"""Basic types defining qubits, gates, and operations."""

from typing import Sequence


class QubitId:
    """Identifies a qubit. Child classes provide specific types of qubits.

    Child classes must be equatable and hashable."""
    pass


class NamedQubit(QubitId):
    """A qubit identified by name."""

    def __init__(self, name: str):
        self.name = name

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.name == other.name

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((NamedQubit, self.name))


class Gate:
    """An operation type that can be applied to a collection of qubits.

    Gates can be applied to qubits by calling their on() method with
    the qubits to be applied to supplied, or, alternatively, by simply
    calling the gate on the qubits.  In other words calling MyGate.on(q1, q2)
    to create an Operation on q1 and q2 is equivalent to MyGate(q1,q2).
    """

    # noinspection PyMethodMayBeStatic
    def validate_args(self, qubits: Sequence[QubitId]) -> type(None):
        """Checks if this gate can be applied to the given qubits.

        Does no checks by default. Child classes can override.

        Args:
            qubits: The collection of qubits to potentially apply the gate to.

        Throws:
            ValueError: The gate can't be applied to the qubits.
        """
        pass

    def on(self, *qubits: QubitId) -> 'Operation':
        """Returns an application of this gate to the given qubits.

        Args:
            *qubits: The collection of qubits to potentially apply the gate to.
        """
        self.validate_args(qubits)
        return Operation(self, list(qubits))

    def __call__(self, *args):
        return self.on(*args)


class InterchangeableQubitsGate:
    """Indicates operations should be equal under any qubit permutation."""
    pass


class Operation:
    """An application of a gate to a collection of qubits."""

    def __init__(self, gate: Gate, qubits: Sequence[QubitId]):
        self.gate = gate
        self.qubits = tuple(qubits)

    def __repr__(self):
        return 'Operation({}, {})'.format(repr(self.gate), repr(self.qubits))

    def __str__(self):
        return '{}({})'.format(self.gate,
                               ', '.join(str(e) for e in self.qubits))

    def __hash__(self):
        q = self.qubits
        if isinstance(self.gate, InterchangeableQubitsGate):
            q = frozenset(q)
        return hash((Operation, self.gate, q))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        q1, q2 = self.qubits, other.qubits
        if isinstance(self.gate, InterchangeableQubitsGate):
            q1 = frozenset(q1)
            q2 = frozenset(q2)
        return self.gate == other.gate and q1 == q2

    def __ne__(self, other):
        return not self == other

    def __pow__(self, power: float) -> 'Operation':
        """Raise gate to a power, then reapply to the same qubits.

        Only works if the gate implements gate_features.ExtrapolatableGate.
        For extrapolatable gate G this means the following two are equivalent:

            (G ** 1.5)(qubit)  or  G(qubit) ** 1.5

        Args:
            power: The amount to scale the gate's effect by.

        Returns:
            A new operation on the same qubits with the scaled gate.
        """
        return (self.gate ** power).on(*self.qubits)
