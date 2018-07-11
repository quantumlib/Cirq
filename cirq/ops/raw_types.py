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

"""Basic types defining qubits, gates, and operations."""

from typing import Sequence, Tuple, TYPE_CHECKING, Callable, TypeVar

from cirq import abc

if TYPE_CHECKING:
    # pylint: disable=unused-import
    from cirq.ops import gate_operation


class QubitId:
    """Identifies a qubit. Child classes provide specific types of qubits.

    Child classes must be equatable and hashable."""
    pass


class NamedQubit(QubitId):
    """A qubit identified by name."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return 'NamedQubit({})'.format(repr(self.name))

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
    def validate_args(self, qubits: Sequence[QubitId]) -> None:
        """Checks if this gate can be applied to the given qubits.

        Does no checks by default. Child classes can override.

        Args:
            qubits: The collection of qubits to potentially apply the gate to.

        Throws:
            ValueError: The gate can't be applied to the qubits.
        """
        pass

    def on(self, *qubits: QubitId) -> 'gate_operation.GateOperation':
        """Returns an application of this gate to the given qubits.

        Args:
            *qubits: The collection of qubits to potentially apply the gate to.
        """
        # Avoids circular import.
        from cirq.ops import gate_operation

        if len(qubits) == 0:
            raise ValueError(
                "Applied a gate to an empty set of qubits. Gate: {}".format(
                    repr(self)))
        self.validate_args(qubits)
        return gate_operation.GateOperation(self, list(qubits))

    def __call__(self, *args):
        return self.on(*args)


TSelf_Operation = TypeVar('TSelf_Operation', bound='Operation')


class Operation(metaclass=abc.ABCMeta):
    """An effect applied to a collection of qubits.

    The most common kind of Operation is a GateOperation, which separates its
    effect into a qubit-independent Gate and the qubits it should be applied to.
    """

    @abc.abstractproperty
    def qubits(self) -> Tuple[QubitId, ...]:
        raise NotImplementedError()

    @property
    def gate(self) -> Gate:
        # TODO: remove this once there are operations besides GateOperation.
        raise NotImplementedError()

    @abc.abstractmethod
    def with_qubits(self: TSelf_Operation,
                    *new_qubits: QubitId) -> TSelf_Operation:
        pass

    def transform_qubits(self: TSelf_Operation,
                         func: Callable[[QubitId], QubitId]) -> TSelf_Operation:
        """Returns the same operation, but with different qubits.

        Args:
            func: The function to use to turn each current qubit into a desired
                new qubit.

        Returns:
            The receiving operation but with qubits transformed by the given
                function.
        """
        return self.with_qubits(*(func(q) for q in self.qubits))
