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

from typing import Sequence, Tuple, TYPE_CHECKING, Callable, TypeVar, Any

import abc

from cirq import protocols, value

if TYPE_CHECKING:
    # pylint: disable=unused-import
    from cirq.ops import gate_operation


class Qid(metaclass=abc.ABCMeta):
    """Identifies a quantum object such as a qubit, qudit, resonator, etc.

    Child classes represent specific types of objects, such as a qubit at a
    particular location on a chip or a qubit with a particular name.

    The main criteria that a custom qid must satisfy is *comparability*. Child
    classes meet this criteria by implementing the `_comparison_key` method. For
    example, `cirq.LineQubit`'s `_comparison_key` method returns `self.x`. This
    ensures that line qubits with the same `x` are equal, and that line qubits
    will be sorted ascending by `x`. `Qid` implements all equality,
    comparison, and hashing methods via `_comparison_key`.
    """

    @abc.abstractmethod
    def _comparison_key(self) -> Any:
        """Returns a value used to sort and compare this qubit with others.

        By default, qubits of differing type are sorted ascending according to
        their type name. Qubits of the same type are then sorted using their
        comparison key.
        """
        pass

    def _cmp_tuple(self):
        return type(self).__name__, repr(type(self)), self._comparison_key()

    def __hash__(self):
        return hash((Qid, self._comparison_key()))

    def __eq__(self, other):
        if not isinstance(other, Qid):
            return NotImplemented
        return self._cmp_tuple() == other._cmp_tuple()

    def __ne__(self, other):
        if not isinstance(other, Qid):
            return NotImplemented
        return self._cmp_tuple() != other._cmp_tuple()

    def __lt__(self, other):
        if not isinstance(other, Qid):
            return NotImplemented
        return self._cmp_tuple() < other._cmp_tuple()

    def __gt__(self, other):
        if not isinstance(other, Qid):
            return NotImplemented
        return self._cmp_tuple() > other._cmp_tuple()

    def __le__(self, other):
        if not isinstance(other, Qid):
            return NotImplemented
        return self._cmp_tuple() <= other._cmp_tuple()

    def __ge__(self, other):
        if not isinstance(other, Qid):
            return NotImplemented
        return self._cmp_tuple() >= other._cmp_tuple()


class Gate(metaclass=abc.ABCMeta):
    """An operation type that can be applied to a collection of qubits.

    Gates can be applied to qubits by calling their on() method with
    the qubits to be applied to supplied, or, alternatively, by simply
    calling the gate on the qubits.  In other words calling MyGate.on(q1, q2)
    to create an Operation on q1 and q2 is equivalent to MyGate(q1,q2).

    Gates operate on a certain number of qubits. All implementations of gate
    must implement the `num_qubits` method declaring how many qubits they
    act on. The gate feature classes `SingleQubitGate` and `TwoQubitGate`
    can be used to avoid writing this boilerplate.
    """

    def validate_args(self, qubits: Sequence[Qid]) -> None:
        """Checks if this gate can be applied to the given qubits.

        By default only checks qubit count. Child classes can override.

        Args:
            qubits: The collection of qubits to potentially apply the gate to.

        Throws:
            ValueError: The gate can't be applied to the qubits.
        """
        if len(qubits) != self.num_qubits():
            raise ValueError(
                'Wrong number of qubits for <{!r}>. '
                'Expected {} qubits but got <{!r}>.'.format(
                    self,
                    self.num_qubits(),
                    qubits))

    def on(self, *qubits: Qid) -> 'gate_operation.GateOperation':
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
        return gate_operation.GateOperation(self, list(qubits))

    def __pow__(self, power):
        if power == 1:
            return self

        if power == -1:
            # HACK: break cycle
            from cirq.line import line_qubit

            decomposed = protocols.decompose_once_with_qubits(
                self,
                qubits=line_qubit.LineQubit.range(self.num_qubits()),
                default=None)
            if decomposed is None:
                return NotImplemented

            inverse_decomposed = protocols.inverse(decomposed, None)
            if inverse_decomposed is None:
                return NotImplemented

            return _InverseCompositeGate(self)

        return NotImplemented

    def __call__(self, *args, **kwargs):
        return self.on(*args, **kwargs)

    @abc.abstractmethod
    def num_qubits(self) -> int:
        """The number of qubits this gate acts on."""
        raise NotImplementedError()


TSelf_Operation = TypeVar('TSelf_Operation', bound='Operation')


class Operation(metaclass=abc.ABCMeta):
    """An effect applied to a collection of qubits.

    The most common kind of Operation is a GateOperation, which separates its
    effect into a qubit-independent Gate and the qubits it should be applied to.
    """

    @abc.abstractproperty
    def qubits(self) -> Tuple[Qid, ...]:
        raise NotImplementedError()

    @abc.abstractmethod
    def with_qubits(self: TSelf_Operation,
                    *new_qubits: Qid) -> TSelf_Operation:
        pass

    def transform_qubits(self: TSelf_Operation,
                         func: Callable[[Qid], Qid]) -> TSelf_Operation:
        """Returns the same operation, but with different qubits.

        Args:
            func: The function to use to turn each current qubit into a desired
                new qubit.

        Returns:
            The receiving operation but with qubits transformed by the given
                function.
        """
        return self.with_qubits(*(func(q) for q in self.qubits))


@value.value_equality
class _InverseCompositeGate(Gate):
    """The inverse of a composite gate."""

    def __init__(self, original: Gate) -> None:
        self._original = original

    def num_qubits(self):
        return self._original.num_qubits()

    def __pow__(self, power):
        if power == 1:
            return self
        if power == -1:
            return self._original
        return NotImplemented

    def _decompose_(self, qubits):
        return protocols.inverse(protocols.decompose_once_with_qubits(
            self._original, qubits))

    def _value_equality_values_(self):
        return self._original

    def __repr__(self):
        return '({!r}**-1)'.format(self._original)
