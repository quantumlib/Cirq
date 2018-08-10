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

"""A simplified time-slice of operations within a sequenced circuit."""

from typing import Any, Iterable, TypeVar, Callable, Sequence

from cirq import ops

TSelf_Moment = TypeVar('TSelf_Moment', bound='Moment')

class Moment(object):
    """A simplified time-slice of operations within a sequenced circuit.

    Note that grouping sequenced circuits into moments is an abstraction that
    may not carry over directly to the scheduling on the hardware or simulator.
    Operations in the same moment may or may not actually end up scheduled to
    occur at the same time. However the topological quantum circuit ordering
    will be preserved, and many schedulers or consumers will attempt to
    maximize the moment representation.

    Attributes:
        operations: A tuple of the Operations for this Moment.
        qubits: A set of the qubits acted upon by this Moment.
    """

    def __init__(self, operations: Iterable[ops.Operation] = ()) -> None:
        """Constructs a moment with the given operations.

        Args:
            operations: The operations applied within the moment.
                Will be frozen into a tuple before storing.

        Raises:
            ValueError: A qubit appears more than once.
        """
        self.operations = tuple(operations)

        # Check that operations don't overlap.
        affected_qubits = [q for op in self.operations for q in op.qubits]
        self.qubits = frozenset(affected_qubits)
        if len(affected_qubits) != len(self.qubits):
            raise ValueError(
                'Overlapping operations: {}'.format(self.operations))

    def operates_on(self, qubits: Iterable[ops.QubitId]) -> bool:
        """Determines if the moment has operations touching the given qubits.

        Args:
            qubits: The qubits that may or may not be touched by operations.

        Returns:
            Whether this moment has operations involving the qubits.
        """
        qubits = frozenset(qubits)
        return any(q in qubits for op in self.operations for q in op.qubits)

    def with_operation(self, operation: ops.Operation):
        """Returns an equal moment, but with the given op added.

        Args:
            operation: The operation to append.

        Returns:
            The new moment.
        """
        return Moment(self.operations + (operation,))

    def without_operations_touching(self, qubits: Iterable[ops.QubitId]):
        """Returns an equal moment, but without ops on the given qubits.

        Args:
            qubits: Operations that touch these will be removed.

        Returns:
            The new moment.
        """
        qubits = frozenset(qubits)
        if not self.operates_on(qubits):
            return self
        return Moment(
            operation for operation in self.operations
            if qubits.isdisjoint(frozenset(operation.qubits)))

    def __copy__(self):
        return type(self)(self.operations)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.operations == other.operations

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((Moment, self.operations))

    def __repr__(self):
        if not self.operations:
            return 'cirq.Moment()'
        return 'cirq.Moment(operations={})'.format(
            _list_repr_with_indented_item_lines(self.operations))

    def __str__(self):
        return ' and '.join(str(op) for op in self.operations)

    def transform_qubits(self: TSelf_Moment,
                         func: Callable[[ops.QubitId], ops.QubitId]
                         ) -> TSelf_Moment:
        return self.__class__(op.transform_qubits(func)
                for op in self.operations)


def _list_repr_with_indented_item_lines(items: Sequence[Any]) -> str:
    block = '\n'.join([repr(op) + ',' for op in items])
    indented = '    ' + '\n    '.join(block.split('\n'))
    return '[\n{}\n]'.format(indented)
