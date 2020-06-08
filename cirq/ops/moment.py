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

from typing import (Any, Callable, Dict, FrozenSet, Iterable, Iterator,
                    overload, Tuple, TYPE_CHECKING, TypeVar, Union)
from cirq import protocols
from cirq._compat import deprecated_parameter
from cirq.ops import raw_types

if TYPE_CHECKING:
    import cirq

TSelf_Moment = TypeVar('TSelf_Moment', bound='Moment')


class Moment:
    """A time-slice of operations within a circuit.

    Grouping operations into moments is intended to be a strong suggestion to
    whatever is scheduling operations on real hardware. Operations in the same
    moment should execute at the same time (to the extent possible; not all
    operations have the same duration) and it is expected that all operations
    in a moment should be completed before beginning the next moment.

    Moment can be indexed by qubit or list of qubits:
        moment[qubit] returns the Operation in the moment which touches the
            given qubit, or throws KeyError if there is no such operation.
        moment[qubits] returns another Moment which consists only of those
            operations which touch at least one of the given qubits. If there
            are no such operations, returns an empty Moment.
    """

    @deprecated_parameter(
        deadline='v0.9',
        fix='Don\'t specify a keyword.',
        match=lambda _, kwargs: 'operations' in kwargs,
        parameter_desc='operations',
        rewrite=lambda args, kwargs: (args + (kwargs['operations'],), {}))
    def __init__(self, *contents: 'cirq.OP_TREE') -> None:
        """Constructs a moment with the given operations.

        Args:
            operations: The operations applied within the moment.
                Will be flattened and frozen into a tuple before storing.

        Raises:
            ValueError: A qubit appears more than once.
        """
        from cirq.ops import op_tree
        self._operations = tuple(op_tree.flatten_to_ops(contents))

        # Check that operations don't overlap.
        affected_qubits = [q for op in self.operations for q in op.qubits]
        self._qubits = frozenset(affected_qubits)
        if len(affected_qubits) != len(self._qubits):
            raise ValueError(
                'Overlapping operations: {}'.format(self.operations))

    @property
    def operations(self) -> Tuple['cirq.Operation', ...]:
        return self._operations

    @property
    def qubits(self) -> FrozenSet['cirq.Qid']:
        return self._qubits

    def operates_on_single_qubit(self, qubit: 'cirq.Qid') -> bool:
        """Determines if the moment has operations touching the given qubit.
        Args:
            qubit: The qubit that may or may not be touched by operations.
        Returns:
            Whether this moment has operations involving the qubit.
        """
        return qubit in self.qubits

    def operates_on(self, qubits: Iterable['cirq.Qid']) -> bool:
        """Determines if the moment has operations touching the given qubits.

        Args:
            qubits: The qubits that may or may not be touched by operations.

        Returns:
            Whether this moment has operations involving the qubits.
        """
        return bool(set(qubits) & self.qubits)

    def with_operation(self, operation: 'cirq.Operation') -> 'cirq.Moment':
        """Returns an equal moment, but with the given op added.

        Args:
            operation: The operation to append.

        Returns:
            The new moment.
        """
        if any(q in self._qubits for q in operation.qubits):
            raise ValueError('Overlapping operations: {}'.format(operation))

        # Use private variables to facilitate a quick copy
        m = Moment()
        m._operations = self.operations + (operation,)
        m._qubits = frozenset(self._qubits.union(set(operation.qubits)))

        return m

    def without_operations_touching(self, qubits: Iterable['cirq.Qid']
                                   ) -> 'cirq.Moment':
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

    def _operation_touching(self, qubit: raw_types.Qid) -> 'cirq.Operation':
        """Returns the operation touching given qubit.
        Args:
            qubit: Operations that touch this qubit will be returned.
        Returns:
            The operation which touches `qubit`.
        """
        for op in self.operations:
            if qubit in op.qubits:
                return op
        raise KeyError("Moment doesn't act on given qubit")

    def __copy__(self):
        return type(self)(self.operations)

    def __bool__(self) -> bool:
        return bool(self.operations)

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented

        return (sorted(self.operations, key=lambda op: op.qubits) == sorted(
            other.operations, key=lambda op: op.qubits))

    def _approx_eq_(self, other: Any, atol: Union[int, float]) -> bool:
        """See `cirq.protocols.SupportsApproximateEquality`."""
        if not isinstance(other, type(self)):
            return NotImplemented

        return protocols.approx_eq(sorted(self.operations,
                                          key=lambda op: op.qubits),
                                   sorted(other.operations,
                                          key=lambda op: op.qubits),
                                   atol=atol)

    def __ne__(self, other) -> bool:
        return not self == other

    def __hash__(self):
        return hash(
            (Moment, tuple(sorted(self.operations, key=lambda op: op.qubits))))

    def __iter__(self) -> Iterator['cirq.Operation']:
        return iter(self.operations)

    def __pow__(self, power):
        if power == 1:
            return self
        new_ops = []
        for op in self.operations:
            new_op = protocols.pow(op, power, default=None)
            if new_op is None:
                return NotImplemented
            new_ops.append(new_op)
        return Moment(new_ops)

    def __len__(self) -> int:
        return len(self.operations)

    def __repr__(self) -> str:
        if not self.operations:
            return 'cirq.Moment()'

        block = '\n'.join([repr(op) + ',' for op in self.operations])
        indented = '    ' + '\n    '.join(block.split('\n'))

        return f'cirq.Moment(\n{indented}\n)'

    def __str__(self) -> str:
        return ' and '.join(str(op) for op in self.operations)

    def transform_qubits(self: TSelf_Moment,
                         func: Callable[['cirq.Qid'], 'cirq.Qid']
                        ) -> TSelf_Moment:
        """Returns the same moment, but with different qubits.

        Args:
            func: The function to use to turn each current qubit into a desired
                new qubit.

        Returns:
            The receiving moment but with qubits transformed by the given
                function.
        """
        return self.__class__(op.transform_qubits(func)
                for op in self.operations)

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['operations'])

    @classmethod
    def _from_json_dict_(cls, operations, **kwargs):
        return Moment(operations)

    def __add__(self,
                other: Union['cirq.Operation', 'cirq.Moment']) -> 'cirq.Moment':
        if isinstance(other, raw_types.Operation):
            return self.with_operation(other)
        if isinstance(other, Moment):
            return Moment(self.operations + other.operations)
        return NotImplemented

    # pylint: disable=function-redefined
    @overload
    def __getitem__(self, key: raw_types.Qid) -> 'cirq.Operation':
        pass

    @overload
    def __getitem__(self, key: Iterable[raw_types.Qid]) -> 'cirq.Moment':
        pass

    def __getitem__(self, key):
        if isinstance(key, raw_types.Qid):
            return self._operation_touching(key)
        elif isinstance(key, Iterable):
            qubits_to_keep = frozenset(key)
            ops_to_keep = tuple(
                op for op in self.operations
                if not qubits_to_keep.isdisjoint(frozenset(op.qubits)))
            return Moment(ops_to_keep)
