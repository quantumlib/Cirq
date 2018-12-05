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

"""The circuit data structure.

Circuits consist of a list of Moments, each Moment made up of a set of
Operations. Each Operation is a Gate that acts on some Qubits, for a given
Moment the Operations must all act on distinct Qubits.
"""

from collections import defaultdict
from fractions import Fraction

from typing import (
    List, Any, Dict, FrozenSet, Callable, Iterable, Iterator, Optional,
    Sequence, Union, Type, Tuple, cast, TypeVar, overload, TYPE_CHECKING)

import numpy as np

from cirq import devices, ops, study, protocols
from cirq.circuits._bucket_priority_queue import BucketPriorityQueue
from cirq.circuits.insert_strategy import InsertStrategy
from cirq.circuits.moment import Moment
from cirq.circuits.text_diagram_drawer import TextDiagramDrawer
from cirq.circuits.qasm_output import QasmOutput
from cirq.type_workarounds import NotImplementedType
import cirq._version

if TYPE_CHECKING:
    # pylint: disable=unused-import
    from typing import Set


T_DESIRED_GATE_TYPE = TypeVar('T_DESIRED_GATE_TYPE', bound='ops.Gate')


class Circuit:
    """A mutable list of groups of operations to apply to some qubits.

    Methods returning information about the circuit:
        next_moment_operating_on
        prev_moment_operating_on
        next_moments_operating_on
        operation_at
        all_qubits
        all_operations
        findall_operations
        findall_operations_with_gate_type
        are_all_measurements_terminal
        to_unitary_matrix
        apply_unitary_effect_to_state
        to_text_diagram
        to_text_diagram_drawer

    Methods for mutation:
        insert
        append
        insert_into_range
        clear_operations_touching
        batch_insert
        batch_remove
        batch_insert_into
        insert_at_frontier

    Circuits can also be iterated over,
        for moment in circuit:
            ...
    and sliced,
        circuit[1:3] is a new Circuit made up of two moments, the first being
            circuit[1] and the second being circuit[2];
    and concatenated,
        circuit1 + circuit2 is a new Circuit made up of the moments in circuit1
            followed by the moments in circuit2;
    and multiplied by an integer,
        circuit * k is a new Circuit made up of the moments in circuit repeated
            k times.
    and mutated,
        circuit[1:7] = [Moment(...)]
    """

    def __init__(self,
                 moments: Iterable[Moment] = (),
                 device: devices.Device = devices.UnconstrainedDevice) -> None:
        """Initializes a circuit.

        Args:
            moments: The initial list of moments defining the circuit.
            device: Hardware that the circuit should be able to run on.
        """
        self._moments = list(moments)
        self._device = device
        self._device.validate_circuit(self)

    @property
    def device(self) -> devices.Device:
        return self._device

    @device.setter
    def device(self, new_device: devices.Device) -> None:
        new_device.validate_circuit(self)
        self._device = new_device

    @staticmethod
    def from_ops(*operations: ops.OP_TREE,
                 strategy: InsertStrategy = InsertStrategy.NEW_THEN_INLINE,
                 device: devices.Device = devices.UnconstrainedDevice
                 ) -> 'Circuit':
        """Creates an empty circuit and appends the given operations.

        Args:
            operations: The operations to append to the new circuit.
            strategy: How to append the operations.
            device: Hardware that the circuit should be able to run on.

        Returns:
            The constructed circuit containing the operations.
        """
        result = Circuit(device=device)
        result.append(operations, strategy)
        return result

    def __copy__(self) -> 'Circuit':
        return self.copy()

    def copy(self) -> 'Circuit':
        return Circuit(self._moments, self._device)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._moments == other._moments and self._device == other._device

    def __ne__(self, other):
        return not self == other

    def __len__(self):
        return len(self._moments)

    def __iter__(self):
        return iter(self._moments)

    def _decompose_(self) -> ops.OP_TREE:
        """See `cirq.SupportsDecompose`."""
        return self.all_operations()

    # pylint: disable=function-redefined
    @overload
    def __getitem__(self, key: slice) -> 'Circuit':
        pass

    @overload
    def __getitem__(self, key: int) -> Moment:
        pass

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Circuit(self._moments[key])
        if isinstance(key, int):
            return self._moments[key]
        else:
            raise TypeError(
                '__getitem__ called with key not of type slice or int.')

    @overload
    def __setitem__(self, key: int, value: Moment):
        pass

    @overload
    def __setitem__(self, key: slice, value: Iterable[Moment]):
        pass

    def __setitem__(self, key, value):
        if isinstance(key, int):
            if not isinstance(value, Moment):
                raise TypeError('Can only assign Moments into Circuits.')
            self._device.validate_moment(value)

        if isinstance(key, slice):
            value = list(value)
            if any(not isinstance(v, Moment) for v in value):
                raise TypeError('Can only assign Moments into Circuits.')
            for moment in value:
                self._device.validate_moment(moment)

        self._moments[key] = value
    # pylint: enable=function-redefined

    def __delitem__(self, key: Union[int, slice]):
        del self._moments[key]

    def __iadd__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        if (other.device != self._device and
                other.device != devices.UnconstrainedDevice):
            raise ValueError("Other circuit's device is not compatible.")
        for moment in other:
            self._device.validate_moment(moment)
        self._moments += other._moments
        return self

    def __add__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        device = (self._device
                  if other.device is devices.UnconstrainedDevice
                  else other.device)
        device_2 = (other.device
                    if self._device is devices.UnconstrainedDevice
                    else self._device)
        if device != device_2:
            raise ValueError("Can't add circuits with incompatible devices.")

        for moment in self:
            device.validate_moment(moment)
        for moment in other:
            device.validate_moment(moment)

        return Circuit(self._moments + other._moments,
                       device=device)

    def __imul__(self, repetitions: int):
        if not isinstance(repetitions, int):
            return NotImplemented
        self._moments *= repetitions
        return self

    def __mul__(self, repetitions: int):
        if not isinstance(repetitions, int):
            return NotImplemented
        return Circuit(self._moments * repetitions,
                       device=self._device)

    def __rmul__(self, repetitions: int):
        if not isinstance(repetitions, int):
            return NotImplemented
        return self * repetitions

    def __repr__(self):
        if not self._moments and self._device == devices.UnconstrainedDevice:
            return 'cirq.Circuit()'

        if not self._moments:
            return 'cirq.Circuit(device={!r})'.format(self._device)

        moment_str = _list_repr_with_indented_item_lines(self._moments)
        if self._device == devices.UnconstrainedDevice:
            return 'cirq.Circuit(moments={})'.format(moment_str)

        return 'cirq.Circuit(moments={}, device={!r})'.format(moment_str,
                                                              self._device)

    def __str__(self):
        return self.to_text_diagram()

    __hash__ = None  # type: ignore

    def with_device(
            self,
            new_device: devices.Device,
            qubit_mapping: Callable[[ops.QubitId], ops.QubitId] = lambda e: e,
    ) -> 'Circuit':
        """Maps the current circuit onto a new device, and validates.

        Args:
            new_device: The new device that the circuit should be on.
            qubit_mapping: How to translate qubits from the old device into
                qubits on the new device.

        Returns:
            The translated circuit.
        """
        return Circuit(
            moments=[Moment(operation.transform_qubits(qubit_mapping)
                            for operation in moment.operations)
                     for moment in self._moments],
            device=new_device
        )

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        """Print ASCII diagram in Jupyter."""
        if cycle:
            # There should never be a cycle.  This is just in case.
            p.text('Circuit(...)')
        else:
            p.text(self.to_text_diagram())

    def _repr_html_(self) -> str:
        """Print ASCII diagram in Jupyter notebook without wrapping lines."""
        return ('<pre style="overflow: auto; white-space: pre;">'
                + self.to_text_diagram()
                + '</pre>')

    def _first_moment_operating_on(self,
                                   qubits: Iterable[ops.QubitId],
                                   indices: Iterable[int]) -> Optional[int]:
        qubits = frozenset(qubits)
        for m in indices:
            if self._has_op_at(m, qubits):
                return m
        return None

    def next_moment_operating_on(self,
                                 qubits: Iterable[ops.QubitId],
                                 start_moment_index: int = 0,
                                 max_distance: int = None) -> Optional[int]:
        """Finds the index of the next moment that touches the given qubits.

        Args:
            qubits: We're looking for operations affecting any of these qubits.
            start_moment_index: The starting point of the search.
            max_distance: The number of moments (starting from the start index
                and moving forward) to check. Defaults to no limit.

        Returns:
            None if there is no matching moment, otherwise the index of the
            earliest matching moment.

        Raises:
          ValueError: negative max_distance.
        """
        max_circuit_distance = len(self._moments) - start_moment_index
        if max_distance is None:
            max_distance = max_circuit_distance
        elif max_distance < 0:
            raise ValueError('Negative max_distance: {}'.format(max_distance))
        else:
            max_distance = min(max_distance, max_circuit_distance)

        return self._first_moment_operating_on(
            qubits,
            range(start_moment_index, start_moment_index + max_distance))

    def next_moments_operating_on(self,
                                  qubits: Iterable[ops.QubitId],
                                  start_moment_index: int = 0
                                  ) -> Dict[ops.QubitId, int]:
        """Finds the index of the next moment that touches each qubit.

        Args:
            qubits: The qubits to find the next moments acting on.
            start_moment_index: The starting point of the search.

        Returns:
            The index of the next moment that touches each qubit. If there
            is no such moment, the next moment is specified as the number of
            moments in the circuit. Equivalently, can be characterized as one
            plus the index of the last moment after start_moment_index
            (inclusive) that does *not* act on a given qubit.
        """
        next_moments = {}
        for q in qubits:
            next_moment = self.next_moment_operating_on(
                [q], start_moment_index)
            next_moments[q] = (len(self._moments) if next_moment is None else
                               next_moment)
        return next_moments

    def prev_moment_operating_on(
            self,
            qubits: Sequence[ops.QubitId],
            end_moment_index: Optional[int] = None,
            max_distance: Optional[int] = None) -> Optional[int]:
        """Finds the index of the next moment that touches the given qubits.

        Args:
            qubits: We're looking for operations affecting any of these qubits.
            end_moment_index: The moment index just after the starting point of
                the reverse search. Defaults to the length of the list of
                moments.
            max_distance: The number of moments (starting just before from the
                end index and moving backward) to check. Defaults to no limit.

        Returns:
            None if there is no matching moment, otherwise the index of the
            latest matching moment.

        Raises:
            ValueError: negative max_distance.
        """
        if end_moment_index is None:
            end_moment_index = len(self._moments)

        if max_distance is None:
            max_distance = len(self._moments)
        elif max_distance < 0:
            raise ValueError('Negative max_distance: {}'.format(max_distance))
        else:
            max_distance = min(end_moment_index, max_distance)

        # Don't bother searching indices past the end of the list.
        if end_moment_index > len(self._moments):
            d = end_moment_index - len(self._moments)
            end_moment_index -= d
            max_distance -= d
        if max_distance <= 0:
            return None

        return self._first_moment_operating_on(qubits,
                                               (end_moment_index - k - 1
                                                for k in range(max_distance)))

    def _prev_moment_available(
            self,
            op: ops.Operation,
            end_moment_index: int) -> Optional[int]:
        last_available = end_moment_index
        k = end_moment_index
        while k > 0:
            k -= 1
            if not self._can_commute_past(k, op):
                return last_available
            if self._can_add_op_at(k, op):
                last_available = k
        return last_available

    def reachable_frontier_from(
            self,
            start_frontier: Dict[ops.QubitId, int],
            *,
            is_blocker: Callable[[ops.Operation], bool] = lambda op: False
    ) -> Dict[ops.QubitId, int]:
        """Determines how far can be reached into a circuit under certain rules.

        The location L = (qubit, moment_index) is *reachable* if and only if:

            a) L is one of the items in `start_frontier`.

            OR

            b) There is no operation at L and prev(L) = (qubit, moment_index-1)
                is reachable and L is within the bounds of the circuit.

            OR

            c) There is an operation P covering L and, for every location
                M = (q', moment_index) that P covers, the location
                prev(M) = (q', moment_index-1) is reachable. Also, P must not be
                classified as a blocker by the given `is_blocker` argument.

        In other words, the reachable region extends forward through time along
        each qubit until it hits a blocked operation or an operation that
        crosses into the set of not-involved-at-the-moment qubits.

        For each qubit q in `start_frontier`, the reachable locations will
        correspond to a contiguous range starting at start_frontier[q] and
        ending just before some index end_q. The result of this method is a
        dictionary, and that dictionary maps each qubit q to its end_q.

        Examples:

            If start_frontier is {
                cirq.LineQubit(0): 6,
                cirq.LineQubit(1): 2,
                cirq.LineQubit(2): 2,
            } then the reachable wire locations in the following circuit are
            highlighted with '█' characters:

                0   1   2   3   4   5   6   7   8   9   10  11  12  13
            0: ───H───@─────────────────█████████████████████─@───H───
                      │                                       │
            1: ───────@─██H███@██████████████████████─@───H───@───────
                              │                       │
            2: ─────────██████@███H██─@───────@───H───@───────────────
                                      │       │
            3: ───────────────────────@───H───@───────────────────────

            And the computed end_frontier is {
                cirq.LineQubit(0): 11,
                cirq.LineQubit(1): 9,
                cirq.LineQubit(2): 6,
            }

            Note that the frontier indices (shown above the circuit) are
            best thought of (and shown) as happening *between* moment indices.

            If we specify a blocker as follows:

                is_blocker=lambda: op == cirq.CZ(cirq.LineQubit(1),
                                                 cirq.LineQubit(2))

            and use this start_frontier:

                {
                    cirq.LineQubit(0): 0,
                    cirq.LineQubit(1): 0,
                    cirq.LineQubit(2): 0,
                    cirq.LineQubit(3): 0,
                }

            Then this is the reachable area:

                0   1   2   3   4   5   6   7   8   9   10  11  12  13
            0: ─██H███@██████████████████████████████████████─@───H───
                      │                                       │
            1: ─██████@███H██─@───────────────────────@───H───@───────
                              │                       │
            2: ─█████████████─@───H───@───────@───H───@───────────────
                                      │       │
            3: ─█████████████████████─@───H───@───────────────────────

            and the computed end_frontier is:

                {
                    cirq.LineQubit(0): 11,
                    cirq.LineQubit(1): 3,
                    cirq.LineQubit(2): 3,
                    cirq.LineQubit(3): 5,
                }

        Args:
            start_frontier: A starting set of reachable locations.
            is_blocker: A predicate that determines if operations block
                reachability. Any location covered by an operation that causes
                `is_blocker` to return True is considered to be an unreachable
                location.

        Returns:
            An end_frontier dictionary, containing an end index for each qubit q
            mapped to a start index by the given `start_frontier` dictionary.

            To determine if a location (q, i) was reachable, you can use
            this expression:

                q in start_frontier and start_frontier[q] <= i < end_frontier[q]

            where i is the moment index, q is the qubit, and end_frontier is the
            result of this method.
        """
        active = set()  # type: Set[ops.QubitId]
        end_frontier = {}
        queue = BucketPriorityQueue[ops.Operation](drop_duplicate_entries=True)

        def enqueue_next(qubit: ops.QubitId, moment: int) -> None:
            next_moment = self.next_moment_operating_on([qubit], moment)
            if next_moment is None:
                end_frontier[qubit] = max(len(self), start_frontier[qubit])
                if qubit in active:
                    active.remove(qubit)
            else:
                next_op = self.operation_at(qubit, next_moment)
                assert next_op is not None
                queue.enqueue(next_moment, next_op)

        for start_qubit, start_moment in start_frontier.items():
            enqueue_next(start_qubit, start_moment)

        while queue:
            cur_moment, cur_op = queue.dequeue()
            for q in cur_op.qubits:
                if (q in start_frontier and
                        cur_moment >= start_frontier[q] and
                        q not in end_frontier):
                    active.add(q)

            continue_past = (
                cur_op is not None and
                active.issuperset(cur_op.qubits) and
                not is_blocker(cur_op)
            )
            if continue_past:
                for q in cur_op.qubits:
                    enqueue_next(q, cur_moment + 1)
            else:
                for q in cur_op.qubits:
                    if q in active:
                        end_frontier[q] = cur_moment
                        active.remove(q)

        return end_frontier

    def findall_operations_between(self,
                                   start_frontier: Dict[ops.QubitId, int],
                                   end_frontier: Dict[ops.QubitId, int],
                                   omit_crossing_operations: bool = False
                                   ) -> List[Tuple[int, ops.Operation]]:
        """Finds operations between the two given frontiers.

        If a qubit is in `start_frontier` but not `end_frontier`, its end index
        defaults to the end of the circuit. If a qubit is in `end_frontier` but
        not `start_frontier`, its start index defaults to the start of the
        circuit. Operations on qubits not mentioned in either frontier are not
        included in the results.

        Args:
            start_frontier: Just before where to start searching for operations,
                for each qubit of interest. Start frontier indices are
                inclusive.
            end_frontier: Just before where to stop searching for operations,
                for each qubit of interest. End frontier indices are exclusive.
            omit_crossing_operations: Determines whether or not operations that
                cross from a location between the two frontiers to a location
                outside the two frontiers are included or excluded. (Operations
                completely inside are always included, and operations completely
                outside are always excluded.)

        Returns:
            A list of tuples. Each tuple describes an operation found between
            the two frontiers. The first item of each tuple is the index of the
            moment containing the operation, and the second item is the
            operation itself. The list is sorted so that the moment index
            increases monotonically.
        """
        result = BucketPriorityQueue[ops.Operation](
            drop_duplicate_entries=True)

        involved_qubits = set(start_frontier.keys()) | set(end_frontier.keys())
        # Note: only sorted to ensure a deterministic result ordering.
        for q in sorted(involved_qubits):
            for i in range(start_frontier.get(q, 0),
                           end_frontier.get(q, len(self))):
                op = self.operation_at(q, i)
                if op is None:
                    continue
                if (omit_crossing_operations and
                        not involved_qubits.issuperset(op.qubits)):
                    continue
                result.enqueue(i, op)

        return list(result)

    def operation_at(self,
                     qubit: ops.QubitId,
                     moment_index: int) -> Optional[ops.Operation]:
        """Finds the operation on a qubit within a moment, if any.

        Args:
            qubit: The qubit to check for an operation on.
            moment_index: The index of the moment to check for an operation
                within. Allowed to be beyond the end of the circuit.

        Returns:
            None if there is no operation on the qubit at the given moment, or
            else the operation.
        """
        if not 0 <= moment_index < len(self._moments):
            return None
        for op in self._moments[moment_index].operations:
            if qubit in op.qubits:
                return op
        return None

    def findall_operations(self, predicate: Callable[[ops.Operation], bool]
                           ) -> Iterable[Tuple[int, ops.Operation]]:
        """Find the locations of all operations that satisfy a given condition.

        This returns an iterator of (index, operation) tuples where each
        operation satisfies op_cond(operation) is truthy. The indices are
        in order of the moments and then order of the ops within that moment.

        Args:
            predicate: A method that takes an Operation and returns a Truthy
                value indicating the operation meets the find condition.

        Returns:
            An iterator (index, operation)'s that satisfy the op_condition.
        """
        for index, moment in enumerate(self._moments):
            for op in moment.operations:
                if predicate(op):
                    yield index, op

    def findall_operations_with_gate_type(
            self,
            gate_type: Type[T_DESIRED_GATE_TYPE]
    ) -> Iterable[Tuple[int,
                        ops.GateOperation,
                        T_DESIRED_GATE_TYPE]]:
        """Find the locations of all gate operations of a given type.

        Args:
            gate_type: The type of gate to find, e.g. XPowGate or
                MeasurementGate.

        Returns:
            An iterator (index, operation, gate)'s for operations with the given
            gate type.
        """
        result = self.findall_operations(
            lambda operation: (isinstance(operation, ops.GateOperation) and
                               isinstance(operation.gate, gate_type)))
        for index, op in result:
            gate_op = cast(ops.GateOperation, op)
            yield index, gate_op, cast(T_DESIRED_GATE_TYPE, gate_op.gate)

    def are_all_measurements_terminal(self):
        return all(
            self.next_moment_operating_on(op.qubits, i + 1) is None for (i, op)
            in self.findall_operations(ops.MeasurementGate.is_measurement))

    def _pick_or_create_inserted_op_moment_index(
            self, splitter_index: int, op: ops.Operation,
            strategy: InsertStrategy) -> int:
        """Determines and prepares where an insertion will occur.

        Args:
            splitter_index: The index to insert at.
            op: The operation that will be inserted.
            strategy: The insertion strategy.

        Returns:
            The index of the (possibly new) moment where the insertion should
                occur.

        Raises:
            ValueError: Unrecognized append strategy.
        """

        if (strategy is InsertStrategy.NEW or
                strategy is InsertStrategy.NEW_THEN_INLINE):
            self._moments.insert(splitter_index, Moment())
            return splitter_index

        if strategy is InsertStrategy.INLINE:
            if (0 <= splitter_index - 1 < len(self._moments) and
                    self._can_add_op_at(splitter_index - 1, op)):
                return splitter_index - 1

            return self._pick_or_create_inserted_op_moment_index(
                splitter_index, op, InsertStrategy.NEW)

        if strategy is InsertStrategy.EARLIEST:
            if self._can_add_op_at(splitter_index, op):
                p = self._prev_moment_available(op, splitter_index)
                return p or 0

            return self._pick_or_create_inserted_op_moment_index(
                splitter_index, op, InsertStrategy.INLINE)

        raise ValueError('Unrecognized append strategy: {}'.format(strategy))

    def _has_op_at(self,
                   moment_index: int,
                   qubits: Iterable[ops.QubitId]) -> bool:
        return (0 <= moment_index < len(self._moments) and
                self._moments[moment_index].operates_on(qubits))

    def _can_add_op_at(self,
                       moment_index: int,
                       operation: ops.Operation) -> bool:
        if not 0 <= moment_index < len(self._moments):
            return True
        return self._device.can_add_operation_into_moment(
            operation,
            self._moments[moment_index])

    def _can_commute_past(self,
                          moment_index: int,
                          operation: ops.Operation) -> bool:
        return not self._moments[moment_index].operates_on(operation.qubits)

    def insert(
            self,
            index: int,
            moment_or_operation_tree: Union[Moment, ops.OP_TREE],
            strategy: InsertStrategy = InsertStrategy.NEW_THEN_INLINE) -> int:
        """ Inserts a moment or operations into the circuit.
            Moments are inserted at the specified index.
            Operations are inserted into the moment specified by the index and
            'InsertStrategy'.

        Args:
            index: The index to insert all of the operations at.
            moment_or_operation_tree: An operation or tree of operations.
            strategy: How to pick/create the moment to put operations into.

        Returns:
            The insertion index that will place operations just after the
            operations that were inserted by this method.

        Raises:
            ValueError: Bad insertion strategy.
        """
        if isinstance(moment_or_operation_tree, Moment):
            self._device.validate_moment(moment_or_operation_tree)
            self._moments.insert(index, moment_or_operation_tree)
            return index + 1

        operations = list(ops.flatten_op_tree(ops.transform_op_tree(
            moment_or_operation_tree,
            self._device.decompose_operation)))
        for op in operations:
            self._device.validate_operation(op)

        # limit index to 0..len(self._moments), also deal with indices smaller 0
        k = max(min(index if index >= 0 else len(self._moments) + index,
                    len(self._moments)), 0)
        for op in operations:
            p = self._pick_or_create_inserted_op_moment_index(k, op, strategy)
            while p >= len(self._moments):
                self._moments.append(Moment())
            self._moments[p] = self._moments[p].with_operation(op)
            self._device.validate_moment(self._moments[p])
            k = max(k, p + 1)
            if strategy is InsertStrategy.NEW_THEN_INLINE:
                strategy = InsertStrategy.INLINE
        return k

    def insert_into_range(self,
                          operations: ops.OP_TREE,
                          start: int,
                          end: int) -> int:
        """Writes operations inline into an area of the circuit.

        Args:
            start: The start of the range (inclusive) to write the
                given operations into.
            end: The end of the range (exclusive) to write the given
                operations into. If there are still operations remaining,
                new moments are created to fit them.
            operations: An operation or tree of operations to insert.

        Returns:
            An insertion index that will place operations after the operations
            that were inserted by this method.

        Raises:
            IndexError: Bad inline_start and/or inline_end.
        """
        if not 0 <= start <= end <= len(self):
            raise IndexError('Bad insert indices: [{}, {})'.format(
                start, end))

        operations = list(ops.flatten_op_tree(operations))
        for op in operations:
            self._device.validate_operation(op)

        i = start
        op_index = 0
        while op_index < len(operations):
            op = operations[op_index]
            while i < end and not self._device.can_add_operation_into_moment(
                    op, self._moments[i]):
                i += 1
            if i >= end:
                break
            self._moments[i] = self._moments[i].with_operation(op)
            op_index += 1

        if op_index >= len(operations):
            return end

        return self.insert(end, operations[op_index:])

    @staticmethod
    def _pick_inserted_ops_moment_indices(operations: Sequence[ops.Operation],
                                          start: int=0,
                                          frontier: Dict[ops.QubitId, int]=None
                                          ) -> Tuple[Sequence[int],
                                                     Dict[ops.QubitId, int]]:
        """Greedily assigns operations to moments.

        Args:
            operations: The operations to assign to moments.
            start: The first moment to consider assignment to.
            frontier: The first moment to which an operation acting on a qubit
                can be assigned. Updated in place as operations are assigned.

        Returns:
            The frontier giving the index of the moment after the last one to
            which an operation that acts on each qubit is assigned. If a
            frontier was specified as an argument, this is the same object.
        """
        if frontier is None:
            frontier = defaultdict(lambda: 0)
        moment_indices = []
        for op in operations:
            op_start = max(start, max(frontier[q] for q in op.qubits))
            moment_indices.append(op_start)
            for q in op.qubits:
                frontier[q] = max(frontier[q], op_start + 1)

        return moment_indices, frontier

    def _push_frontier(self,
                       early_frontier: Dict[ops.QubitId, int],
                       late_frontier: Dict[ops.QubitId, int],
                       update_qubits: Iterable[ops.QubitId]=None
                       ) -> Tuple[int, int]:
        """Inserts moments to separate two frontiers.

        After insertion n_new moments, the following holds:
           for q in late_frontier:
               early_frontier[q] <= late_frontier[q] + n_new
           for q in update_qubits:
               early_frontier[q] the identifies the same moment as before
                   (but whose index may have changed if this moment is after
                   those inserted).

        Args:
            early_frontier: The earlier frontier. For qubits not in the later
                frontier, this is updated to account for the newly inserted
                moments.
            late_frontier: The later frontier. This is not modified.
            update_qubits: The qubits for which to update early_frontier to
                account for the newly inserted moments.

        Returns:
            (index at which new moments were inserted, how many new moments
            were inserted) if new moments were indeed inserted. (0, 0)
            otherwise.
        """
        if update_qubits is None:
            update_qubits = set(early_frontier).difference(late_frontier)
        n_new_moments = (max(early_frontier.get(q, 0) - late_frontier[q]
                             for q in late_frontier)
                         if late_frontier else 0)
        if n_new_moments > 0:
            insert_index = min(late_frontier.values())
            self._moments[insert_index:insert_index] = (
                [Moment()] * n_new_moments)
            for q in update_qubits:
                if early_frontier.get(q, 0) > insert_index:
                    early_frontier[q] += n_new_moments
            return insert_index, n_new_moments
        return (0, 0)

    def _insert_operations(self,
                           operations: Sequence[ops.Operation],
                           insertion_indices: Sequence[int]) -> None:
        """Inserts operations at the specified moments. Appends new moments if
        necessary.

        Args:
            operations: The operations to insert.
            insertion_indices: Where to insert them, i.e. operations[i] is
                inserted into moments[insertion_indices[i].

        Raises:
            ValueError: operations and insert_indices have different lengths.

        NB: It's on the caller to ensure that the operations won't conflict
        with operations already in the moment or even each other.
        """
        if len(operations) != len(insertion_indices):
            raise ValueError('operations and insertion_indices must have the'
                             'same length.')
        self._moments += [Moment() for _ in range(1 + max(insertion_indices) -
                                                  len(self))]
        moment_to_ops = defaultdict(list
                                    )  # type: Dict[int, List[ops.Operation]]
        for op_index, moment_index in enumerate(insertion_indices):
            moment_to_ops[moment_index].append(operations[op_index])
        for moment_index, new_ops in moment_to_ops.items():
            self._moments[moment_index] = Moment(
                self._moments[moment_index].operations + tuple(new_ops))

    def insert_at_frontier(self,
                           operations: ops.OP_TREE,
                           start: int,
                           frontier: Dict[ops.QubitId, int]=None
                           ) -> Dict[ops.QubitId, int]:
        """Inserts operations inline at frontier.

        Args:
            operations: the operations to insert
            start: the moment at which to start inserting the operations
            frontier: frontier[q] is the earliest moment in which an operation
                acting on qubit q can be placed.
        """
        if frontier is None:
            frontier = defaultdict(lambda: 0)
        operations = tuple(ops.flatten_op_tree(operations))
        if not operations:
            return frontier
        qubits = set(q for op in operations for q in op.qubits)
        if any(frontier[q] > start for q in qubits):
            raise ValueError('The frontier for qubits on which the operations'
                             'to insert act cannot be after start.')

        next_moments = self.next_moments_operating_on(qubits, start)

        insertion_indices, _ = self._pick_inserted_ops_moment_indices(
            operations, start, frontier)

        self._push_frontier(frontier, next_moments)

        self._insert_operations(operations, insertion_indices)

        return frontier

    def batch_remove(self,
                     removals: Iterable[Tuple[int, ops.Operation]]) -> None:
        """Removes several operations from a circuit.

        Args:
            removals: A sequence of (moment_index, operation) tuples indicating
                operations to delete from the moments that are present. All
                listed operations must actually be present or the edit will
                fail (without making any changes to the circuit).

        ValueError:
            One of the operations to delete wasn't present to start with.

        IndexError:
            Deleted from a moment that doesn't exist.
        """
        copy = self.copy()
        for i, op in removals:
            if op not in copy._moments[i].operations:
                raise ValueError(
                    "Can't remove {} @ {} because it doesn't exist.".format(
                        op, i))
            copy._moments[i] = Moment(old_op
                                      for old_op in copy._moments[i].operations
                                      if op != old_op)
        self._device.validate_circuit(copy)
        self._moments = copy._moments

    def batch_insert_into(self,
                          insert_intos: Iterable[Tuple[int, ops.Operation]]
                          ) -> None:
        """Inserts operations into empty spaces in existing moments.

        If any of the insertions fails (due to colliding with an existing
        operation), this method fails without making any changes to the circuit.

        Args:
            insert_intos: A sequence of (moment_index, new_operation)
                pairs indicating a moment to add a new operation into.

        ValueError:
            One of the insertions collided with an existing operation.

        IndexError:
            Inserted into a moment index that doesn't exist.
        """
        copy = self.copy()
        for i, op in insert_intos:
            copy._moments[i] = copy._moments[i].with_operation(op)
        self._device.validate_circuit(copy)
        self._moments = copy._moments

    def batch_insert(self,
                     insertions: Iterable[Tuple[int, ops.OP_TREE]]) -> None:
        """Applies a batched insert operation to the circuit.

        Transparently handles the fact that earlier insertions may shift
        the index that later insertions should occur at. For example, if you
        insert an operation at index 2 and at index 4, but the insert at index 2
        causes a new moment to be created, then the insert at "4" will actually
        occur at index 5 to account for the shift from the new moment.

        All insertions are done with the strategy 'EARLIEST'.

        When multiple inserts occur at the same index, the gates from the later
        inserts end up before the gates from the earlier inserts (exactly as if
        you'd called list.insert several times with the same index: the later
        inserts shift the earliest inserts forward).

        Args:
            insertions: A sequence of (insert_index, operations) pairs
                indicating operations to add into the circuit at specific
                places.
        """
        # Work on a copy in case validation fails halfway through.
        copy = self.copy()
        shift = 0
        # Note: python `sorted` is guaranteed to be stable. This matters.
        insertions = sorted(insertions, key=lambda e: e[0])
        groups = _group_until_different(insertions,
                                        key=lambda e: e[0],
                                        value=lambda e: e[1])
        for i, group in groups:
            insert_index = i + shift
            next_index = copy.insert(insert_index,
                                     reversed(group),
                                     InsertStrategy.EARLIEST)
            if next_index > insert_index:
                shift += next_index - insert_index
        self._moments = copy._moments

    def append(
            self,
            moment_or_operation_tree: Union[Moment, ops.OP_TREE],
            strategy: InsertStrategy = InsertStrategy.NEW_THEN_INLINE):
        """Appends operations onto the end of the circuit.

        Args:
            moment_or_operation_tree: An operation or tree of operations.
            strategy: How to pick/create the moment to put operations into.
        """
        self.insert(len(self._moments), moment_or_operation_tree, strategy)

    def clear_operations_touching(self,
                                  qubits: Iterable[ops.QubitId],
                                  moment_indices: Iterable[int]):
        """Clears operations that are touching given qubits at given moments.

        Args:
            qubits: The qubits to check for operations on.
            moment_indices: The indices of moments to check for operations
                within.
        """
        qubits = frozenset(qubits)
        for k in moment_indices:
            if 0 <= k < len(self._moments):
                self._moments[k] = self._moments[k].without_operations_touching(
                    qubits)

    def all_qubits(self) -> FrozenSet[ops.QubitId]:
        """Returns the qubits acted upon by Operations in this circuit."""
        return frozenset(q for m in self._moments for q in m.qubits)

    def all_operations(self) -> Iterator[ops.Operation]:
        """Iterates over the operations applied by this circuit.

        Operations from earlier moments will be iterated over first. Operations
        within a moment are iterated in the order they were given to the
        moment's constructor.
        """
        return (op for moment in self for op in moment.operations)

    def _has_unitary_(self) -> bool:
        if not self.are_all_measurements_terminal():
            return False

        unitary_ops = protocols.decompose(
            self.all_operations(),
            keep=protocols.has_unitary,
            intercepting_decomposer=_decompose_measurement_inversions,
            on_stuck_raise=None)
        return all(protocols.has_unitary(e) for e in unitary_ops)

    def _unitary_(self) -> Union[np.ndarray, NotImplementedType]:
        """Converts the circuit into a unitary matrix, if possible.

        If the circuit contains any non-terminal measurements, the conversion
        into a unitary matrix fails (i.e. returns NotImplemented). Terminal
        measurements are ignored when computing the unitary matrix. The unitary
        matrix is the product of the unitary matrix of all operations in the
        circuit (after expanding them to apply to the whole system).
        """
        if not self._has_unitary_():
            return NotImplemented
        return self.to_unitary_matrix(ignore_terminal_measurements=True)

    def to_unitary_matrix(
            self,
            qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
            qubits_that_should_be_present: Iterable[ops.QubitId] = (),
            ignore_terminal_measurements: bool = True,
            dtype: Type[np.number] = np.complex128) -> np.ndarray:
        """Converts the circuit into a unitary matrix, if possible.

        Args:
            qubit_order: Determines how qubits are ordered when passing matrices
                into np.kron.
            qubits_that_should_be_present: Qubits that may or may not appear
                in operations within the circuit, but that should be included
                regardless when generating the matrix.
            ignore_terminal_measurements: When set, measurements at the end of
                the circuit are ignored instead of causing the method to
                fail.
            dtype: The numpy dtype for the returned unitary. Defaults to
                np.complex128. Specifying np.complex64 will run faster at the
                cost of precision. `dtype` must be a complex np.dtype, unless
                all operations in the circuit have unitary matrices with
                exclusively real coefficients (e.g. an H + TOFFOLI circuit).

        Returns:
            A (possibly gigantic) 2d numpy array corresponding to a matrix
            equivalent to the circuit's effect on a quantum state.

        Raises:
            ValueError: The circuit contains measurement gates that are not
                ignored.
            TypeError: The circuit contains gates that don't have a known
                unitary matrix, e.g. gates parameterized by a Symbol.
        """

        if not ignore_terminal_measurements and any(
                ops.MeasurementGate.is_measurement(op)
                for op in self.all_operations()):
            raise ValueError('Circuit contains a measurement.')

        if not self.are_all_measurements_terminal():
            raise ValueError('Circuit contains a non-terminal measurement.')

        qs = ops.QubitOrder.as_qubit_order(qubit_order).order_for(
            self.all_qubits().union(qubits_that_should_be_present))
        n = len(qs)

        state = np.eye(1 << n, dtype=np.complex128)
        state.shape = (2,) * (2 * n)

        result = _apply_unitary_circuit(self, state, qs, dtype)
        return result.reshape((1 << n, 1 << n))

    def apply_unitary_effect_to_state(
            self,
            initial_state: Union[int, np.ndarray] = 0,
            qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
            qubits_that_should_be_present: Iterable[ops.QubitId] = (),
            ignore_terminal_measurements: bool = True,
            dtype: Type[np.number] = np.complex128) -> np.ndarray:
        """Left-multiplies a state vector by the circuit's unitary effect.

        A circuit's "unitary effect" is the unitary matrix produced by
        multiplying together all of its gates' unitary matrices. A circuit
        with non-unitary gates (such as measurement or parameterized gates) does
        not have a well-defined unitary effect, and the method will fail if such
        operations are present.

        For convenience, terminal measurements are automatically ignored
        instead of causing a failure. Set the `ignore_terminal_measurements`
        argument to False to disable this behavior.

        This method is equivalent to left-multiplying the input state by
        `cirq.unitary(circuit)` but it's computed in a more efficient
        way.

        Args:
            initial_state: The input state for the circuit. This can be an int
                or a vector. When this is an int, it refers to a computational
                basis state (e.g. 5 means initialize to ``|5⟩ = |...000101⟩``).
                If this is a state vector, it directly specifies the initial
                state's amplitudes. The vector must be a flat numpy array with a
                type that can be converted to np.complex128.
            qubit_order: Determines how qubits are ordered when passing matrices
                into np.kron.
            qubits_that_should_be_present: Qubits that may or may not appear
                in operations within the circuit, but that should be included
                regardless when generating the matrix.
            ignore_terminal_measurements: When set, measurements at the end of
                the circuit are ignored instead of causing the method to
                fail.
            dtype: The numpy dtype for the returned unitary. Defaults to
                np.complex128. Specifying np.complex64 will run faster at the
                cost of precision. `dtype` must be a complex np.dtype, unless
                all operations in the circuit have unitary matrices with
                exclusively real coefficients (e.g. an H + TOFFOLI circuit).

        Returns:
            A (possibly gigantic) numpy array storing the superposition that
            came out of the circuit for the given input state.

        Raises:
            ValueError: The circuit contains measurement gates that are not
                ignored.
            TypeError: The circuit contains gates that don't have a known
                unitary matrix, e.g. gates parameterized by a Symbol.
        """

        if not ignore_terminal_measurements and any(
                ops.MeasurementGate.is_measurement(op)
                for op in self.all_operations()):
            raise ValueError('Circuit contains a measurement.')

        if not self.are_all_measurements_terminal():
            raise ValueError('Circuit contains a non-terminal measurement.')

        qs = ops.QubitOrder.as_qubit_order(qubit_order).order_for(
            self.all_qubits().union(qubits_that_should_be_present))
        n = len(qs)

        if isinstance(initial_state, int):
            state = np.zeros(1 << n, dtype=dtype)
            state[initial_state] = 1
        else:
            state = initial_state.astype(dtype)
        state.shape = (2,) * n

        result = _apply_unitary_circuit(self, state, qs, dtype)
        return result.reshape((1 << n,))

    def to_text_diagram(
            self,
            *,
            use_unicode_characters: bool = True,
            transpose: bool = False,
            precision: Optional[int] = 3,
            qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT) -> str:
        """Returns text containing a diagram describing the circuit.

        Args:
            use_unicode_characters: Determines if unicode characters are
                allowed (as opposed to ascii-only diagrams).
            transpose: Arranges qubit wires vertically instead of horizontally.
            precision: Number of digits to display in text diagram
            qubit_order: Determines how qubits are ordered in the diagram.

        Returns:
            The text diagram.
        """
        diagram = self.to_text_diagram_drawer(
            use_unicode_characters=use_unicode_characters,
            qubit_name_suffix='' if transpose else ': ',
            precision=precision,
            qubit_order=qubit_order,
            transpose=transpose)

        return diagram.render(
            crossing_char=(None
                           if use_unicode_characters
                           else ('-' if transpose else '|')),
            horizontal_spacing=1 if transpose else 3,
            use_unicode_characters=use_unicode_characters)

    def to_text_diagram_drawer(
            self,
            *,
            use_unicode_characters: bool = True,
            qubit_name_suffix: str = '',
            transpose: bool = False,
            precision: Optional[int] = 3,
            qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT
    ) -> TextDiagramDrawer:
        """Returns a TextDiagramDrawer with the circuit drawn into it.

        Args:
            use_unicode_characters: Determines if unicode characters are
                allowed (as opposed to ascii-only diagrams).
            qubit_name_suffix: Appended to qubit names in the diagram.
            transpose: Arranges qubit wires vertically instead of horizontally.
            precision: Number of digits to use when representing numbers.
            qubit_order: Determines how qubits are ordered in the diagram.

        Returns:
            The TextDiagramDrawer instance.
        """
        qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(
            self.all_qubits())
        qubit_map = {qubits[i]: i for i in range(len(qubits))}

        diagram = TextDiagramDrawer()
        for q, i in qubit_map.items():
            diagram.write(0, i, str(q) + qubit_name_suffix)

        moment_groups = []  # type: List[Tuple[int, int]]
        for moment in self._moments:
            _draw_moment_in_diagram(moment,
                                    use_unicode_characters,
                                    qubit_map,
                                    diagram,
                                    precision,
                                    moment_groups)

        w = diagram.width()
        for i in qubit_map.values():
            diagram.horizontal_line(i, 0, w)

        if moment_groups:
            _draw_moment_groups_in_diagram(moment_groups,
                                           use_unicode_characters,
                                           diagram,
                                           transpose)

        if transpose:
            diagram = diagram.transpose()

        return diagram

    def _is_parameterized_(self) -> bool:
        return any(protocols.is_parameterized(op)
                   for op in self.all_operations())

    def _resolve_parameters_(self,
                             param_resolver: study.ParamResolver) -> 'Circuit':
        resolved_moments = []
        for moment in self:
            resolved_operations = _resolve_operations(
                moment.operations,
                param_resolver)
            new_moment = Moment(resolved_operations)
            resolved_moments.append(new_moment)
        resolved_circuit = Circuit(resolved_moments)
        return resolved_circuit

    def _qasm_(self) -> str:
        return self.to_qasm()

    def _to_qasm_output(
            self,
            header: Optional[str] = None,
            precision: int = 10,
            qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
    ) -> QasmOutput:
        """Returns a QASM object equivalent to the circuit.

        Args:
            header: A multi-line string that is placed in a comment at the top
                of the QASM. Defaults to a cirq version specifier.
            precision: Number of digits to use when representing numbers.
            qubit_order: Determines how qubits are ordered in the QASM
                register.
        """
        if header is None:
            header = 'Generated from Cirq v{}'.format(
                cirq._version.__version__)
        qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(
            self.all_qubits())
        return QasmOutput(operations=self.all_operations(),
                          qubits=qubits,
                          header=header,
                          precision=precision,
                          version='2.0')

    def to_qasm(self,
                header: Optional[str] = None,
                precision: int = 10,
                qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
                ) -> str:
        """Returns QASM equivalent to the circuit.

        Args:
            header: A multi-line string that is placed in a comment at the top
                of the QASM. Defaults to a cirq version specifier.
            precision: Number of digits to use when representing numbers.
            qubit_order: Determines how qubits are ordered in the QASM
                register.
        """
        return str(self._to_qasm_output(header, precision, qubit_order))

    def save_qasm(self,
                  file_path: Union[str, bytes, int],
                  header: Optional[str] = None,
                  precision: int = 10,
                  qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
                  ) -> None:
        """Save a QASM file equivalent to the circuit.

        Args:
            file_path: The location of the file where the qasm will be written.
            header: A multi-line string that is placed in a comment at the top
                of the QASM. Defaults to a cirq version specifier.
            precision: Number of digits to use when representing numbers.
            qubit_order: Determines how qubits are ordered in the QASM
                register.
        """
        self._to_qasm_output(header, precision, qubit_order).save(file_path)


def _resolve_operations(
        operations: Iterable[ops.Operation],
        param_resolver: study.ParamResolver) -> List[ops.Operation]:
    resolved_operations = []  # type: List[ops.Operation]
    for op in operations:
        resolved_operations.append(protocols.resolve_parameters(
            op, param_resolver))
    return resolved_operations


def _get_operation_circuit_diagram_info_with_fallback(
        op: ops.Operation,
        args: protocols.CircuitDiagramInfoArgs) -> protocols.CircuitDiagramInfo:
    info = protocols.circuit_diagram_info(op, args, None)
    if info is not None:
        if len(op.qubits) != len(info.wire_symbols):
            raise ValueError(
                'Wanted diagram info from {!r} for {} '
                'qubits but got {!r}'.format(
                    op,
                    len(op.qubits),
                    info))
        return info

    # Fallback to a default representation using the operation's __str__.
    name = str(op)

    # Representation usually looks like 'gate(qubit1, qubit2, etc)'.
    # Try to cut off the qubit part, since that would be redundant information.
    redundant_tail = '({})'.format(', '.join(str(e) for e in op.qubits))
    if name.endswith(redundant_tail):
        name = name[:-len(redundant_tail)]

    # Include ordering in the qubit labels.
    symbols = (name,) + tuple('#{}'.format(i + 1)
                              for i in range(1, len(op.qubits)))

    return protocols.CircuitDiagramInfo(wire_symbols=symbols)


def _formatted_exponent(info: protocols.CircuitDiagramInfo,
                        args: protocols.CircuitDiagramInfoArgs
                        ) -> Optional[str]:
    if info.exponent == 0:
        return '0'

    # 1 is not shown.
    if info.exponent == 1:
        return None

    # Round -1.0 into -1.
    if info.exponent == -1:
        return '-1'

    # If it's a float, show the desired precision.
    if isinstance(info.exponent, float):
        if args.precision is not None:
            # funky behavior of fraction, cast to str in constructor helps.
            approx_frac = Fraction(info.exponent).limit_denominator(16)
            if approx_frac.denominator not in [2, 4, 5, 10]:
                if abs(float(approx_frac)
                       - info.exponent) < 10**-args.precision:
                    return '({})'.format(approx_frac)

            return '{{:.{}}}'.format(args.precision).format(info.exponent)
        return repr(info.exponent)

    # If the exponent is any other object, use its string representation.
    s = str(info.exponent)
    if '+' in s or ' ' in s or '-' in s[1:]:
        # The string has confusing characters. Put parens around it.
        return '({})'.format(info.exponent)
    return s


def _draw_moment_in_diagram(moment: Moment,
                            use_unicode_characters: bool,
                            qubit_map: Dict[ops.QubitId, int],
                            out_diagram: TextDiagramDrawer,
                            precision: Optional[int],
                            moment_groups: List[Tuple[int, int]]):
    x0 = out_diagram.width()
    for op in moment.operations:
        indices = [qubit_map[q] for q in op.qubits]
        y1 = min(indices)
        y2 = max(indices)

        # Find an available column.
        x = x0
        while any(out_diagram.content_present(x, y)
                  for y in range(y1, y2 + 1)):
            out_diagram.force_horizontal_padding_after(x, 0)
            x += 1

        args = protocols.CircuitDiagramInfoArgs(
            known_qubits=op.qubits,
            known_qubit_count=len(op.qubits),
            use_unicode_characters=use_unicode_characters,
            qubit_map=qubit_map,
            precision=precision)
        info = _get_operation_circuit_diagram_info_with_fallback(op, args)

        # Draw vertical line linking the gate's qubits.
        if y2 > y1 and info.connected:
            out_diagram.vertical_line(x, y1, y2)

        # Print gate qubit labels.
        for s, q in zip(info.wire_symbols, op.qubits):
            out_diagram.write(x, qubit_map[q], s)

        # Add an exponent to the last label.
        exponent = _formatted_exponent(info, args)
        if exponent is not None:
            out_diagram.write(x, y2, '^' + exponent)

    # Group together columns belonging to the same Moment.
    if moment.operations and x > x0:
        moment_groups.append((x0, x))


def _draw_moment_groups_in_diagram(moment_groups: List[Tuple[int, int]],
                                   use_unicode_characters: bool,
                                   out_diagram: TextDiagramDrawer,
                                   transpose: bool):
    out_diagram.insert_empty_rows(0)
    h = out_diagram.height()

    top_left = '┌' if use_unicode_characters else '/'
    top_right = '┐' if use_unicode_characters else '\\'
    bottom_left = '└' if use_unicode_characters else '\\'
    bottom_right = '┘' if use_unicode_characters else '/'

    # Insert columns starting from the back since the insertion
    # affects subsequent indices.
    for x1, x2 in reversed(moment_groups):
        out_diagram.insert_empty_columns(x2 + 1)
        out_diagram.force_horizontal_padding_after(x2, 0)

        out_diagram.write(x2 + 1, 0, top_right, bottom_left)
        out_diagram.write(x2 + 1, h, bottom_right, bottom_right)
        out_diagram.force_horizontal_padding_after(x2 + 1,
                                                   2 if not transpose else 0)

        for y in [0, h]:
            out_diagram.horizontal_line(y, x1, x2 + 1)

        out_diagram.insert_empty_columns(x1)
        out_diagram.force_horizontal_padding_after(x1, 0)
        out_diagram.write(x1, 0, top_left, top_left)
        out_diagram.write(x1, h, bottom_left, top_right)

        out_diagram.force_horizontal_padding_after(x1 - 1,
                                                   2 if not transpose else 0)

    if not transpose:
        out_diagram.force_vertical_padding_after(0, 0)
        out_diagram.force_vertical_padding_after(h - 1, 0)


def _apply_unitary_circuit(circuit: Circuit,
                           state: np.ndarray,
                           qubits: Tuple[ops.QubitId, ...],
                           dtype: Type[np.number]) -> np.ndarray:
    """Applies a circuit's unitary effect to the given vector or matrix.

    This method assumes that the caller wants to ignore measurements.

    Args:
        circuit: The circuit to simulate. All operations must have a known
            matrix or decompositions leading to known matrices. Measurements
            are allowed to be in the circuit, but they will be ignored.
        state: The initial state tensor (i.e. superposition or unitary matrix).
            This is what will be left-multiplied by the circuit's effective
            unitary. If this is a state vector, it must have shape
            (2,) * num_qubits. If it is a unitary matrix it should have shape
            (2,) * (2*num_qubits).
        qubits: The qubits in the state tensor. Determines which axes operations
            apply to. An operation targeting the k'th qubit in this list will
            operate on the k'th axis of the state tensor.
        dtype: The numpy dtype to use for applying the unitary. Must be a
            complex dtype.

    Returns:
        The left-multiplied state tensor.
    """
    qubit_map = {q: i for i, q in enumerate(qubits)}
    buffer = np.zeros(state.shape, dtype=dtype)

    def on_stuck(bad_op):
        return TypeError(
            'Operation without a known matrix or decomposition: {!r}'.format(
                bad_op))

    unitary_ops = protocols.decompose(
        circuit.all_operations(),
        keep=protocols.has_unitary,
        intercepting_decomposer=_decompose_measurement_inversions,
        on_stuck_raise=on_stuck)

    for op in unitary_ops:
        indices = [qubit_map[q] for q in op.qubits]
        result = protocols.apply_unitary(
            unitary_value=op,
            args=protocols.ApplyUnitaryArgs(state, buffer, indices))
        if result is buffer:
            buffer = state
        state = result
    return state


def _decompose_measurement_inversions(op: ops.Operation) -> ops.OP_TREE:
    if ops.MeasurementGate.is_measurement(op):
        gate = cast(ops.MeasurementGate, cast(ops.GateOperation, op).gate)
        return [ops.X(q)
                for q, b in zip(op.qubits, gate.invert_mask)
                if b]

    return NotImplemented


def _list_repr_with_indented_item_lines(items: Sequence[Any]) -> str:
    block = '\n'.join([repr(op) + ',' for op in items])
    indented = '    ' + '\n    '.join(block.split('\n'))
    return '[\n{}\n]'.format(indented)


TIn = TypeVar('TIn')
TOut = TypeVar('TOut')
TKey = TypeVar('TKey')


@overload
def _group_until_different(items: Iterable[TIn],
                           key: Callable[[TIn], TKey],
                           ) -> Iterable[Tuple[TKey, List[TIn]]]:
    pass


@overload
def _group_until_different(items: Iterable[TIn],
                           key: Callable[[TIn], TKey],
                           value: Callable[[TIn], TOut]
                           ) -> Iterable[Tuple[TKey, List[TOut]]]:
    pass


def _group_until_different(items: Iterable[TIn],
                           key: Callable[[TIn], TKey],
                           value=lambda e: e):
    """Groups runs of items that are identical according to a keying function.

    Args:
        items: The items to group.
        key: If two adjacent items produce the same output from this function,
            they will be grouped.
        value: Maps each item into a value to put in the group. Defaults to the
            item itself.

    Examples:
        _group_until_different(range(11), key=is_prime) yields
            (False, [0, 1])
            (True, [2, 3])
            (False, [4])
            (True, [5])
            (False, [6])
            (True, [7])
            (False, [8, 9, 10])

    Yields:
        Tuples containing the group key and item values.
    """
    prev_item_key = None
    cur_items = []  # type: List[Any]
    for item in items:
        item_key = key(item)
        if cur_items and item_key != prev_item_key:
            yield prev_item_key, cur_items
            cur_items = []
        cur_items.append(value(item))
        prev_item_key = item_key

    if cur_items:
        yield prev_item_key, cur_items
