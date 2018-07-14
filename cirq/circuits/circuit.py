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

from typing import (
    List, Any, Dict, FrozenSet, Callable, Iterable, Iterator,
    Optional, Sequence, Union, Type, Tuple, cast, TypeVar
)

import numpy as np

from cirq import ops
from cirq.circuits.insert_strategy import InsertStrategy
from cirq.circuits.moment import Moment
from cirq.circuits.text_diagram_drawer import TextDiagramDrawer
from cirq.extension import Extensions
from cirq.ops import QubitId


T_DESIRED_GATE_TYPE = TypeVar('T_DESIRED_GATE_TYPE', bound='ops.Gate')


class Circuit(object):
    """A mutable list of groups of operations to apply to some qubits.

    Methods returning information about the circuit:
        next_moment_operating_on
        prev_moment_operating_on
        operation_at
        qubits
        findall_operations
        to_unitary_matrix
        to_text_diagram
        to_text_diagram_drawer

    Methods for mutation:
        insert
        append
        insert_into_range
        clear_operations_touching

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

    Attributes:
        moments: A list of the Moments of the circuit.
    """

    def __init__(self, moments: Iterable[Moment] = ()) -> None:
        """Initializes a circuit.

        Args:
            moments: The initial list of moments defining the circuit.
        """
        self.moments = list(moments)

    @staticmethod
    def from_ops(*operations: ops.OP_TREE,
                 strategy: InsertStrategy = InsertStrategy.NEW_THEN_INLINE
                 ) -> 'Circuit':
        """Creates an empty circuit and appends the given operations.

        Args:
            operations: The operations to append to the new circuit.
            strategy: How to append the operations.

        Returns:
            The constructed circuit containing the operations.
        """
        result = Circuit()
        result.append(operations, strategy)
        return result

    def __copy__(self):
        return type(self)(self.moments)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.moments == other.moments

    def __ne__(self, other):
        return not self == other

    def __getitem__(self, key: Union[int, slice]) -> Union['Circuit', Moment]:
        if isinstance(key, slice):
            return Circuit(self.moments[key])
        if isinstance(key, int):
            return self.moments[key]
        else:
            raise TypeError(
                '__getitem__ called with key not of type slice or int.')

    def __iadd__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        self.moments += other.moments
        return self

    def __add__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return Circuit(self.moments + other.moments)

    def __imul__(self, repetitions: int):
        if not isinstance(repetitions, int):
            return NotImplemented
        self.moments *= repetitions
        return self

    def __mul__(self, repetitions: int):
        if not isinstance(repetitions, int):
            return NotImplemented
        return Circuit(self.moments * repetitions)

    def __rmul__(self, repetitions: int):
        if not isinstance(repetitions, int):
            return NotImplemented
        return self * repetitions

    def __len__(self):
        return len(self.moments)

    def __iter__(self):
        return iter(self.moments)

    def __repr__(self):
        moment_lines = ('\n    ' + repr(moment) for moment in self.moments)
        return 'Circuit([{}])'.format(','.join(moment_lines))

    def __str__(self):
        return self.to_text_diagram()

    __hash__ = None  # type: ignore

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

    def _first_moment_operating_on(self, qubits: Iterable[ops.QubitId],
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
        max_circuit_distance = len(self.moments) - start_moment_index
        if max_distance is None:
            max_distance = max_circuit_distance
        elif max_distance < 0:
            raise ValueError('Negative max_distance: {}'.format(max_distance))
        else:
            max_distance = min(max_distance, max_circuit_distance)

        return self._first_moment_operating_on(
            qubits,
            range(start_moment_index, start_moment_index + max_distance))

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
            end_moment_index = len(self.moments)

        if max_distance is None:
            max_distance = len(self.moments)
        elif max_distance < 0:
            raise ValueError('Negative max_distance: {}'.format(max_distance))
        else:
            max_distance = min(end_moment_index, max_distance)

        # Don't bother searching indices past the end of the list.
        if end_moment_index > len(self.moments):
            d = end_moment_index - len(self.moments)
            end_moment_index -= d
            max_distance -= d
        if max_distance <= 0:
            return None

        return self._first_moment_operating_on(qubits,
                                               (end_moment_index - k - 1
                                                for k in range(max_distance)))

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
        if not 0 <= moment_index < len(self.moments):
            return None
        for op in self.moments[moment_index].operations:
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
        for index, moment in enumerate(self.moments):
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
            gate_type: The type of gate to find, e.g. RotXGate or
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
            self.moments.insert(splitter_index, Moment())
            return splitter_index

        if strategy is InsertStrategy.INLINE:
            if (not self._has_op_at(splitter_index - 1, op.qubits) and
                    0 <= splitter_index - 1 < len(self.moments)):
                return splitter_index - 1

            return self._pick_or_create_inserted_op_moment_index(
                splitter_index, op, InsertStrategy.NEW)

        if strategy is InsertStrategy.EARLIEST:
            if not self._has_op_at(splitter_index, op.qubits):
                p = self.prev_moment_operating_on(op.qubits, splitter_index)
                return p + 1 if p is not None else 0

            return self._pick_or_create_inserted_op_moment_index(
                splitter_index, op, InsertStrategy.INLINE)

        raise ValueError('Unrecognized append strategy: {}'.format(strategy))

    def _has_op_at(self, moment_index, qubits):
        return (0 <= moment_index < len(self.moments) and
                self.moments[moment_index].operates_on(qubits))

    def insert(
            self,
            index: int,
            moment_or_operation_tree: Union[Moment, ops.OP_TREE],
            strategy: InsertStrategy = InsertStrategy.NEW_THEN_INLINE) -> int:
        """Inserts operations into the middle of the circuit.

        Args:
            index: The index to insert all of the operations at.
            moment_or_operation_tree: An operation or tree of operations.
            strategy: How to pick/create the moment to put operations into.

        Returns:
            The insertion index that will place operations just after the
            operations that were inserted by this method.

        Raises:
            IndexError: Bad insertion index.
            ValueError: Bad insertion strategy.
        """
        if isinstance(moment_or_operation_tree, Moment):
            self.moments.insert(index, moment_or_operation_tree)
            return index + 1

        if not 0 <= index <= len(self.moments):
            raise IndexError('Insert index out of range: {}'.format(index))

        k = index
        for op in ops.flatten_op_tree(moment_or_operation_tree):
            p = self._pick_or_create_inserted_op_moment_index(k, op, strategy)
            while p >= len(self.moments):
                self.moments.append(Moment())
            self.moments[p] = self.moments[p].with_operation(op)
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
        if not 0 <= start <= end <= len(self.moments):
            raise IndexError('Bad insert indices: [{}, {})'.format(
                start, end))

        operations = list(ops.flatten_op_tree(operations))
        i = start
        op_index = 0
        while op_index < len(operations):
            op = operations[op_index]
            while i < end and self.moments[i].operates_on(op.qubits):
                i += 1
            if i >= end:
                break
            self.moments[i] = self.moments[i].with_operation(op)
            op_index += 1

        if op_index >= len(operations):
            return end

        return self.insert(end, operations[op_index:])

    def append(
            self,
            moment_or_operation_tree: Union[Moment, ops.OP_TREE],
            strategy: InsertStrategy = InsertStrategy.NEW_THEN_INLINE):
        """Appends operations onto the end of the circuit.

        Args:
            moment_or_operation_tree: An operation or tree of operations.
            strategy: How to pick/create the moment to put operations into.
        """
        self.insert(len(self.moments), moment_or_operation_tree, strategy)

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
            if 0 <= k < len(self.moments):
                self.moments[k] = self.moments[k].without_operations_touching(
                    qubits)

    def all_qubits(self) -> FrozenSet[QubitId]:
        """Returns the qubits acted upon by Operations in this circuit."""
        return frozenset(q for m in self.moments for q in m.qubits)

    def all_operations(self) -> Iterator[ops.Operation]:
        """Iterates over the operations applied by this circuit.

        Operations from earlier moments will be iterated over first. Operations
        within a moment are iterated in the order they were given to the
        moment's constructor.
        """
        return (op for moment in self for op in moment.operations)

    def to_unitary_matrix(
            self,
            qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
            qubits_that_should_be_present: Iterable[QubitId] = (),
            ignore_terminal_measurements: bool = True,
            ext: Extensions = None) -> np.ndarray:
        """Converts the circuit into a unitary matrix, if possible.

        Args:
            qubit_order: Determines how qubits are ordered when passing matrices
                into np.kron.
            ext: The extensions to use when attempting to cast operations into
                KnownMatrix instances.
            qubits_that_should_be_present: Qubits that may or may not appear
                in operations within the circuit, but that should be included
                regardless when generating the matrix.
            ignore_terminal_measurements: When set, measurements at the end of
                the circuit are ignored instead of causing the method to
                fail.

        Returns:
            A (possibly gigantic) 2d numpy array corresponding to a matrix
            equivalent to the circuit's effect on a quantum state.

        Raises:
            ValueError: The circuit contains measurement gates that are not
                ignored.
            TypeError: The circuit contains gates that don't have a known
                unitary matrix, e.g. gates parameterized by a Symbol.
        """

        if ext is None:
            ext = Extensions()

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

        result = _apply_unitary_circuit(self, state, qs, ext)
        return result.reshape((1 << n, 1 << n))

    def to_unitary_output(
            self,
            initial_state: Union[int, np.ndarray] = 0,
            qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
            qubits_that_should_be_present: Iterable[QubitId] = (),
            ignore_terminal_measurements: bool = True,
            ext: Extensions = None) -> np.ndarray:
        """Computes the output state vector of the circuit.

        Args:
            qubit_order: Determines how qubits are ordered when passing matrices
                into np.kron.
            initial_state: The input state for the circuit. This can be an int
                or a vector. When this is an int, it refers to a computational
                basis state (e.g. 5 means initialize to |5> = |...000101>). If
                this is a state vector, it specifies the initial state. The
                vector must be a flat numpy array with a type that can be
                converted to np.complex128.
            qubits_that_should_be_present: Qubits that may or may not appear
                in operations within the circuit, but that should be included
                regardless when generating the matrix.
            ignore_terminal_measurements: When set, measurements at the end of
                the circuit are ignored instead of causing the method to
                fail.
            ext: The extensions to use when attempting to cast operations into
                KnownMatrix instances.

        Returns:
            A (possibly gigantic) numpy array storing the superposition that
            came out of the circuit for the given input state.

        Raises:
            ValueError: The circuit contains measurement gates that are not
                ignored.
            TypeError: The circuit contains gates that don't have a known
                unitary matrix, e.g. gates parameterized by a Symbol.
        """

        if ext is None:
            ext = Extensions()

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
            state = np.zeros(1 << n, dtype=np.complex128)
            state[initial_state] = 1
        else:
            state = initial_state.astype(np.complex128)
        state.shape = (2,) * n

        result = _apply_unitary_circuit(self, state, qs, ext)
        return result.reshape((1 << n,))

    def to_text_diagram(
            self,
            ext: Extensions = None,
            use_unicode_characters: bool = True,
            transpose: bool = False,
            precision: Optional[int] = 3,
            qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT) -> str:
        """Returns text containing a diagram describing the circuit.

        Args:
            ext: For extending operations/gates to implement TextDiagrammable.
            use_unicode_characters: Determines if unicode characters are
                allowed (as opposed to ascii-only diagrams).
            transpose: Arranges qubit wires vertically instead of horizontally.
            precision: Number of digits to display in text diagram
            qubit_order: Determines how qubits are ordered in the diagram.

        Returns:
            The text diagram.
        """
        diagram = self.to_text_diagram_drawer(
            ext=ext,
            use_unicode_characters=use_unicode_characters,
            qubit_name_suffix='' if transpose else ': ',
            precision=precision,
            qubit_order=qubit_order)

        if transpose:
            return diagram.transpose().render(
                crossing_char='┼' if use_unicode_characters else '-',
                use_unicode_characters=use_unicode_characters)
        return diagram.render(
            crossing_char='┼' if use_unicode_characters else '|',
            horizontal_spacing=3,
            use_unicode_characters=use_unicode_characters)

    def to_text_diagram_drawer(
            self,
            ext: Extensions = None,
            use_unicode_characters: bool = True,
            qubit_name_suffix: str = '',
            precision: Optional[int] = 3,
            qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
    ) -> TextDiagramDrawer:
        """Returns a TextDiagramDrawer with the circuit drawn into it.

        Args:
            ext: For extending operations/gates to implement TextDiagrammable.
            use_unicode_characters: Determines if unicode characters are
                allowed (as opposed to ascii-only diagrams).
            qubit_name_suffix: Appended to qubit names in the diagram.
            precision: Number of digits to use when representing numbers.
            qubit_order: Determines how qubits are ordered in the diagram.

        Returns:
            The TextDiagramDrawer instance.
        """
        if ext is None:
            ext = Extensions()

        qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(
            self.all_qubits())
        qubit_map = {qubits[i]: i for i in range(len(qubits))}

        diagram = TextDiagramDrawer()
        for q, i in qubit_map.items():
            diagram.write(0, i, str(q) + qubit_name_suffix)

        for moment in [Moment()] * 2 + self.moments + [Moment()]:
            _draw_moment_in_diagram(moment,
                                    ext,
                                    use_unicode_characters,
                                    qubit_map,
                                    diagram,
                                    precision)

        w = diagram.width()
        for i in qubit_map.values():
            diagram.horizontal_line(i, 0, w)

        return diagram


def _get_operation_text_diagram_info_with_fallback(
        op: ops.Operation,
        args: ops.TextDiagramInfoArgs,
        ext: Extensions) -> ops.TextDiagramInfo:
    text_diagrammable_op = ext.try_cast(ops.TextDiagrammable, op)
    if text_diagrammable_op is not None:
        info = text_diagrammable_op.text_diagram_info(args)
        if len(op.qubits) != len(info.wire_symbols):
            raise ValueError(
                'Wanted diagram info from {!r} for {} '
                'qubits but got {!r}'.format(
                    op,
                    len(info.wire_symbols),
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
    if len(op.qubits) != 1:
        symbols = tuple('{}:{}'.format(name, i)
                        for i in range(len(op.qubits)))
    else:
        symbols = (name,)

    return ops.TextDiagramInfo(wire_symbols=symbols)


def _formatted_exponent(info: ops.TextDiagramInfo,
                        args: ops.TextDiagramInfoArgs) -> Optional[str]:
    # 1 is not shown.
    if info.exponent == 1:
        return None

    # Round -1.0 into -1.
    if info.exponent == -1:
        return '-1'

    # If it's a float, show the desired precision.
    if isinstance(info.exponent, float):
        if args.precision is not None:
            return '{{:.{}}}'.format(args.precision).format(info.exponent)
        return repr(info.exponent)

    # If the exponent is any other object, use its string representation.
    s = str(info.exponent)
    if '+' in s or ' ' in s or '-' in s[1:]:
        # The string has confusing characters. Put parens around it.
        return '({})'.format(info.exponent)
    return s


def _draw_moment_in_diagram(moment: Moment,
                            ext: Extensions,
                            use_unicode_characters: bool,
                            qubit_map: Dict[QubitId, int],
                            out_diagram: TextDiagramDrawer,
                            precision: Optional[int]):
    if not moment.operations:
        return []

    x0 = out_diagram.width()
    for op in moment.operations:
        indices = [qubit_map[q] for q in op.qubits]
        y1 = min(indices)
        y2 = max(indices)

        # Find an available column.
        x = x0
        while any(out_diagram.content_present(x, y)
                  for y in range(y1, y2 + 1)):
            x += 1

        # Draw vertical line linking the gate's qubits.
        if y2 > y1:
            out_diagram.vertical_line(x, y1, y2)

        args = ops.TextDiagramInfoArgs(
            known_qubits=op.qubits,
            known_qubit_count=len(op.qubits),
            use_unicode_characters=use_unicode_characters,
            precision=precision)
        info = _get_operation_text_diagram_info_with_fallback(op, args, ext)

        # Print gate qubit labels.
        for s, q in zip(info.wire_symbols, op.qubits):
            out_diagram.write(x, qubit_map[q], s)

        # Add an exponent to the last label.
        exponent = _formatted_exponent(info, args)
        if exponent is not None:
            out_diagram.write(x, y2, '^' + exponent)


def _apply_unitary_circuit(circuit: Circuit,
                           state: np.ndarray,
                           qubits: Tuple[ops.QubitId, ...],
                           ext: Extensions) -> np.ndarray:
    """Applies a circuit to the given vector or matrix.

    Args:
        circuit: The circuit to simulate. All operations must have a known
            matrix or decompositions leading to known matrices. Measurements
            are allowed to be in the circuit, but they will be ignored.
        state: The initial state tensor (i.e. superposition or unitary matrix).
            This is what will be left-multiplied by the circuit's effective
            unitary. If this is a state vector, it must have shape
            (2,)*num_qubits. If it is a unitary matrix it should have shape
            (2,) * (2*num_qubits).
        qubits: The qubits in the state tensor. Determines which axes operations
            apply to. An operation targeting the k'th qubit in this list will
            operate on the k'th axis of the state tensor.
        ext: Extensions used when attempting to get matrices and decompositions
            of the operations.

    Returns:
        The left-multiplied state tensor.
    """
    qubit_map = {q: i for i, q in enumerate(qubits)}
    buffer = np.zeros(state.shape, dtype=np.complex128)
    for op, qs in _extract_unitaries(circuit.all_operations(), ext):
        matrix = op.matrix().astype(np.complex128).reshape((2,) * (2 * len(qs)))
        indices = [qubit_map[q] for q in qs]
        _apply_unitary_operation(state, matrix, indices, buffer)
        state, buffer = buffer, state
    return state


def _extract_unitaries(operations: Iterable[ops.Operation],
                       ext: Extensions
                       ) -> Iterable[Tuple[ops.KnownMatrix,
                                           Tuple[ops.QubitId]]]:
    """Yields a sequence of unitary matrices equivalent to the circuit's effect.
    """
    for op in operations:
        # Check if the operation has a known matrix.
        known_matrix = ext.try_cast(ops.KnownMatrix, op)
        if known_matrix is not None:
            yield known_matrix, op.qubits
            continue

        # If not, check if it has a decomposition.
        composite_op = ext.try_cast(ops.CompositeOperation, op)
        if composite_op is not None:
            # Recurse decomposition to get known matrix gates.
            op_tree = composite_op.default_decompose()
            op_list = ops.flatten_op_tree(op_tree)
            for op2 in _extract_unitaries(op_list, ext):
                yield op2
            continue

        if ops.MeasurementGate.is_measurement(op):
            # Ignore measurement gates. The caller should have checked for this
            # and failed already.
            continue

        # Otherwise, fail
        raise TypeError(
            'Operation without a known matrix or decomposition: {!r}'
                .format(op))


def _apply_unitary_operation(state: np.ndarray,
                             matrix: np.ndarray,
                             target_axes: List[int],
                             out: Optional[np.ndarray]) -> np.ndarray:
    """Left-multiplies the given axes of the state tensor by the given matrix.

    Args:
        state: The state tensor to left-multiple.
        matrix: What to left-multiply the state tensor by.
        target_axes: Which axes of the tensor are being operated on.
        out: The buffer to store the results in. If None, a new buffer is used.

    Returns:
        The output tensor.
    """
    k = len(target_axes)
    d = len(state.shape)
    work_indices = tuple(range(k))
    data_indices = tuple(range(k, k + d))
    used_data_indices = tuple(data_indices[q] for q in target_axes)
    output_indices = list(data_indices)
    for w, t in zip(work_indices, target_axes):
        output_indices[t] = w

    return np.einsum(matrix, work_indices + used_data_indices,
                     state, data_indices,
                     output_indices,
                     out=out)
