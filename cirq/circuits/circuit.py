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

from typing import Any, Dict, FrozenSet, Callable, Generator, Iterable, Iterator
from typing import Optional, Sequence, Union, TYPE_CHECKING

import numpy as np

from cirq import ops
from cirq.circuits.insert_strategy import InsertStrategy
from cirq.circuits.moment import Moment
from cirq.circuits.text_diagram_drawer import TextDiagramDrawer
from cirq.extension import Extensions
from cirq.ops import QubitId

if TYPE_CHECKING:
    # pylint: disable=unused-import
    from typing import Set


class Circuit(object):
    """A mutable list of groups of operations to apply to some qubits.

    Methods returning information about the circuit:
        next_moment_operating_on
        prev_moment_operating_on
        operation_on
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

    def iter_ops(self) -> Iterator[ops.Operation]:
        return (op for moment in self for op in moment.operations)

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

    def findall_operations(self, predicate: Callable[[ops.Operation], bool]):
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

    def are_all_measurements_terminal(self):
        is_meas_gate = lambda op: isinstance(op.gate, ops.MeasurementGate)
        return all(
            self.next_moment_operating_on(op.qubits, i + 1) is None for (i, op)
            in self.findall_operations(is_meas_gate))


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
            operation_tree: ops.OP_TREE,
            strategy: InsertStrategy = InsertStrategy.NEW_THEN_INLINE) -> int:
        """Inserts operations into the middle of the circuit.

        Args:
            index: The index to insert all of the operations at.
            operation_tree: An operation or tree of operations.
            strategy: How to pick/create the moment to put operations into.

        Returns:
            The insertion index that will place operations just after the
            operations that were inserted by this method.

        Raises:
            IndexError: Bad insertion index.
            ValueError: Bad insertion strategy.
        """
        if not 0 <= index <= len(self.moments):
            raise IndexError('Insert index out of range: {}'.format(index))

        k = index
        for op in ops.flatten_op_tree(operation_tree):
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
        if not 0 <= start < end <= len(self.moments):
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
            operation_tree: ops.OP_TREE,
            strategy: InsertStrategy = InsertStrategy.NEW_THEN_INLINE):
        """Appends operations onto the end of the circuit.

        Args:
            operation_tree: An operation or tree of operations.
            strategy: How to pick/create the moment to put operations into.
        """
        self.insert(len(self.moments), operation_tree, strategy)

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

    def qubits(self) -> FrozenSet[QubitId]:
        """Returns the qubits acted upon by Operations in this circuit."""
        return frozenset(q for m in self.moments for q in m.qubits)

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
            ext: The extensions to use when attempting to cast gates into
                KnownMatrixGate instances.
            qubits_that_should_be_present: Qubits that may or may not appear
                in operations within the circuit, but that should be included
                regardless when generating the matrix.
            ignore_terminal_measurements: When set, measurements at the end of
                the circuit are ignored instead of causing the conversion to
                fail.

        Returns:
            A (possibly gigantic) 2d numpy array corresponding to a matrix
            equivalent to the circuit's effect on a quantum state.

        Raises:
            TypeError: The circuit contains gates that don't have a known
                unitary matrix, such as measurement gates, gates parameterized
                by a Symbol, etc.
        """

        if ext is None:
            ext = Extensions()
        qs = ops.QubitOrder.as_qubit_order(qubit_order).order_for(
            self.qubits().union(qubits_that_should_be_present))
        qubit_map = {i: q
                     for q, i in enumerate(qs)}  # type: Dict[QubitId, int]
        matrix_ops = _flatten_to_known_matrix_ops(self.iter_ops(), ext)
        if not self.are_all_measurements_terminal():
            raise TypeError('Circuit contains a non-terminal measurement')
        return _operations_to_unitary_matrix(matrix_ops,
                                             qubit_map,
                                             ignore_terminal_measurements,
                                             ext)

    def to_text_diagram(
            self,
            ext: Extensions = None,
            use_unicode_characters: bool = True,
            transpose: bool = False,
            precision: Optional[int] = 3,
            qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT) -> str:
        """Returns text containing a diagram describing the circuit.

        Args:
            ext: For extending gates to implement TextDiagrammableGate.
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
                crossing_char='─' if use_unicode_characters else '-',
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
            ext: For extending gates to implement TextDiagrammableGate.
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
            self.qubits())
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


def _get_operation_text_diagram_symbols(op: ops.Operation,
                                        ext: Extensions,
                                        use_unicode_characters: bool,
                                        precision: Optional[int]
                                        ) -> Iterable[str]:
    text_diagram_gate = ext.try_cast(ops.TextDiagrammableGate, op.gate)
    if text_diagram_gate is not None:
        wire_symbols = text_diagram_gate.text_diagram_wire_symbols(
            qubit_count=len(op.qubits),
            use_unicode_characters=use_unicode_characters,
            precision=precision)
        if len(op.qubits) == len(wire_symbols):
            return wire_symbols
        elif len(wire_symbols) == 1:
            return len(op.qubits) * wire_symbols
        else:
            raise ValueError(
                'Multi-qubit operation with TextDiagrammableGate {} that '
                'requires {} qubits but found {} qubits'.format(
                    repr(op.gate), len(wire_symbols), len(op.qubits)))

    name = repr(op.gate)
    if len(op.qubits) == 1:
        return [name]
    return ['{}:{}'.format(name, i) for i in range(len(op.qubits))]


def _get_operation_text_diagram_exponent(op: ops.Operation,
                                         ext: Extensions,
                                         precision: Optional[int]
                                         ) -> Optional[str]:
    text_diagram_gate = ext.try_cast(ops.TextDiagrammableGate, op.gate)
    if text_diagram_gate is None:
        return None
    exponent = text_diagram_gate.text_diagram_exponent()
    if exponent == 1:
        return None
    if exponent == -1:
        return '-1'
    if isinstance(exponent, float) and precision is not None:
        return '{{:.{}}}'.format(precision).format(exponent)
    s = str(exponent)
    if '+' in s or ' ' in s or '-' in s[1:]:
        return '({})'.format(exponent)
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

        # Print gate qubit labels.
        symbols = _get_operation_text_diagram_symbols(op,
                                                      ext,
                                                      use_unicode_characters,
                                                      precision)
        for s, q in zip(symbols, op.qubits):
            out_diagram.write(x, qubit_map[q], s)

        # Add an exponent to the last label.
        exponent = _get_operation_text_diagram_exponent(op, ext, precision)
        if exponent is not None:
            out_diagram.write(x, y2, '^' + exponent)


def _flatten_to_known_matrix_ops(iter_ops: Iterable[ops.Operation],
                                 ext: Extensions
                                 ) -> Generator[ops.Operation, None, None]:
    for op in iter_ops:
        # Check if the operation has a known matrix
        known_matrix_gate = ext.try_cast(ops.KnownMatrixGate, op.gate)
        if known_matrix_gate is not None:
            yield op
            continue

        # If not, check if it has a decomposition
        composite_gate = ext.try_cast(ops.CompositeGate, op.gate)
        if composite_gate is not None:
            # Recurse decomposition to get known matrix gates.
            op_tree = composite_gate.default_decompose(op.qubits)
            op_list = ops.flatten_op_tree(op_tree)
            for op in _flatten_to_known_matrix_ops(op_list, ext):
                yield op
            continue

        # Pass measurement gates through
        meas_gate = ext.try_cast(ops.MeasurementGate, op.gate)
        if meas_gate is not None:
            yield op
            continue

        # Otherwise, fail
        raise TypeError(
            'Operation without a known matrix or decomposition: {!r}'
            .format(op))


def _operations_to_unitary_matrix(iter_ops: Iterable[ops.Operation],
                                  qubit_map: Dict[QubitId, int],
                                  ignore_terminal_measurements: bool,
                                  ext: Extensions) -> np.ndarray:
    # Precondition is that circuit has only terminal measurements.
    total = np.eye(1 << len(qubit_map))
    for op in iter_ops:
        meas_gate = ext.try_cast(ops.MeasurementGate, op.gate)
        if meas_gate is not None:
            if not ignore_terminal_measurements:
                raise TypeError(
                    'Terminal measurement operation but not ignoring these '
                    'measurements: {!r}'.format(op))
            continue  # coverage: ignore
        mat = _operation_to_unitary_matrix(op, qubit_map, ext)
        total = np.matmul(mat, total)
    return total


def _operation_to_unitary_matrix(op: ops.Operation,
                                 qubit_map: Dict[QubitId, int],
                                 ext: Extensions) -> np.ndarray:
    known_matrix_gate = ext.try_cast(ops.KnownMatrixGate, op.gate)
    if known_matrix_gate is None:
        raise TypeError(
            'Operation without a known matrix: {!r}'.format(op))
    sub_mat = known_matrix_gate.matrix()
    qubit_count = len(qubit_map)
    bit_locs = [qubit_count - qubit_map[q] - 1 for q in op.qubits][::-1]
    over_mask = ~sum(1 << b for b in bit_locs)

    result = np.zeros(shape=(1 << qubit_count, 1 << qubit_count),
                      dtype=np.complex128)
    for i in range(1 << qubit_count):
        sub_i = sum(_moved_bit(i, b, k) for k, b in enumerate(bit_locs))
        over_i = i & over_mask

        for sub_j in range(sub_mat.shape[1]):
            j = sum(_moved_bit(sub_j, k, b) for k, b in enumerate(bit_locs))
            result[i, over_i | j] = sub_mat[sub_i, sub_j]

    return result


def _moved_bit(val: int, at: int, to: int) -> int:
    return ((val >> at) & 1) << to
