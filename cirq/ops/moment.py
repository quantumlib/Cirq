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

from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    overload,
    Optional,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
    Union,
)
from cirq import protocols, ops
from cirq._compat import deprecated_parameter
from cirq.ops import raw_types
from cirq.type_workarounds import NotImplementedType

if TYPE_CHECKING:
    import cirq

TSelf_Moment = TypeVar('TSelf_Moment', bound='Moment')


def _default_breakdown(qid: 'cirq.Qid') -> Tuple[Any, Any]:
    # Attempt to convert into a position on the complex plane.
    try:
        plane_pos = complex(qid)  # type: ignore
        return plane_pos.real, plane_pos.imag
    except TypeError:
        return None, qid


class Moment:
    """A time-slice of operations within a circuit.

    Grouping operations into moments is intended to be a strong suggestion to
    whatever is scheduling operations on real hardware. Operations in the same
    moment should execute at the same time (to the extent possible; not all
    operations have the same duration) and it is expected that all operations
    in a moment should be completed before beginning the next moment.

    Moment can be indexed by qubit or list of qubits:

    *   `moment[qubit]` returns the Operation in the moment which touches the
            given qubit, or throws KeyError if there is no such operation.
    *   `moment[qubits]` returns another Moment which consists only of those
            operations which touch at least one of the given qubits. If there
            are no such operations, returns an empty Moment.
    """

    @deprecated_parameter(
        deadline='v0.9',
        fix='Don\'t specify a keyword.',
        match=lambda _, kwargs: 'operations' in kwargs,
        parameter_desc='operations',
        rewrite=lambda args, kwargs: (args + (kwargs['operations'],), {}),
    )
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

        # An internal dictionary to support efficient operation access by qubit.
        self._qubit_to_op: Dict['cirq.Qid', 'cirq.Operation'] = {}
        for op in self.operations:
            for q in op.qubits:
                # Check that operations don't overlap.
                if q in self._qubit_to_op:
                    raise ValueError('Overlapping operations: {}'.format(self.operations))
                self._qubit_to_op[q] = op

        self._qubits = frozenset(self._qubit_to_op.keys())

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
        return qubit in self._qubit_to_op

    def operates_on(self, qubits: Iterable['cirq.Qid']) -> bool:
        """Determines if the moment has operations touching the given qubits.

        Args:
            qubits: The qubits that may or may not be touched by operations.

        Returns:
            Whether this moment has operations involving the qubits.
        """
        return bool(set(qubits) & self.qubits)

    def operation_at(self, qubit: raw_types.Qid) -> Optional['cirq.Operation']:
        """Returns the operation on a certain qubit for the moment.

        Args:
            qubit: The qubit on which the returned Operation operates
                on.

        Returns:
            The operation that operates on the qubit for that moment.
        """
        if self.operates_on([qubit]):
            return self.__getitem__(qubit)
        else:
            return None

    def with_operation(self, operation: 'cirq.Operation') -> 'cirq.Moment':
        """Returns an equal moment, but with the given op added.

        Args:
            operation: The operation to append.

        Returns:
            The new moment.
        """
        if any(q in self._qubits for q in operation.qubits):
            raise ValueError('Overlapping operations: {}'.format(operation))

        # Use private variables to facilitate a quick copy.
        m = Moment()
        m._operations = self._operations + (operation,)
        m._qubits = frozenset(self._qubits.union(set(operation.qubits)))
        m._qubit_to_op = self._qubit_to_op.copy()
        for q in operation.qubits:
            m._qubit_to_op[q] = operation

        return m

    def with_operations(self, *contents: 'cirq.OP_TREE') -> 'cirq.Moment':
        """Returns a new moment with the given contents added.

        Args:
            contents: New operations to add to this moment.

        Returns:
            The new moment.
        """
        from cirq.ops import op_tree

        operations = list(self._operations)
        qubits = set(self._qubits)
        for op in op_tree.flatten_to_ops(contents):
            if any(q in qubits for q in op.qubits):
                raise ValueError('Overlapping operations: {}'.format(op))
            operations.append(op)
            qubits.update(op.qubits)

        # Use private variables to facilitate a quick copy.
        m = Moment()
        m._operations = tuple(operations)
        m._qubits = frozenset(qubits)
        m._qubit_to_op = self._qubit_to_op.copy()
        for op in operations:
            for q in op.qubits:
                m._qubit_to_op[q] = op

        return m

    def without_operations_touching(self, qubits: Iterable['cirq.Qid']) -> 'cirq.Moment':
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
            operation
            for operation in self.operations
            if qubits.isdisjoint(frozenset(operation.qubits))
        )

    def _with_measurement_key_mapping_(self, key_map: Dict[str, str]):
        return Moment(
            protocols.with_measurement_key_mapping(op, key_map)
            if protocols.is_measurement(op)
            else op
            for op in self.operations
        )

    def __copy__(self):
        return type(self)(self.operations)

    def __bool__(self) -> bool:
        return bool(self.operations)

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented

        return sorted(self.operations, key=lambda op: op.qubits) == sorted(
            other.operations, key=lambda op: op.qubits
        )

    def _approx_eq_(self, other: Any, atol: Union[int, float]) -> bool:
        """See `cirq.protocols.SupportsApproximateEquality`."""
        if not isinstance(other, type(self)):
            return NotImplemented

        return protocols.approx_eq(
            sorted(self.operations, key=lambda op: op.qubits),
            sorted(other.operations, key=lambda op: op.qubits),
            atol=atol,
        )

    def __ne__(self, other) -> bool:
        return not self == other

    def __hash__(self):
        return hash((Moment, tuple(sorted(self.operations, key=lambda op: op.qubits))))

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
        return self.to_text_diagram()

    def transform_qubits(
        self: TSelf_Moment, func: Callable[['cirq.Qid'], 'cirq.Qid']
    ) -> TSelf_Moment:
        """Returns the same moment, but with different qubits.

        Args:
            func: The function to use to turn each current qubit into a desired
                new qubit.

        Returns:
            The receiving moment but with qubits transformed by the given
                function.
        """
        return self.__class__(op.transform_qubits(func) for op in self.operations)

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['operations'])

    @classmethod
    def _from_json_dict_(cls, operations, **kwargs):
        return Moment(operations)

    def __add__(self, other: 'cirq.OP_TREE') -> 'cirq.Moment':
        from cirq.circuits import circuit

        if isinstance(other, circuit.AbstractCircuit):
            return NotImplemented  # Delegate to Circuit.__radd__.
        return self.with_operations(other)

    def __sub__(self, other: 'cirq.OP_TREE') -> 'cirq.Moment':
        from cirq.ops import op_tree

        must_remove = set(op_tree.flatten_to_ops(other))
        new_ops = []
        for op in self.operations:
            if op in must_remove:
                must_remove.remove(op)
            else:
                new_ops.append(op)
        if must_remove:
            raise ValueError(
                f"Subtracted missing operations from a moment.\n"
                f"Missing operations: {must_remove!r}\n"
                f"Moment: {self!r}"
            )
        return Moment(new_ops)

    # pylint: disable=function-redefined
    @overload
    def __getitem__(self, key: raw_types.Qid) -> 'cirq.Operation':
        pass

    @overload
    def __getitem__(self, key: Iterable[raw_types.Qid]) -> 'cirq.Moment':
        pass

    def __getitem__(self, key):
        if isinstance(key, raw_types.Qid):
            if key not in self._qubit_to_op:
                raise KeyError("Moment doesn't act on given qubit")
            return self._qubit_to_op[key]
        elif isinstance(key, Iterable):
            qubits_to_keep = frozenset(key)
            ops_to_keep = []
            for q in qubits_to_keep:
                if q in self._qubit_to_op:
                    ops_to_keep.append(self._qubit_to_op[q])
            return Moment(frozenset(ops_to_keep))

    def to_text_diagram(
        self: 'cirq.Moment',
        *,
        xy_breakdown_func: Callable[['cirq.Qid'], Tuple[Any, Any]] = _default_breakdown,
        extra_qubits: Iterable['cirq.Qid'] = (),
        use_unicode_characters: bool = True,
        precision: Optional[int] = None,
        include_tags: bool = True,
    ):
        """
        Args:
            xy_breakdown_func: A function to split qubits/qudits into x and y
                components. For example, the default breakdown turns
                `cirq.GridQubit(row, col)` into the tuple `(col, row)` and
                `cirq.LineQubit(x)` into `(x, 0)`.
            extra_qubits: Extra qubits/qudits to include in the diagram, even
                if they don't have any operations applied in the moment.
            use_unicode_characters: Whether or not the output should use fancy
                unicode characters or stick to plain ASCII. Unicode characters
                look nicer, but some environments don't draw them with the same
                width as ascii characters (which ruins the diagrams).
            precision: How precise numbers, such as angles, should be. Use None
                for infinite precision, or an integer for a certain number of
                digits of precision.
            include_tags: Whether or not to include operation tags in the
                diagram.

        Returns:
            The text diagram rendered into text.
        """

        # Figure out where to place everything.
        qs = set(self.qubits) | set(extra_qubits)
        points = {xy_breakdown_func(q) for q in qs}
        x_keys = sorted({pt[0] for pt in points}, key=_SortByValFallbackToType)
        y_keys = sorted({pt[1] for pt in points}, key=_SortByValFallbackToType)
        x_map = {x_key: x + 2 for x, x_key in enumerate(x_keys)}
        y_map = {y_key: y + 2 for y, y_key in enumerate(y_keys)}
        qubit_positions = {}
        for q in qs:
            a, b = xy_breakdown_func(q)
            qubit_positions[q] = x_map[a], y_map[b]

        from cirq.circuits.text_diagram_drawer import TextDiagramDrawer

        diagram = TextDiagramDrawer()

        def cleanup_key(key: Any) -> Any:
            if isinstance(key, float) and key == int(key):
                return str(int(key))
            return str(key)

        # Add table headers.
        for key, x in x_map.items():
            diagram.write(x, 0, cleanup_key(key))
        for key, y in y_map.items():
            diagram.write(0, y, cleanup_key(key))
        diagram.horizontal_line(1, 0, len(x_map) + 2)
        diagram.vertical_line(1, 0, len(y_map) + 2)
        diagram.force_vertical_padding_after(0, 0)
        diagram.force_vertical_padding_after(1, 0)

        # Add operations.
        for op in self.operations:
            args = protocols.CircuitDiagramInfoArgs(
                known_qubits=op.qubits,
                known_qubit_count=len(op.qubits),
                use_unicode_characters=use_unicode_characters,
                qubit_map=None,
                precision=precision,
                include_tags=include_tags,
            )
            info = protocols.CircuitDiagramInfo._op_info_with_fallback(op, args=args)
            symbols = info._wire_symbols_including_formatted_exponent(args)
            for label, q in zip(symbols, op.qubits):
                x, y = qubit_positions[q]
                diagram.write(x, y, label)
            if info.connected:
                for q1, q2 in zip(op.qubits, op.qubits[1:]):
                    # Sort to get a more consistent orientation for diagonals.
                    # This reduces how often lines overlap in the diagram.
                    q1, q2 = sorted([q1, q2])

                    x1, y1 = qubit_positions[q1]
                    x2, y2 = qubit_positions[q2]
                    if x1 != x2:
                        diagram.horizontal_line(y1, x1, x2)
                    if y1 != y2:
                        diagram.vertical_line(x2, y1, y2)

        return diagram.render()

    def _commutes_(
        self, other: Any, *, atol: Union[int, float] = 1e-8
    ) -> Union[bool, NotImplementedType]:
        """Determines whether Moment commutes with the Operation.

        Args:
            other: An Operation object. Other types are not implemented yet.
                In case a different type is specified, NotImplemented is
                returned.
            atol: Absolute error tolerance. If all entries in v1@v2 - v2@v1
                have a magnitude less than this tolerance, v1 and v2 can be
                reported as commuting. Defaults to 1e-8.

        Returns:
            True: The Moment and Operation commute OR they don't have shared
            quibits.
            False: The two values do not commute.
            NotImplemented: In case we don't know how to check this, e.g.
                the parameter type is not supported yet.
        """
        if not isinstance(other, ops.Operation):
            return NotImplemented

        other_qubits = set(other.qubits)
        for op in self.operations:
            if not other_qubits.intersection(set(op.qubits)):
                continue

            commutes = protocols.commutes(op, other, atol=atol, default=NotImplemented)

            if not commutes or commutes is NotImplemented:
                return commutes

        return True


class _SortByValFallbackToType:
    def __init__(self, value):
        self.value = value

    def __lt__(self, other):
        try:
            return self.value < other.value
        except TypeError:
            t1 = type(self.value)
            t2 = type(other.value)
            return str(t1) < str(t2)
