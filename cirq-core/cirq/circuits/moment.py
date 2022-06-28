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

import itertools
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    Mapping,
    overload,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

import numpy as np

from cirq import protocols, ops, qis
from cirq._import import LazyLoader
from cirq.ops import raw_types, op_tree
from cirq.protocols import circuit_diagram_info_protocol
from cirq.type_workarounds import NotImplementedType

if TYPE_CHECKING:
    import cirq

# Lazy imports to break circular dependencies.
circuits = LazyLoader("circuits", globals(), "cirq.circuits.circuit")
op_tree = LazyLoader("op_tree", globals(), "cirq.ops.op_tree")
text_diagram_drawer = LazyLoader(
    "text_diagram_drawer", globals(), "cirq.circuits.text_diagram_drawer"
)

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

    def __init__(self, *contents: 'cirq.OP_TREE') -> None:
        """Constructs a moment with the given operations.

        Args:
            contents: The operations applied within the moment.
                Will be flattened and frozen into a tuple before storing.

        Raises:
            ValueError: A qubit appears more than once.
        """
        self._operations = tuple(op_tree.flatten_to_ops(contents))
        self._sorted_operations: Optional[Tuple['cirq.Operation', ...]] = None

        # An internal dictionary to support efficient operation access by qubit.
        self._qubit_to_op: Dict['cirq.Qid', 'cirq.Operation'] = {}
        for op in self.operations:
            for q in op.qubits:
                # Check that operations don't overlap.
                if q in self._qubit_to_op:
                    raise ValueError(f'Overlapping operations: {self.operations}')
                self._qubit_to_op[q] = op

        self._qubits = frozenset(self._qubit_to_op.keys())
        self._measurement_key_objs: Optional[FrozenSet['cirq.MeasurementKey']] = None
        self._control_keys: Optional[FrozenSet['cirq.MeasurementKey']] = None

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
        return not self._qubits.isdisjoint(qubits)

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

        Raises:
            ValueError: If the operation given overlaps a current operation in the moment.
        """
        if any(q in self._qubits for q in operation.qubits):
            raise ValueError(f'Overlapping operations: {operation}')

        # Use private variables to facilitate a quick copy.
        m = Moment()
        m._operations = self._operations + (operation,)
        m._sorted_operations = None
        m._qubits = self._qubits.union(operation.qubits)
        m._qubit_to_op = {**self._qubit_to_op, **{q: operation for q in operation.qubits}}

        m._measurement_key_objs = self._measurement_key_objs_().union(
            protocols.measurement_key_objs(operation)
        )
        m._control_keys = self._control_keys_().union(protocols.control_keys(operation))

        return m

    def with_operations(self, *contents: 'cirq.OP_TREE') -> 'cirq.Moment':
        """Returns a new moment with the given contents added.

        Args:
            *contents: New operations to add to this moment.

        Returns:
            The new moment.

        Raises:
            ValueError: If the contents given overlaps a current operation in the moment.
        """
        flattened_contents = tuple(op_tree.flatten_to_ops(contents))

        m = Moment()
        # Use private variables to facilitate a quick copy.
        m._qubit_to_op = self._qubit_to_op.copy()
        qubits = set(self._qubits)
        for op in flattened_contents:
            if any(q in qubits for q in op.qubits):
                raise ValueError(f'Overlapping operations: {op}')
            qubits.update(op.qubits)
            for q in op.qubits:
                m._qubit_to_op[q] = op
        m._qubits = frozenset(qubits)

        m._operations = self._operations + flattened_contents
        m._sorted_operations = None
        m._measurement_key_objs = self._measurement_key_objs_().union(
            set(itertools.chain(*(protocols.measurement_key_objs(op) for op in flattened_contents)))
        )
        m._control_keys = self._control_keys_().union(
            set(itertools.chain(*(protocols.control_keys(op) for op in flattened_contents)))
        )

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

    def _with_measurement_key_mapping_(self, key_map: Mapping[str, str]):
        return Moment(
            protocols.with_measurement_key_mapping(op, key_map)
            if protocols.measurement_keys_touched(op)
            else op
            for op in self.operations
        )

    def _measurement_key_names_(self) -> FrozenSet[str]:
        return frozenset(str(key) for key in self._measurement_key_objs_())

    def _measurement_key_objs_(self) -> FrozenSet['cirq.MeasurementKey']:
        if self._measurement_key_objs is None:
            self._measurement_key_objs = frozenset(
                key for op in self.operations for key in protocols.measurement_key_objs(op)
            )
        return self._measurement_key_objs

    def _control_keys_(self) -> FrozenSet['cirq.MeasurementKey']:
        if self._control_keys is None:
            self._control_keys = frozenset(
                k for op in self.operations for k in protocols.control_keys(op)
            )
        return self._control_keys

    def _sorted_operations_(self) -> Tuple['cirq.Operation', ...]:
        if self._sorted_operations is None:
            self._sorted_operations = tuple(sorted(self._operations, key=lambda op: op.qubits))
        return self._sorted_operations

    def _with_key_path_(self, path: Tuple[str, ...]):
        return Moment(
            protocols.with_key_path(op, path) if protocols.is_measurement(op) else op
            for op in self.operations
        )

    def _with_key_path_prefix_(self, prefix: Tuple[str, ...]):
        return Moment(
            protocols.with_key_path_prefix(op, prefix)
            if protocols.measurement_keys_touched(op)
            else op
            for op in self.operations
        )

    def _with_rescoped_keys_(
        self, path: Tuple[str, ...], bindable_keys: FrozenSet['cirq.MeasurementKey']
    ):
        return Moment(
            protocols.with_rescoped_keys(op, path, bindable_keys) for op in self.operations
        )

    def __copy__(self):
        return type(self)(self.operations)

    def __bool__(self) -> bool:
        return bool(self.operations)

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented

        return self._sorted_operations_() == other._sorted_operations_()

    def _approx_eq_(self, other: Any, atol: Union[int, float]) -> bool:
        """See `cirq.protocols.SupportsApproximateEquality`."""
        if not isinstance(other, type(self)):
            return NotImplemented

        return protocols.approx_eq(
            self._sorted_operations_(), other._sorted_operations_(), atol=atol
        )

    def __ne__(self, other) -> bool:
        return not self == other

    def __hash__(self):
        return hash((Moment, self._sorted_operations_()))

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

    def _decompose_(self) -> 'cirq.OP_TREE':
        """See `cirq.SupportsDecompose`."""
        return self._operations

    def transform_qubits(
        self: TSelf_Moment,
        qubit_map: Union[Dict['cirq.Qid', 'cirq.Qid'], Callable[['cirq.Qid'], 'cirq.Qid']],
    ) -> TSelf_Moment:
        """Returns the same moment, but with different qubits.

        Args:
           qubit_map: A function or a dict mapping each current qubit into a
                      desired new qubit.

        Returns:
            The receiving moment but with qubits transformed by the given
                function.
        """
        return self.__class__(op.transform_qubits(qubit_map) for op in self.operations)

    def expand_to(self, qubits: Iterable['cirq.Qid']) -> 'cirq.Moment':
        """Returns self expanded to given superset of qubits by making identities explicit.

        Args:
            qubits: Iterable of `cirq.Qid`s to expand this moment to.

        Returns:
            A new `cirq.Moment` with identity operations on the new qubits
            not currently found in the moment.

        Raises:
            ValueError: if this moments' qubits are not a subset of `qubits`.
        """
        if not self.qubits.issubset(qubits):
            raise ValueError(f'{qubits} is not a superset of {self.qubits}')

        operations = list(self.operations)
        for q in set(qubits) - self.qubits:
            operations.append(ops.I(q))
        return Moment(*operations)

    def _has_kraus_(self) -> bool:
        """Returns True if self has a Kraus representation and self uses <= 10 qubits."""
        return all(protocols.has_kraus(op) for op in self.operations) and len(self.qubits) <= 10

    def _kraus_(self) -> Sequence[np.ndarray]:
        r"""Returns Kraus representation of self.

        The method computes a Kraus representation of self from Kraus representations of its
        constituent operations by taking the tensor product of Kraus operators along all paths
        corresponding to the possible choices of the Kraus operators of the operations. More
        precisely, it computes all terms in the expression

        $$
        \sum_{i_1} \sum_{i_2} \dots \sum_{i_m} \bigotimes_{k=1}^m K_{k,i_k}
        $$

        where $K_{k,j}$ is the jth Kraus operator of the kth operation in self. Each term becomes
        an element in the sequence returned by this method.

        Args:
            self: This Moment.
        Returns:
            A Kraus representation of self if `self._has_kraus_()` is True else `NotImplemented`.
        """
        if not self._has_kraus_():
            return NotImplemented

        qubits = sorted(self.qubits)
        n = len(qubits)
        if n < 1:
            return (np.array([[1 + 0j]]),)

        qubit_to_row_subscript = dict(zip(qubits, 'abcdefghij'))
        qubit_to_col_subscript = dict(zip(qubits, 'ABCDEFGHIJ'))

        def row_subscripts(qs: Sequence['cirq.Qid']) -> str:
            return ''.join(qubit_to_row_subscript[q] for q in qs)

        def col_subscripts(qs: Sequence['cirq.Qid']) -> str:
            return ''.join(qubit_to_col_subscript[q] for q in qs)

        def kraus_tensors(op: 'cirq.Operation') -> Sequence[np.ndarray]:
            return tuple(np.reshape(k, (2, 2) * len(op.qubits)) for k in protocols.kraus(op))

        input_subscripts = ','.join(
            row_subscripts(op.qubits) + col_subscripts(op.qubits) for op in self.operations
        )
        output_subscripts = row_subscripts(qubits) + col_subscripts(qubits)
        assert len(input_subscripts) == 2 * n + len(self.operations) - 1
        assert len(output_subscripts) == 2 * n

        transpose = input_subscripts + '->' + output_subscripts

        r = []
        d = 2**n
        kss = [kraus_tensors(op) for op in self.operations]
        for ks in itertools.product(*kss):
            k = np.einsum(transpose, *ks)
            r.append(np.reshape(k, (d, d)))
        return r

    def _has_superoperator_(self) -> bool:
        """Returns True if self has superoperator representation."""
        return self._has_kraus_()

    def _superoperator_(self) -> np.ndarray:
        """Returns superoperator representation of self if possible, else `NotImplemented`."""
        if not self._has_superoperator_():
            return NotImplemented
        return qis.kraus_to_superoperator(self._kraus_())

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['operations'])

    @classmethod
    def _from_json_dict_(cls, operations, **kwargs):
        return Moment(operations)

    def __add__(self, other: 'cirq.OP_TREE') -> 'cirq.Moment':

        if isinstance(other, circuits.AbstractCircuit):
            return NotImplemented  # Delegate to Circuit.__radd__.
        return self.with_operations(other)

    def __sub__(self, other: 'cirq.OP_TREE') -> 'cirq.Moment':

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
    ) -> str:
        """Create a text diagram for the moment.

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

        diagram = text_diagram_drawer.TextDiagramDrawer()

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
        for op in self._sorted_operations_():
            args = protocols.CircuitDiagramInfoArgs(
                known_qubits=op.qubits,
                known_qubit_count=len(op.qubits),
                use_unicode_characters=use_unicode_characters,
                label_map=None,
                precision=precision,
                include_tags=include_tags,
            )
            info = circuit_diagram_info_protocol._op_info_with_fallback(op, args=args)
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

    def _commutes_(self, other: Any, *, atol: float = 1e-8) -> Union[bool, NotImplementedType]:
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
