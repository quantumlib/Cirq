# Copyright 2021 The Cirq Developers
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

"""Defines primitives for common transformer patterns."""

from collections import defaultdict
import bisect
import dataclasses

from typing import (
    cast,
    Callable,
    Dict,
    Hashable,
    List,
    Optional,
    Sequence,
    Set,
    Union,
    Tuple,
    TYPE_CHECKING,
)

from cirq import circuits, ops, protocols
from cirq.circuits.circuit import CIRCUIT_TYPE

if TYPE_CHECKING:
    import cirq

MAPPED_CIRCUIT_OP_TAG = '<mapped_circuit_op>'


def _to_target_circuit_type(
    circuit: circuits.AbstractCircuit, target_circuit: CIRCUIT_TYPE
) -> CIRCUIT_TYPE:
    return cast(
        CIRCUIT_TYPE,
        circuit.unfreeze(copy=False)
        if isinstance(target_circuit, circuits.Circuit)
        else circuit.freeze(),
    )


def _create_target_circuit_type(ops: ops.OP_TREE, target_circuit: CIRCUIT_TYPE) -> CIRCUIT_TYPE:
    return cast(
        CIRCUIT_TYPE,
        circuits.Circuit(ops)
        if isinstance(target_circuit, circuits.Circuit)
        else circuits.FrozenCircuit(ops),
    )


def map_moments(
    circuit: CIRCUIT_TYPE,
    map_func: Callable[[circuits.Moment, int], Union[circuits.Moment, Sequence[circuits.Moment]]],
    *,
    tags_to_ignore: Sequence[Hashable] = (),
    deep: bool = False,
) -> CIRCUIT_TYPE:
    """Applies local transformation on moments, by calling `map_func(moment)` for each moment.

    Args:
        circuit: Input circuit to apply the transformations on. The input circuit is not mutated.
        map_func: Mapping function from (cirq.Moment, moment_index) to a sequence of moments.
        tags_to_ignore: Tagged circuit operations marked with any of `tags_to_ignore` will be
            ignored when recursively applying the transformer primitive to sub-circuits, given
            deep=True.
        deep: If true, `map_func` will be recursively applied to circuits wrapped inside
            any circuit operations contained within `circuit`.

    Returns:
        Copy of input circuit with mapped moments.
    """
    mutable_circuit = circuit.unfreeze(copy=False)
    if deep:
        batch_replace = []
        for i, op in circuit.findall_operations(
            lambda o: isinstance(o.untagged, circuits.CircuitOperation)
        ):
            if set(op.tags).intersection(tags_to_ignore):
                continue
            op_untagged = cast(circuits.CircuitOperation, op.untagged)
            mapped_op = op_untagged.replace(
                circuit=map_moments(
                    op_untagged.circuit, map_func, tags_to_ignore=tags_to_ignore, deep=deep
                )
            ).with_tags(*op.tags)
            batch_replace.append((i, op, mapped_op))
        mutable_circuit = circuit.unfreeze(copy=True)
        mutable_circuit.batch_replace(batch_replace)
    return _create_target_circuit_type(
        (map_func(mutable_circuit[i], i) for i in range(len(mutable_circuit))), circuit
    )


def _map_operations_impl(
    circuit: CIRCUIT_TYPE,
    map_func: Callable[[ops.Operation, int], ops.OP_TREE],
    *,
    deep: bool = False,
    raise_if_add_qubits=True,
    tags_to_ignore: Sequence[Hashable] = (),
    wrap_in_circuit_op: bool = True,
) -> CIRCUIT_TYPE:
    """Applies local transformations, by calling `map_func(op, moment_index)` for each operation.

    This method provides a fast, iterative implementation for the two `map_operations_*` variants
    exposed as public transformer primitives. The high level idea for the iterative implementation
    is to
        1) For each operation `op`, find the corresponding mapped operation(s) `mapped_ops`. The
            set of mapped operations can be either wrapped in a circuit operation or not, depending
            on the value of flag `wrap_in_circuit_op` and whether the mapped operations will end up
            occupying more than one moment or not.
        2) Use the `get_earliest_accommodating_moment_index` infrastructure built for `cirq.Circuit`
            construction to determine the index at which the mapped operations should be inserted.
            This step takes care of the nuances that arise due to (a) preserving moment structure
            and (b) mapped operations spanning across multiple moments (these both are trivial when
            `op` is mapped to a single `mapped_op` that acts on the same set of qubits).

    By default, the function assumes `issubset(qubit_set(map_func(op, moment_index)), op.qubits)` is
    True.

    Args:
        circuit: Input circuit to apply the transformations on. The input circuit is not mutated.
        map_func: Mapping function from (cirq.Operation, moment_index) to a cirq.OP_TREE. If the
            resulting optree spans more than 1 moment, it's either wrapped in a tagged circuit
            operation and inserted in-place in the same moment (if  `wrap_in_circuit_op` is True)
            OR the mapped operations are inserted directly in the circuit, preserving moment
            strucutre. The effect is equivalent to (but much faster) a two-step approach of first
            wrapping the operations in a circuit operation and then calling `cirq.unroll_circuit_op`
            to unroll the corresponding circuit ops.
        deep: If true, `map_func` will be recursively applied to circuits wrapped inside
            any circuit operations contained within `circuit`.
        raise_if_add_qubits: Set to True by default. If True, raises ValueError if
            `map_func(op, idx)` adds operations on qubits outside of `op.qubits`.
        tags_to_ignore: Sequence of tags which should be ignored while applying `map_func` on
            tagged operations -- i.e. `map_func(op, idx)` will be called only for operations that
            satisfy `set(op.tags).isdisjoint(tags_to_ignore)`.
        wrap_in_circuit_op: If True, the mapped operations will be wrapped in a tagged circuit
        operation and inserted in-place if they occupy more than one moment.

    Raises:
          ValueError if `issubset(qubit_set(map_func(op, idx)), op.qubits) is False` and
            `raise_if_add_qubits is True`.

    Returns:
        Copy of input circuit with mapped operations.
    """
    tags_to_ignore_set = set(tags_to_ignore)

    def apply_map_func(op: 'cirq.Operation', idx: int) -> List['cirq.Operation']:
        if tags_to_ignore_set.intersection(op.tags):
            return [op]
        if deep and isinstance(op.untagged, circuits.CircuitOperation):
            op = op.untagged.replace(
                circuit=_map_operations_impl(
                    op.untagged.circuit,
                    map_func,
                    deep=deep,
                    raise_if_add_qubits=raise_if_add_qubits,
                    tags_to_ignore=tags_to_ignore,
                    wrap_in_circuit_op=wrap_in_circuit_op,
                )
            ).with_tags(*op.tags)
        mapped_ops = [*ops.flatten_to_ops(map_func(op, idx))]
        op_qubits = set(op.qubits)
        mapped_ops_qubits: Set['cirq.Qid'] = set()
        has_overlapping_ops = False
        for mapped_op in mapped_ops:
            if raise_if_add_qubits and not op_qubits.issuperset(mapped_op.qubits):
                raise ValueError(
                    f"Mapped operations {mapped_ops} should act on a subset "
                    f"of qubits of the original operation {op}"
                )
            if mapped_ops_qubits.intersection(mapped_op.qubits):
                has_overlapping_ops = True
            mapped_ops_qubits = mapped_ops_qubits.union(mapped_op.qubits)
        if wrap_in_circuit_op and has_overlapping_ops:
            # Mapped operations should be wrapped in a `CircuitOperation` only iff they occupy more
            # than one moment, i.e. there are at least two operations that share a qubit.
            mapped_ops = [
                circuits.CircuitOperation(circuits.FrozenCircuit(mapped_ops)).with_tags(
                    MAPPED_CIRCUIT_OP_TAG
                )
            ]
        return mapped_ops

    new_moments: List[List['cirq.Operation']] = []

    # Keep track of the latest time index for each qubit, measurement key, and control key.
    qubit_time_index: Dict['cirq.Qid', int] = {}
    measurement_time_index: Dict['cirq.MeasurementKey', int] = {}
    control_time_index: Dict['cirq.MeasurementKey', int] = {}

    # New mapped operations in the current moment should be inserted after `last_moment_time_index`.
    last_moment_time_index = -1

    for idx, moment in enumerate(circuit):
        if wrap_in_circuit_op:
            new_moments.append([])
        for op in moment:
            mapped_ops = apply_map_func(op, idx)

            for mapped_op in mapped_ops:
                # Identify the earliest moment that can accommodate this op.
                placement_index = circuits.circuit.get_earliest_accommodating_moment_index(
                    mapped_op, qubit_time_index, measurement_time_index, control_time_index
                )
                placement_index = max(placement_index, last_moment_time_index + 1)
                new_moments.extend([[] for _ in range(placement_index - len(new_moments) + 1)])
                new_moments[placement_index].append(mapped_op)
                for qubit in mapped_op.qubits:
                    qubit_time_index[qubit] = placement_index
                for key in protocols.measurement_key_objs(mapped_op):
                    measurement_time_index[key] = placement_index
                for key in protocols.control_keys(mapped_op):
                    control_time_index[key] = placement_index

        last_moment_time_index = len(new_moments) - 1

    return _create_target_circuit_type([circuits.Moment(moment) for moment in new_moments], circuit)


def map_operations(
    circuit: CIRCUIT_TYPE,
    map_func: Callable[[ops.Operation, int], ops.OP_TREE],
    *,
    deep: bool = False,
    raise_if_add_qubits=True,
    tags_to_ignore: Sequence[Hashable] = (),
) -> CIRCUIT_TYPE:
    """Applies local transformations, by calling `map_func(op, moment_index)` for each operation.

    By default, the function assumes `issubset(qubit_set(map_func(op, moment_index)), op.qubits)` is
    True.

    Args:
        circuit: Input circuit to apply the transformations on. The input circuit is not mutated.
        map_func: Mapping function from (cirq.Operation, moment_index) to a cirq.OP_TREE. If the
            resulting optree spans more than 1 moment, it's inserted in-place in the same moment as
            `cirq.CircuitOperation(cirq.FrozenCircuit(op_tree)).with_tags(MAPPED_CIRCUIT_OP_TAG)`
            to preserve moment structure. Utility methods like `cirq.unroll_circuit_op` can
            subsequently be used to unroll the mapped circuit operation.
        deep: If true, `map_func` will be recursively applied to circuits wrapped inside
            any circuit operations contained within `circuit`.
        raise_if_add_qubits: Set to True by default. If True, raises ValueError if
            `map_func(op, idx)` adds operations on qubits outside of `op.qubits`.
        tags_to_ignore: Sequence of tags which should be ignored while applying `map_func` on
            tagged operations -- i.e. `map_func(op, idx)` will be called only for operations that
            satisfy `set(op.tags).isdisjoint(tags_to_ignore)`.

    Raises:
          ValueError if `issubset(qubit_set(map_func(op, idx)), op.qubits) is False` and
            `raise_if_add_qubits is True`.

    Returns:
        Copy of input circuit with mapped operations (wrapped in a tagged CircuitOperation).
    """
    return _map_operations_impl(
        circuit,
        map_func,
        deep=deep,
        raise_if_add_qubits=raise_if_add_qubits,
        tags_to_ignore=tags_to_ignore,
        wrap_in_circuit_op=True,
    )


def map_operations_and_unroll(
    circuit: CIRCUIT_TYPE,
    map_func: Callable[[ops.Operation, int], ops.OP_TREE],
    *,
    deep: bool = False,
    raise_if_add_qubits=True,
    tags_to_ignore: Sequence[Hashable] = (),
) -> CIRCUIT_TYPE:
    """Applies local transformations via `cirq.map_operations` & unrolls intermediate circuit ops.

    See `cirq.map_operations` and `cirq.unroll_circuit_op` for more details.

    Args:
        circuit: Input circuit to apply the transformations on. The input circuit is not mutated.
        map_func: Mapping function from (cirq.Operation, moment_index) to a cirq.OP_TREE.
        deep: If true, `map_func` will be recursively applied to circuits wrapped inside
            any circuit operations contained within `circuit`.
        raise_if_add_qubits: Set to True by default. If True, raises ValueError if
            `map_func(op, idx)` adds operations on qubits outside `op.qubits`.
        tags_to_ignore: Sequence of tags which should be ignored while applying `map_func` on
            tagged operations -- i.e. `map_func(op, idx)` will be called only for operations that
            satisfy `set(op.tags).isdisjoint(tags_to_ignore)`.

    Returns:
        Copy of input circuit with mapped operations, unrolled in a moment preserving way.
    """
    return _map_operations_impl(
        circuit,
        map_func,
        deep=deep,
        raise_if_add_qubits=raise_if_add_qubits,
        tags_to_ignore=tags_to_ignore,
        wrap_in_circuit_op=False,
    )


@dataclasses.dataclass
class _MergedCircuit:
    """An optimized internal representation of a circuit, tailored for `cirq.merge_operations`

    Attributes:
        qubit_indexes: Mapping from qubits to (sorted) list of moment indexes containing operations
            acting on the qubit.
        mkey_indexes: Mapping from measurement keys to (sorted) list of moment indexes containing
            measurement operations with the same key.
        ckey_indexes: Mapping from measurement keys to (sorted) list of moment indexes containing
            classically controlled operations controlled on the same key.
        ops_by_index: List of circuit moments containing operations. We use a dictionary instead
            of a set to store operations to preserve insertion order.
    """

    qubit_indexes: Dict['cirq.Qid', List[int]] = dataclasses.field(
        default_factory=lambda: defaultdict(lambda: [-1])
    )
    mkey_indexes: Dict['cirq.MeasurementKey', List[int]] = dataclasses.field(
        default_factory=lambda: defaultdict(lambda: [-1])
    )
    ckey_indexes: Dict['cirq.MeasurementKey', List[int]] = dataclasses.field(
        default_factory=lambda: defaultdict(lambda: [-1])
    )
    ops_by_index: List[Dict['cirq.Operation', int]] = dataclasses.field(default_factory=list)

    def append_empty_moment(self) -> None:
        self.ops_by_index.append({})

    def add_op_to_moment(self, moment_index: int, op: 'cirq.Operation') -> None:
        self.ops_by_index[moment_index][op] = 0
        for q in op.qubits:
            if moment_index > self.qubit_indexes[q][-1]:
                self.qubit_indexes[q].append(moment_index)
            else:
                bisect.insort(self.qubit_indexes[q], moment_index)
        for mkey in protocols.measurement_key_objs(op):
            bisect.insort(self.mkey_indexes[mkey], moment_index)
        for ckey in protocols.control_keys(op):
            bisect.insort(self.ckey_indexes[ckey], moment_index)

    def remove_op_from_moment(self, moment_index: int, op: 'cirq.Operation') -> None:
        self.ops_by_index[moment_index].pop(op)
        for q in op.qubits:
            if self.qubit_indexes[q][-1] == moment_index:
                self.qubit_indexes[q].pop()
            else:
                self.qubit_indexes[q].remove(moment_index)
        for mkey in protocols.measurement_key_objs(op):
            self.mkey_indexes[mkey].remove(moment_index)
        for ckey in protocols.control_keys(op):
            self.ckey_indexes[ckey].remove(moment_index)

    def get_mergeable_ops(
        self, op: 'cirq.Operation', op_qs: Set['cirq.Qid']
    ) -> Tuple[int, List['cirq.Operation']]:
        # Find the index of previous moment which can be merged with `op`.
        idx = max([self.qubit_indexes[q][-1] for q in op_qs], default=-1)
        idx = max([idx] + [self.mkey_indexes[ckey][-1] for ckey in protocols.control_keys(op)])
        idx = max(
            [idx] + [self.ckey_indexes[mkey][-1] for mkey in protocols.measurement_key_objs(op)]
        )
        # Return the set of overlapping ops in moment with index `idx`.
        if idx == -1:
            return idx, []

        return idx, [
            left_op for left_op in self.ops_by_index[idx] if not op_qs.isdisjoint(left_op.qubits)
        ]

    def get_cirq_circuit(self) -> 'cirq.Circuit':
        return circuits.Circuit(circuits.Moment(m.keys()) for m in self.ops_by_index)


def merge_operations(
    circuit: CIRCUIT_TYPE,
    merge_func: Callable[[ops.Operation, ops.Operation], Optional[ops.Operation]],
    *,
    tags_to_ignore: Sequence[Hashable] = (),
    deep: bool = False,
) -> CIRCUIT_TYPE:
    """Merges operations in a circuit by calling `merge_func` iteratively on operations.

    Two operations op1 and op2 are merge-able if
        - There is no other operations between op1 and op2 in the circuit
        - is_subset(op1.qubits, op2.qubits) or is_subset(op2.qubits, op1.qubits)

    The `merge_func` is a callable which, given two merge-able operations
    op1 and op2, decides whether they should be merged into a single operation
    or not. If not, it should return None, else it should return the single merged
    operations `op`.

    The method iterates on the input circuit moment-by-moment from left to right and attempts
    to repeatedly merge each operation in the latest moment with all the corresponding merge-able
    operations to its left.

    If op1 and op2 are merged, both op1 and op2 are deleted from the circuit and
    the resulting `merged_op` is inserted at the index corresponding to the larger
    of op1/op2. If both op1 and op2 act on the same number of qubits, `merged_op` is
    inserted in the smaller moment index to minimize circuit depth.

    The number of calls to `merge_func` is O(N), where N = Total no. of operations, because:
        - Every time the `merge_func` returns a new operation, the number of operations in the
            circuit reduce by 1 and hence this can happen at most O(N) times
        - Every time the `merge_func` returns None, the current operation is inserted into the
            frontier and we go on to process the next operation, which can also happen at-most
            O(N) times.

    Args:
        circuit: Input circuit to apply the transformations on. The input circuit is not mutated.
        merge_func: Callable to determine whether two merge-able operations in the circuit should
            be merged. If the operations can be merged, the callable should return the merged
            operation, else None.
        tags_to_ignore: Sequence of tags which should be ignored while applying `merge_func` on
            tagged operations -- i.e. `merge_func(op1, op2)` will be called only if both `op1` and
            `op2` satisfy `set(op.tags).isdisjoint(tags_to_ignore)`.
        deep: If true, the transformer primitive will be recursively applied to all circuits
            wrapped inside circuit operations.


    Returns:
        Copy of input circuit with merged operations.

    Raises:
        ValueError if the merged operation acts on new qubits outside the set of qubits
            corresponding to the original operations to be merged.
    """
    _circuit_op_tag = "_internal_tag_to_mark_circuit_ops_in_circuit"
    tags_to_ignore_set = set(tags_to_ignore) | {_circuit_op_tag}

    def apply_merge_func(op1: ops.Operation, op2: ops.Operation) -> Optional[ops.Operation]:
        if not all(tags_to_ignore_set.isdisjoint(op.tags) for op in [op1, op2]):
            return None
        new_op = merge_func(op1, op2)
        qubit_set = frozenset(op1.qubits + op2.qubits)
        if new_op is not None and not qubit_set.issuperset(new_op.qubits):
            raise ValueError(
                f"Merged operation {new_op} must act on a subset of qubits of "
                f"original operations {op1} and {op2}"
            )
        return new_op

    merged_circuit = _MergedCircuit()
    for moment_idx, current_moment in enumerate(cast(List['cirq.Moment'], circuit)):
        merged_circuit.append_empty_moment()
        for op in sorted(current_moment.operations, key=lambda op: op.qubits):
            if (
                deep
                and isinstance(op.untagged, circuits.CircuitOperation)
                and tags_to_ignore_set.isdisjoint(op.tags)
            ):
                op_untagged = op.untagged
                merged_circuit.add_op_to_moment(
                    moment_idx,
                    op_untagged.replace(
                        circuit=merge_operations(
                            op_untagged.circuit,
                            merge_func,
                            tags_to_ignore=tags_to_ignore,
                            deep=True,
                        )
                    ).with_tags(*op.tags, _circuit_op_tag),
                )
                continue

            op_qs = set(op.qubits)
            left_idx, left_ops = merged_circuit.get_mergeable_ops(op, op_qs)
            if len(left_ops) == 1 and op_qs.issubset(left_ops[0].qubits):
                # Case-1: Try to merge op with the larger operation on the left.
                new_op = apply_merge_func(left_ops[0], op)
                if new_op is not None:
                    merged_circuit.remove_op_from_moment(left_idx, left_ops[0])
                    merged_circuit.add_op_to_moment(left_idx, new_op)
                else:
                    merged_circuit.add_op_to_moment(moment_idx, op)
                continue

            while left_ops and op_qs:
                # Case-2: left_ops will merge right into `op` whenever possible.
                for left_op in left_ops:
                    is_merged = False
                    if op_qs.issuperset(left_op.qubits):
                        # Try to merge left_op into op
                        new_op = apply_merge_func(left_op, op)
                        if new_op is not None:
                            merged_circuit.remove_op_from_moment(left_idx, left_op)
                            op, is_merged = new_op, True
                    if not is_merged:
                        op_qs -= frozenset(left_op.qubits)
                left_idx, left_ops = merged_circuit.get_mergeable_ops(op, op_qs)
            merged_circuit.add_op_to_moment(moment_idx, op)
    ret_circuit = merged_circuit.get_cirq_circuit()
    if deep:
        ret_circuit = map_operations(
            ret_circuit,
            lambda o, _: o.untagged.with_tags(*(set(o.tags) - {_circuit_op_tag})),
            deep=True,
        )
    return _to_target_circuit_type(ret_circuit, circuit)


def merge_operations_to_circuit_op(
    circuit: CIRCUIT_TYPE,
    can_merge: Callable[[Sequence['cirq.Operation'], Sequence['cirq.Operation']], bool],
    *,
    tags_to_ignore: Sequence[Hashable] = (),
    merged_circuit_op_tag: str = "Merged connected component",
    deep: bool = False,
) -> CIRCUIT_TYPE:
    """Merges connected components of operations and wraps each component into a circuit operation.

    Uses `cirq.merge_operations` to identify connected components of operations. Moment structure
    is preserved for operations that do not participate in merging. For merged operations, the
    newly created circuit operations are constructed by inserting operations using EARLIEST
    strategy.
    If you need more control on moment structure of newly created circuit operations, consider
    using `cirq.merge_operations` directly with a custom `merge_func`.

    Args:
        circuit: Input circuit to apply the transformations on. The input circuit is not mutated.
        can_merge: Callable to determine whether a new operation `right_op` can be merged into an
            existing connected component of operations `left_ops` based on boolen returned by
            `can_merge(left_ops, right_op)`.
        tags_to_ignore: Tagged operations marked any of `tags_to_ignore` will not be considered as
            potential candidates for any connected component.
        merged_circuit_op_tag: Tag to be applied on circuit operations wrapping valid connected
            components.
        deep: If true, the transformer primitive will be recursively applied to all circuits
            wrapped inside circuit operations.

    Returns:
        Copy of input circuit with valid connected components wrapped in tagged circuit operations.
    """

    def merge_func(op1: 'cirq.Operation', op2: 'cirq.Operation') -> Optional['cirq.Operation']:
        def get_ops(op: 'cirq.Operation'):
            op_untagged = op.untagged
            return (
                [*op_untagged.circuit.all_operations()]
                if isinstance(op_untagged, circuits.CircuitOperation)
                and merged_circuit_op_tag in op.tags
                else [op]
            )

        left_ops, right_ops = get_ops(op1), get_ops(op2)
        if not can_merge(left_ops, right_ops):
            return None
        return circuits.CircuitOperation(circuits.FrozenCircuit(left_ops, right_ops)).with_tags(
            merged_circuit_op_tag
        )

    return merge_operations(circuit, merge_func, tags_to_ignore=tags_to_ignore, deep=deep)


def merge_k_qubit_unitaries_to_circuit_op(
    circuit: CIRCUIT_TYPE,
    k: int,
    *,
    tags_to_ignore: Sequence[Hashable] = (),
    merged_circuit_op_tag: Optional[str] = None,
    deep: bool = False,
) -> CIRCUIT_TYPE:
    """Merges connected components of operations, acting on <= k qubits, into circuit operations.

    Uses `cirq.merge_operations_to_circuit_op` to identify and merge connected components of
    unitary operations acting on at-most k-qubits. Moment structure is preserved for operations
    that do not participate in merging. For merged operations, the newly created circuit operations
    are constructed by inserting operations using EARLIEST strategy.

    Args:
        circuit: Input circuit to apply the transformations on. The input circuit is not mutated.
        k: Merge-able operations acting on <= k qubits are merged into a connected component.
        tags_to_ignore: Tagged operations marked any of `tags_to_ignore` will not be considered as
            potential candidates for any connected component.
        merged_circuit_op_tag: Tag to be applied on circuit operations wrapping valid connected
            components. A default tag is applied if left None.
        deep: If true, the transformer primitive will be recursively applied to all circuits
            wrapped inside circuit operations.

    Returns:
        Copy of input circuit with valid connected components wrapped in tagged circuit operations.
    """

    def can_merge(ops1: Sequence['cirq.Operation'], ops2: Sequence['cirq.Operation']) -> bool:
        return all(
            protocols.num_qubits(op) <= k and protocols.has_unitary(op)
            for op_list in [ops1, ops2]
            for op in op_list
        )

    return merge_operations_to_circuit_op(
        circuit,
        can_merge,
        tags_to_ignore=tags_to_ignore,
        merged_circuit_op_tag=merged_circuit_op_tag or f"Merged {k}q unitary connected component.",
        deep=deep,
    )


def merge_moments(
    circuit: CIRCUIT_TYPE,
    merge_func: Callable[[circuits.Moment, circuits.Moment], Optional[circuits.Moment]],
    *,
    tags_to_ignore: Sequence[Hashable] = (),
    deep: bool = False,
) -> CIRCUIT_TYPE:
    """Merges adjacent moments, one by one from left to right, by calling `merge_func(m1, m2)`.

    Args:
        circuit: Input circuit to apply the transformations on. The input circuit is not mutated.
        merge_func: Callable to determine whether two adjacent moments in the circuit should be
            merged. If the moments can be merged, the callable should return the merged moment,
            else None.
        tags_to_ignore: Tagged circuit operations marked with any of `tags_to_ignore` will be
            ignored when recursively applying the transformer primitive to sub-circuits, given
            deep=True.
        deep: If true, the transformer primitive will be recursively applied to all circuits
            wrapped inside circuit operations.

    Returns:
        Copy of input circuit with merged moments.
    """
    if not circuit:
        return circuit
    if deep:
        circuit = map_operations(
            circuit,
            lambda op, _: op.untagged.replace(
                circuit=merge_moments(op.untagged.circuit, merge_func, deep=deep)
            ).with_tags(*op.tags)
            if isinstance(op.untagged, circuits.CircuitOperation)
            else op,
            tags_to_ignore=tags_to_ignore,
        )
    merged_moments: List[circuits.Moment] = [circuit[0]]
    for current_moment in circuit[1:]:
        merged_moment = merge_func(merged_moments[-1], current_moment)
        if merged_moment is None:
            merged_moments.append(current_moment)
        else:
            merged_moments[-1] = merged_moment
    return _create_target_circuit_type(merged_moments, circuit)


def unroll_circuit_op(
    circuit: CIRCUIT_TYPE,
    *,
    deep: bool = False,
    tags_to_check: Optional[Sequence[Hashable]] = (MAPPED_CIRCUIT_OP_TAG,),
) -> CIRCUIT_TYPE:
    """Unrolls (tagged) `cirq.CircuitOperation`s while preserving the moment structure.

    Each moment containing a matching circuit operation is expanded into a list of moments with the
    unrolled operations, hence preserving the original moment structure.

    Args:
        circuit: Input circuit to apply the transformations on. The input circuit is not mutated.
        deep: If true, the transformer primitive will be recursively applied to all circuits
            wrapped inside circuit operations.
        tags_to_check: If specified, only circuit operations tagged with one of the `tags_to_check`
            are unrolled.

    Returns:
        Copy of input circuit with (Tagged) CircuitOperation's expanded in a moment preserving way.
    """

    def map_func(m: circuits.Moment, _: int):
        to_zip: List['cirq.AbstractCircuit'] = []
        for op in m:
            op_untagged = op.untagged
            if isinstance(op_untagged, circuits.CircuitOperation):
                if deep:
                    op_untagged = op_untagged.replace(
                        circuit=unroll_circuit_op(
                            op_untagged.circuit, deep=deep, tags_to_check=tags_to_check
                        )
                    )
                to_zip.append(
                    op_untagged.mapped_circuit()
                    if (tags_to_check is None or set(tags_to_check).intersection(op.tags))
                    else circuits.Circuit(op_untagged.with_tags(*op.tags))
                )
            else:
                to_zip.append(circuits.Circuit(op))
        return circuits.Circuit.zip(*to_zip).moments

    return map_moments(circuit, map_func)


def unroll_circuit_op_greedy_earliest(
    circuit: CIRCUIT_TYPE,
    *,
    deep: bool = False,
    tags_to_check: Optional[Sequence[Hashable]] = (MAPPED_CIRCUIT_OP_TAG,),
) -> CIRCUIT_TYPE:
    """Unrolls (tagged) `cirq.CircuitOperation`s by inserting operations using EARLIEST strategy.

    Each matching `cirq.CircuitOperation` is replaced by inserting underlying operations using the
    `cirq.InsertStrategy.EARLIEST` strategy. The greedy approach attempts to minimize circuit depth
    of the resulting circuit.

    Args:
        circuit: Input circuit to apply the transformations on. The input circuit is not mutated.
        deep: If true, the transformer primitive will be recursively applied to all circuits
            wrapped inside circuit operations.
        tags_to_check: If specified, only circuit operations tagged with one of the `tags_to_check`
            are unrolled.

    Returns:
        Copy of input circuit with (Tagged) CircuitOperation's expanded using EARLIEST strategy.
    """
    batch_replace = []
    batch_remove = []
    batch_insert = []
    for i, op in circuit.findall_operations(
        lambda o: isinstance(o.untagged, circuits.CircuitOperation)
    ):
        op_untagged = cast(circuits.CircuitOperation, op.untagged)
        if deep:
            op_untagged = op_untagged.replace(
                circuit=unroll_circuit_op_greedy_earliest(
                    op_untagged.circuit, deep=deep, tags_to_check=tags_to_check
                )
            )
        if tags_to_check is None or set(tags_to_check).intersection(op.tags):
            batch_remove.append((i, op))
            batch_insert.append((i, op_untagged.mapped_circuit().all_operations()))
        elif deep:
            batch_replace.append((i, op, op_untagged.with_tags(*op.tags)))
    unrolled_circuit = circuit.unfreeze(copy=True)
    unrolled_circuit.batch_replace(batch_replace)
    unrolled_circuit.batch_remove(batch_remove)
    unrolled_circuit.batch_insert(batch_insert)
    return _to_target_circuit_type(unrolled_circuit, circuit)


def unroll_circuit_op_greedy_frontier(
    circuit: CIRCUIT_TYPE,
    *,
    deep: bool = False,
    tags_to_check: Optional[Sequence[Hashable]] = (MAPPED_CIRCUIT_OP_TAG,),
) -> CIRCUIT_TYPE:
    """Unrolls (tagged) `cirq.CircuitOperation`s by inserting operations inline at qubit frontier.

    Each matching `cirq.CircuitOperation` is replaced by inserting underlying operations using the
    `circuit.insert_at_frontier` method. The greedy approach attempts to reuse any available space
    in existing moments on the right of circuit_op before inserting new moments.

    Args:
        circuit: Input circuit to apply the transformations on. The input circuit is not mutated.
        deep: If true, the transformer primitive will be recursively applied to all circuits
            wrapped inside circuit operations.
        tags_to_check: If specified, only circuit operations tagged with one of the `tags_to_check`
            are unrolled.

    Returns:
        Copy of input circuit with (Tagged) CircuitOperation's expanded inline at qubit frontier.
    """
    unrolled_circuit = circuit.unfreeze(copy=True)
    frontier: Dict['cirq.Qid', int] = defaultdict(lambda: 0)
    idx = 0
    while idx < len(unrolled_circuit):
        for op in unrolled_circuit[idx].operations:
            # Don't touch stuff inserted by unrolling previous circuit ops.
            if not isinstance(op.untagged, circuits.CircuitOperation):
                continue
            if any(frontier[q] > idx for q in op.qubits):
                continue
            op_untagged = op.untagged
            if deep:
                op_untagged = op_untagged.replace(
                    circuit=unroll_circuit_op_greedy_frontier(
                        op_untagged.circuit, deep=deep, tags_to_check=tags_to_check
                    )
                )
            if tags_to_check is None or set(tags_to_check).intersection(op.tags):
                unrolled_circuit.clear_operations_touching(op.qubits, [idx])
                frontier = unrolled_circuit.insert_at_frontier(
                    op_untagged.mapped_circuit().all_operations(), idx, frontier
                )
            elif deep:
                unrolled_circuit.batch_replace([(idx, op, op_untagged.with_tags(*op.tags))])
        idx += 1
    return _to_target_circuit_type(unrolled_circuit, circuit)


def toggle_tags(circuit: CIRCUIT_TYPE, tags: Sequence[Hashable], *, deep: bool = False):
    """Toggles tags applied on each operation in the circuit, via `op.tags ^= tags`

    For every operations `op` in the input circuit, the tags on `op` are replaced by a symmetric
    difference of `op.tags` and `tags` -- this is useful in scenarios where you mark a small subset
    of operations with a specific tag and then toggle the set of marked operations s.t. every
    marked operation is now unmarked and vice versa.

    Often used in transformer workflows to apply a transformer on a small subset of operations.

    Args:
        circuit: Input circuit to apply the transformations on. The input circuit is not mutated.
        tags: Sequence of tags s.t. `op.tags ^= tags` is done for every operation `op` in circuit.
        deep: If true, tags will be recursively toggled for operations in circuits wrapped inside
            any circuit operations contained within `circuit`.

    Returns:
        Copy of transformed input circuit with operation sets marked with `tags` toggled.
    """
    tags_to_xor = set(tags)

    def map_func(op: 'cirq.Operation', _) -> 'cirq.Operation':
        return (
            op
            if deep and isinstance(op, circuits.CircuitOperation)
            else op.untagged.with_tags(*(set(op.tags) ^ tags_to_xor))
        )

    return map_operations(circuit, map_func, deep=deep)
