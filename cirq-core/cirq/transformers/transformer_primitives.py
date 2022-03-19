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
from typing import (
    cast,
    Callable,
    Dict,
    Hashable,
    List,
    Optional,
    Sequence,
    Union,
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
    deep: bool = False,
) -> CIRCUIT_TYPE:
    """Applies local transformation on moments, by calling `map_func(moment)` for each moment.

    Args:
        circuit: Input circuit to apply the transformations on. The input circuit is not mutated.
        map_func: Mapping function from (cirq.Moment, moment_index) to a sequence of moments.
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
            op_untagged = cast(circuits.CircuitOperation, op.untagged)
            mapped_op = op_untagged.replace(
                circuit=map_moments(op_untagged.circuit, map_func, deep=deep)
            ).with_tags(*op.tags)
            batch_replace.append((i, op, mapped_op))
        mutable_circuit = circuit.unfreeze(copy=True)
        mutable_circuit.batch_replace(batch_replace)
    return _create_target_circuit_type(
        (map_func(mutable_circuit[i], i) for i in range(len(mutable_circuit))), circuit
    )


def map_operations(
    circuit: CIRCUIT_TYPE,
    map_func: Callable[[ops.Operation, int], ops.OP_TREE],
    *,
    deep: bool = False,
    raise_if_add_qubits=True,
    tags_to_ignore: Sequence[Hashable] = (),
) -> CIRCUIT_TYPE:
    """Applies local transformations on operations, by calling `map_func(op)` for each op.

    By default, the function assumes `issubset(qubit_set(map_func(op)), op.qubits)` is True.

    Args:
        circuit: Input circuit to apply the transformations on. The input circuit is not mutated.
        map_func: Mapping function from (cirq.Operation, moment_index) to a cirq.OP_TREE. If the
            resulting optree spans more than 1 moment, it's inserted in-place in the same moment as
            `cirq.CircuitOperation(cirq.FrozenCircuit(op_tree)).with_tags(MAPPED_CIRCUIT_OP_TAG)`
            to preserve moment structure. Utility methods like `cirq.unroll_circuit_op` can
            subsequently be used to unroll the mapped circuit operation.
        deep: If true, `map_func` will be recursively applied to circuits wrapped inside
            any circuit operations contained within `circuit`.
        raise_if_add_qubits: Set to True by default. If True, raises ValueError if `map_func(op)`
            adds operations on qubits outside of `op.qubits`.
        tags_to_ignore: Sequence of tags which should be ignored while applying `map_func` on
            tagged operations -- i.e. `map_func(op, idx)` will be called only for operations that
            satisfy `set(op.tags).isdisjoint(tags_to_ignore)`.

    Raises:
          ValueError if `issubset(qubit_set(map_func(op)), op.qubits) is False` and
            `raise_if_add_qubits is True`.

    Returns:
        Copy of input circuit with mapped operations (wrapped in a tagged CircuitOperation).
    """

    def apply_map(op: ops.Operation, idx: int) -> ops.OP_TREE:
        if not set(op.tags).isdisjoint(tags_to_ignore):
            return op
        c = circuits.FrozenCircuit(map_func(op, idx))
        if raise_if_add_qubits and not c.all_qubits().issubset(op.qubits):
            raise ValueError(
                f"Mapped operations {c.all_operations()} should act on a subset "
                f"of qubits of the original operation {op}"
            )
        if len(c) <= 1:
            # Either empty circuit or all operations act in the same moment;
            # So, we don't need to wrap them in a circuit_op.
            return c[0].operations if c else []
        circuit_op = circuits.CircuitOperation(c).with_tags(MAPPED_CIRCUIT_OP_TAG)
        return circuit_op

    return map_moments(
        circuit, lambda m, i: [circuits.Moment(apply_map(op, i) for op in m.operations)], deep=deep
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
        raise_if_add_qubits: Set to True by default. If True, raises ValueError if `map_func(op)`
            adds operations on qubits outside of `op.qubits`.
        tags_to_ignore: Sequence of tags which should be ignored while applying `map_func` on
            tagged operations -- i.e. `map_func(op, idx)` will be called only for operations that
            satisfy `set(op.tags).isdisjoint(tags_to_ignore)`.

    Returns:
        Copy of input circuit with mapped operations, unrolled in a moment preserving way.
    """
    return unroll_circuit_op(
        map_operations(
            circuit,
            map_func,
            deep=deep,
            raise_if_add_qubits=raise_if_add_qubits,
            tags_to_ignore=tags_to_ignore,
        ),
        deep=deep,
    )


def merge_operations(
    circuit: CIRCUIT_TYPE,
    merge_func: Callable[[ops.Operation, ops.Operation], Optional[ops.Operation]],
    *,
    tags_to_ignore: Sequence[Hashable] = (),
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
    operations to it's left.

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


    Returns:
        Copy of input circuit with merged operations.

    Raises:
        ValueError if the merged operation acts on new qubits outside the set of qubits
            corresponding to the original operations to be merged.
    """

    def apply_merge_func(op1: ops.Operation, op2: ops.Operation) -> Optional[ops.Operation]:
        if not all(set(op.tags).isdisjoint(tags_to_ignore) for op in [op1, op2]):
            return None
        new_op = merge_func(op1, op2)
        qubit_set = frozenset(op1.qubits + op2.qubits)
        if new_op is not None and not qubit_set.issuperset(new_op.qubits):
            raise ValueError(
                f"Merged operation {new_op} must act on a subset of qubits of "
                f"original operations {op1} and {op2}"
            )
        return new_op

    ret_circuit = circuits.Circuit()
    for current_moment in circuit:
        new_moment = circuits.Moment()
        for op in sorted(current_moment.operations, key=lambda op: op.qubits):
            op_qs = set(op.qubits)
            idx = ret_circuit.prev_moment_operating_on(tuple(op_qs))
            if idx is not None and op_qs.issubset(ret_circuit[idx][op_qs].operations[0].qubits):
                # Case-1: Try to merge op with the larger operation on the left.
                left_op = ret_circuit[idx][op_qs].operations[0]
                new_op = apply_merge_func(left_op, op)
                if new_op is not None:
                    ret_circuit.batch_replace([(idx, left_op, new_op)])
                else:
                    new_moment = new_moment.with_operation(op)
                continue

            while idx is not None and len(op_qs) > 0:
                # Case-2: left_ops will merge right into `op` whenever possible.
                for left_op in ret_circuit[idx][op_qs].operations:
                    is_merged = False
                    if op_qs.issuperset(left_op.qubits):
                        # Try to merge left_op into op
                        new_op = apply_merge_func(left_op, op)
                        if new_op is not None:
                            ret_circuit.batch_remove([(idx, left_op)])
                            op, is_merged = new_op, True
                    if not is_merged:
                        op_qs -= frozenset(left_op.qubits)
                idx = ret_circuit.prev_moment_operating_on(tuple(op_qs))
            new_moment = new_moment.with_operation(op)
        ret_circuit += new_moment
    return _to_target_circuit_type(ret_circuit, circuit)


def merge_operations_to_circuit_op(
    circuit: CIRCUIT_TYPE,
    can_merge: Callable[[Sequence['cirq.Operation'], Sequence['cirq.Operation']], bool],
    *,
    tags_to_ignore: Sequence[Hashable] = (),
    merged_circuit_op_tag: str = "Merged connected component",
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

    return merge_operations(circuit, merge_func, tags_to_ignore=tags_to_ignore)


def merge_k_qubit_unitaries_to_circuit_op(
    circuit: CIRCUIT_TYPE,
    k: int,
    *,
    tags_to_ignore: Sequence[Hashable] = (),
    merged_circuit_op_tag: Optional[str] = None,
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
    )


def merge_moments(
    circuit: CIRCUIT_TYPE,
    merge_func: Callable[[circuits.Moment, circuits.Moment], Optional[circuits.Moment]],
) -> CIRCUIT_TYPE:
    """Merges adjacent moments, one by one from left to right, by calling `merge_func(m1, m2)`.

    Args:
        circuit: Input circuit to apply the transformations on. The input circuit is not mutated.
        merge_func: Callable to determine whether two adjacent moments in the circuit should be
            merged. If the moments can be merged, the callable should return the merged moment,
            else None.

    Returns:
        Copy of input circuit with merged moments.
    """
    if not circuit:
        return circuit
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
            op_untagged = cast(circuits.CircuitOperation, op.untagged)
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
