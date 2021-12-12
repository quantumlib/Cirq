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
    map_func: Callable[[ops.Moment, int], Sequence[ops.Moment]],
) -> CIRCUIT_TYPE:
    """Applies local transformation on moments, by calling `map_func(moment)` for each moment.

    Args:
        circuit: Input circuit to apply the transformations on. The input circuit is not mutated.
        map_func: Mapping function from (cirq.Moment, moment_index) to a sequence of moments.

    Returns:
        Copy of input circuit with mapped moments.
    """
    return _create_target_circuit_type(
        (map_func(circuit[i], i) for i in range(len(circuit))), circuit
    )


def map_operations(
    circuit: CIRCUIT_TYPE,
    map_func: Callable[[ops.Operation, int], ops.OP_TREE],
) -> CIRCUIT_TYPE:
    """Applies local transformations on operations, by calling `map_func(op)` for each op.

    Note that the function assumes `issubset(qubit_set(map_func(op)), op.qubits)` is True.

    Args:
        circuit: Input circuit to apply the transformations on. The input circuit is not mutated.
        map_func: Mapping function from (cirq.Operation, moment_index) to a cirq.OP_TREE. If the
            resulting optree spans more than 1 moment, it's inserted in-place in the same moment as
            `cirq.CircuitOperation(cirq.FrozenCircuit(op_tree)).with_tags(MAPPED_CIRCUIT_OP_TAG)`
            to preserve moment structure. Utility methods like `cirq.unroll_circuit_op` can
            subsequently be used to unroll the mapped circuit operation.

    Raises:
          ValueError if `issubset(qubit_set(map_func(op)), op.qubits) is False`.

    Returns:
        Copy of input circuit with mapped operations (wrapped in a tagged CircuitOperation).
    """

    def apply_map(op: ops.Operation, idx: int) -> ops.OP_TREE:
        c = circuits.FrozenCircuit(map_func(op, idx))
        if not c.all_qubits().issubset(op.qubits):
            raise ValueError(
                f"Mapped operations {c.all_operations()} should act on a subset "
                f"of qubits of the original operation {op}"
            )
        if len(c) == 1:
            # All operations act in the same moment; so we don't need to wrap them in a circuit_op.
            return c[0].operations
        circuit_op = circuits.CircuitOperation(c).with_tags(MAPPED_CIRCUIT_OP_TAG)
        return circuit_op

    return map_moments(circuit, lambda m, i: [ops.Moment(apply_map(op, i) for op in m.operations)])


def map_operations_and_unroll(
    circuit: CIRCUIT_TYPE,
    map_func: Callable[[ops.Operation, int], ops.OP_TREE],
) -> CIRCUIT_TYPE:
    """Applies local transformations via `cirq.map_operations` & unrolls intermediate circuit ops.

    See `cirq.map_operations` and `cirq.unroll_circuit_op` for more details.

    Args:
        circuit: Input circuit to apply the transformations on. The input circuit is not mutated.
        map_func: Mapping function from (cirq.Operation, moment_index) to a cirq.OP_TREE.

    Returns:
        Copy of input circuit with mapped operations, unrolled in a moment preserving way.
    """
    return unroll_circuit_op(map_operations(circuit, map_func))


def merge_operations(
    circuit: CIRCUIT_TYPE,
    merge_func: Callable[[ops.Operation, ops.Operation], Optional[ops.Operation]],
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

    Returns:
        Copy of input circuit with merged operations.

    Raises:
        ValueError if the merged operation acts on new qubits outside the set of qubits
            corresponding to the original operations to be merged.
    """

    def apply_merge_func(op1: ops.Operation, op2: ops.Operation) -> Optional[ops.Operation]:
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
        new_moment = ops.Moment()
        for op in current_moment:
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


def merge_moments(
    circuit: CIRCUIT_TYPE,
    merge_func: Callable[[ops.Moment, ops.Moment], Optional[ops.Moment]],
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
    merged_moments: List[ops.Moment] = [circuit[0]]
    for current_moment in circuit[1:]:
        merged_moment = merge_func(merged_moments[-1], current_moment)
        if not merged_moment:
            merged_moments.append(current_moment)
        else:
            merged_moments[-1] = merged_moment
    return _create_target_circuit_type(merged_moments, circuit)


def _check_circuit_op(op, tags_to_check: Optional[Sequence[Hashable]]):
    return isinstance(op.untagged, circuits.CircuitOperation) and (
        tags_to_check is None or any(tag in op.tags for tag in tags_to_check)
    )


def unroll_circuit_op(
    circuit: CIRCUIT_TYPE, *, tags_to_check: Optional[Sequence[Hashable]] = (MAPPED_CIRCUIT_OP_TAG,)
) -> CIRCUIT_TYPE:
    """Unrolls (tagged) `cirq.CircuitOperation`s while preserving the moment structure.

    Each moment containing a matching circuit operation is expanded into a list of moments with the
    unrolled operations, hence preserving the original moment structure.

    Args:
        circuit: Input circuit to apply the transformations on. The input circuit is not mutated.
        tags_to_check: If specified, only circuit operations tagged with one of the `tags_to_check`
            are unrolled.

    Returns:
        Copy of input circuit with (Tagged) CircuitOperation's expanded in a moment preserving way.
    """

    def map_func(m: ops.Moment, _: int):
        to_zip = [
            cast(circuits.CircuitOperation, op.untagged).mapped_circuit()
            if _check_circuit_op(op, tags_to_check)
            else circuits.Circuit(op)
            for op in m
        ]
        return circuits.Circuit.zip(*to_zip).moments

    return map_moments(circuit, map_func)


def unroll_circuit_op_greedy_earliest(
    circuit: CIRCUIT_TYPE, *, tags_to_check=(MAPPED_CIRCUIT_OP_TAG,)
) -> CIRCUIT_TYPE:
    """Unrolls (tagged) `cirq.CircuitOperation`s by inserting operations using EARLIEST strategy.

    Each matching `cirq.CircuitOperation` is replaced by inserting underlying operations using the
    `cirq.InsertStrategy.EARLIEST` strategy. The greedy approach attempts to minimize circuit depth
    of the resulting circuit.

    Args:
        circuit: Input circuit to apply the transformations on. The input circuit is not mutated.
        tags_to_check: If specified, only circuit operations tagged with one of the `tags_to_check`
            are unrolled.

    Returns:
        Copy of input circuit with (Tagged) CircuitOperation's expanded using EARLIEST strategy.
    """
    batch_removals = [*circuit.findall_operations(lambda op: _check_circuit_op(op, tags_to_check))]
    batch_inserts = [(i, protocols.decompose_once(op)) for i, op in batch_removals]
    unrolled_circuit = circuit.unfreeze(copy=True)
    unrolled_circuit.batch_remove(batch_removals)
    unrolled_circuit.batch_insert(batch_inserts)
    return _to_target_circuit_type(unrolled_circuit, circuit)


def unroll_circuit_op_greedy_frontier(
    circuit: CIRCUIT_TYPE, *, tags_to_check=(MAPPED_CIRCUIT_OP_TAG,)
) -> CIRCUIT_TYPE:
    """Unrolls (tagged) `cirq.CircuitOperation`s by inserting operations inline at qubit frontier.

    Each matching `cirq.CircuitOperation` is replaced by inserting underlying operations using the
    `circuit.insert_at_frontier` method. The greedy approach attempts to reuse any available space
    in existing moments on the right of circuit_op before inserting new moments.

    Args:
        circuit: Input circuit to apply the transformations on. The input circuit is not mutated.
        tags_to_check: If specified, only circuit operations tagged with one of the `tags_to_check`
            are unrolled.

    Returns:
        Copy of input circuit with (Tagged) CircuitOperation's expanded inline at qubit frontier.
    """
    unrolled_circuit = circuit.unfreeze(copy=True)
    frontier: Dict['cirq.Qid', int] = defaultdict(lambda: 0)
    for idx, op in circuit.findall_operations(lambda op: _check_circuit_op(op, tags_to_check)):
        idx = max(idx, max(frontier[q] for q in op.qubits))
        unrolled_circuit.clear_operations_touching(op.qubits, [idx])
        frontier = unrolled_circuit.insert_at_frontier(protocols.decompose_once(op), idx, frontier)
    return _to_target_circuit_type(unrolled_circuit, circuit)
