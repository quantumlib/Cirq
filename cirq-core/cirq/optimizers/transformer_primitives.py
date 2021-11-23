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
