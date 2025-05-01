# Copyright 2024 The Cirq Developers
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

"""Transformer pass that adds dynamical decoupling operations to a circuit."""

from __future__ import annotations

from functools import reduce
from itertools import cycle
from typing import Dict, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np

from cirq import circuits, ops, protocols
from cirq.protocols import unitary_protocol
from cirq.protocols.has_stabilizer_effect_protocol import has_stabilizer_effect
from cirq.protocols.has_unitary_protocol import has_unitary
from cirq.transformers import transformer_api
from cirq.transformers.analytical_decompositions import single_qubit_decompositions

if TYPE_CHECKING:
    import cirq


def _get_dd_sequence_from_schema_name(schema: str) -> Tuple[ops.Gate, ...]:
    """Gets dynamical decoupling sequence from a schema name."""
    match schema:
        case 'DEFAULT':
            return (ops.X, ops.Y, ops.X, ops.Y)
        case 'XX_PAIR':
            return (ops.X, ops.X)
        case 'X_XINV':
            return (ops.X, ops.X**-1)
        case 'YY_PAIR':
            return (ops.Y, ops.Y)
        case 'Y_YINV':
            return (ops.Y, ops.Y**-1)
        case _:
            raise ValueError('Invalid schema name.')


def _pauli_up_to_global_phase(gate: ops.Gate) -> Union[ops.Pauli, None]:
    for pauli_gate in [ops.X, ops.Y, ops.Z]:
        if protocols.equal_up_to_global_phase(gate, pauli_gate):
            return pauli_gate
    return None


def _validate_dd_sequence(dd_sequence: Tuple[ops.Gate, ...]) -> None:
    """Validates a given dynamical decoupling sequence.

    The sequence should only consists of Pauli gates and is essentially an identity gate.

    Args:
        dd_sequence: Input dynamical sequence to be validated.

    Raises:
        ValueError: If dd_sequence is not valid.
    """
    if len(dd_sequence) < 2:
        raise ValueError('Invalid dynamical decoupling sequence. Expect more than one gates.')
    for gate in dd_sequence:
        if _pauli_up_to_global_phase(gate) is None:
            raise ValueError(
                'Dynamical decoupling sequence should only contain gates that are essentially'
                ' Pauli gates.'
            )
    matrices = [unitary_protocol.unitary(gate) for gate in dd_sequence]
    product = reduce(np.matmul, matrices)

    if not protocols.equal_up_to_global_phase(product, np.eye(2)):
        raise ValueError(
            'Invalid dynamical decoupling sequence. Expect sequence production equals'
            f' identity up to a global phase, got {product}.'.replace('\n', ' ')
        )


def _parse_dd_sequence(
    schema: Union[str, Tuple[ops.Gate, ...]],
) -> Tuple[Tuple[ops.Gate, ...], Dict[ops.Gate, ops.Pauli]]:
    """Parses and returns dynamical decoupling sequence and its associated pauli map from schema."""
    dd_sequence = None
    if isinstance(schema, str):
        dd_sequence = _get_dd_sequence_from_schema_name(schema)
    else:
        _validate_dd_sequence(schema)
        dd_sequence = schema

    # Map gate to Pauli gate. This is necessary as dd sequence might contain gates like X^-1.
    pauli_map: Dict[ops.Gate, ops.Pauli] = {}
    for gate in dd_sequence:
        pauli_gate = _pauli_up_to_global_phase(gate)
        if pauli_gate is not None:
            pauli_map[gate] = pauli_gate
    for gate in [ops.X, ops.Y, ops.Z]:
        pauli_map[gate] = gate

    return (dd_sequence, pauli_map)


def _is_single_qubit_operation(operation: ops.Operation) -> bool:
    return len(operation.qubits) == 1


def _is_single_qubit_gate_moment(moment: circuits.Moment) -> bool:
    return all(_is_single_qubit_operation(op) for op in moment)


def _is_clifford_op(op: ops.Operation) -> bool:
    return has_unitary(op) and has_stabilizer_effect(op)


def _calc_busy_moment_range_of_each_qubit(
    circuit: circuits.FrozenCircuit,
) -> Dict[ops.Qid, list[int]]:
    busy_moment_range_by_qubit: Dict[ops.Qid, list[int]] = {
        q: [len(circuit), -1] for q in circuit.all_qubits()
    }
    for moment_id, moment in enumerate(circuit):
        for q in moment.qubits:
            busy_moment_range_by_qubit[q][0] = min(busy_moment_range_by_qubit[q][0], moment_id)
            busy_moment_range_by_qubit[q][1] = max(busy_moment_range_by_qubit[q][1], moment_id)
    return busy_moment_range_by_qubit


def _is_insertable_moment(moment: circuits.Moment, single_qubit_gate_moments_only: bool) -> bool:
    return not single_qubit_gate_moments_only or _is_single_qubit_gate_moment(moment)


def _merge_single_qubit_ops_to_phxz(
    q: ops.Qid, operations: Tuple[ops.Operation, ...]
) -> ops.Operation:
    """Merges [op1, op2, ...] and returns an equivalent op"""
    if len(operations) == 1:
        return operations[0]
    matrices = [unitary_protocol.unitary(op) for op in reversed(operations)]
    product = reduce(np.matmul, matrices)
    gate = single_qubit_decompositions.single_qubit_matrix_to_phxz(product) or ops.I
    return gate.on(q)


def _try_merge_single_qubit_ops_of_two_moments(
    m1: circuits.Moment, m2: circuits.Moment
) -> Tuple[circuits.Moment, ...]:
    """Merge single qubit ops of 2 moments if possible, returns 2 moments otherwise."""
    for q in m1.qubits & m2.qubits:
        op1 = m1.operation_at(q)
        op2 = m2.operation_at(q)
        if any(
            not (_is_single_qubit_operation(op) and has_unitary(op))
            for op in [op1, op2]
            if op is not None
        ):
            return (m1, m2)
    merged_ops: set[ops.Operation] = set()
    # Merge all operators on q to a single op.
    for q in m1.qubits | m2.qubits:
        # ops_on_q may contain 1 op or 2 ops.
        ops_on_q = [op for op in [m.operation_at(q) for m in [m1, m2]] if op is not None]
        merged_ops.add(_merge_single_qubit_ops_to_phxz(q, tuple(ops_on_q)))
    return (circuits.Moment(merged_ops),)


def _calc_pulled_through(
    moment: circuits.Moment, input_pauli_ops: ops.PauliString
) -> ops.PauliString:
    """Calculates the pulled_through such that circuit(input_pauli_ops, moment.clifford_ops) is
    equivalent to circuit(moment.clifford_ops, pulled_through).
    """
    clifford_ops_in_moment: list[ops.Operation] = [
        op for op in moment.operations if _is_clifford_op(op)
    ]
    return input_pauli_ops.after(clifford_ops_in_moment)


def _get_stop_qubits(moment: circuits.Moment) -> set[ops.Qid]:
    stop_pulling_through_qubits: set[ops.Qid] = set()
    for op in moment:
        if (not _is_clifford_op(op) and not _is_single_qubit_operation(op)) or not has_unitary(
            op
        ):  # multi-qubit clifford op or non-mergable op.
            stop_pulling_through_qubits.update(op.qubits)
    return stop_pulling_through_qubits


def _need_merge_pulled_through(op_at_q: ops.Operation, is_at_last_busy_moment: bool) -> bool:
    """With a pulling through pauli gate before op_at_q, need to merge with the
    pauli in the conditions below."""
    # The op must be mergable and single-qubit
    if not (_is_single_qubit_operation(op_at_q) and has_unitary(op_at_q)):
        return False
    # Either non-Clifford or at the last busy moment
    return is_at_last_busy_moment or not _is_clifford_op(op_at_q)


@transformer_api.transformer
def add_dynamical_decoupling(
    circuit: cirq.AbstractCircuit,
    *,
    context: Optional[cirq.TransformerContext] = None,
    schema: Union[str, Tuple[ops.Gate, ...]] = 'DEFAULT',
    single_qubit_gate_moments_only: bool = True,
) -> cirq.Circuit:
    """Adds dynamical decoupling gate operations to a given circuit.
    This transformer might add new moments thus change structure of the original circuit.

    Args:
          circuit: Input circuit to transform.
          context: `cirq.TransformerContext` storing common configurable options for transformers.
          schema: Dynamical decoupling schema name or a dynamical decoupling sequence.
            If a schema is specified, provided dynamical decouping sequence will be used.
            Otherwise, customized dynamical decoupling sequence will be applied.
          single_qubit_gate_moments_only: If set True, dynamical decoupling operation will only be
            added in single-qubit gate moments.

    Returns:
          A copy of the input circuit with dynamical decoupling operations.
    """
    base_dd_sequence, pauli_map = _parse_dd_sequence(schema)
    orig_circuit = circuit.freeze()

    busy_moment_range_by_qubit = _calc_busy_moment_range_of_each_qubit(orig_circuit)

    # Stores all the moments of the output circuit chronologically.
    transformed_moments: list[circuits.Moment] = []
    # A PauliString stores the result of 'pulling' Pauli gates past each operations
    # right before the current moment.
    pulled_through: ops.PauliString = ops.PauliString()
    # Iterator of gate to be used in dd sequence for each qubit.
    dd_iter_by_qubits = {q: cycle(base_dd_sequence) for q in circuit.all_qubits()}

    def _update_pulled_through(q: ops.Qid, insert_gate: ops.Gate) -> ops.Operation:
        nonlocal pulled_through, pauli_map
        pulled_through *= pauli_map[insert_gate].on(q)
        return insert_gate.on(q)

    # Insert and pull remaining Pauli ops through the whole circuit.
    # General ideas are
    #   * Pull through Clifford gates.
    #   * Stop at multi-qubit non-Clifford ops (and other non-mergable ops).
    #   * Merge to single-qubit non-Clifford ops.
    #   * Insert a new moment if necessary.
    # After pulling through pulled_through at `moment`, we expect a transformation of
    #  (pulled_through, moment) -> (updated_moment, updated_pulled_through) or
    #  (pulled_through, moment) -> (new_moment, updated_moment, updated_pulled_through)
    # Moments structure changes are split into 3 steps:
    #   1, (..., last_moment, pulled_through1, moment, ...)
    #        -> (..., try_merge(last_moment, new_moment or None), pulled_through2, moment, ...)
    #   2, (..., pulled_through2, moment, ...)  -> (..., pulled_through3, updated_moment, ...)
    #   3, (..., pulled_through3, updated_moment, ...)
    #        -> (..., updated_moment, pulled_through4, ...)
    for moment_id, moment in enumerate(orig_circuit.moments):
        # Step 1, insert new_moment if necessary.
        # In detail: stop pulling through for multi-qubit non-Clifford ops or gates without
        # unitary representation (e.g., measure gates). If there are remaining pulled through ops,
        # insert into a new moment before current moment.
        stop_pulling_through_qubits: set[ops.Qid] = _get_stop_qubits(moment)
        new_moment_ops = []
        for q in stop_pulling_through_qubits:
            # Insert the remaining pulled_through
            remaining_pulled_through_gate = pulled_through.get(q)
            if remaining_pulled_through_gate is not None:
                new_moment_ops.append(_update_pulled_through(q, remaining_pulled_through_gate))
            # Reset dd sequence
            dd_iter_by_qubits[q] = cycle(base_dd_sequence)
        # Need to insert a new moment before current moment
        if new_moment_ops:
            # Fill insertable idle moments in the new moment using dd sequence
            for q in orig_circuit.all_qubits() - stop_pulling_through_qubits:
                if busy_moment_range_by_qubit[q][0] < moment_id <= busy_moment_range_by_qubit[q][1]:
                    new_moment_ops.append(_update_pulled_through(q, next(dd_iter_by_qubits[q])))
            moments_to_be_appended = _try_merge_single_qubit_ops_of_two_moments(
                transformed_moments.pop(), circuits.Moment(new_moment_ops)
            )
            transformed_moments.extend(moments_to_be_appended)

        # Step 2, calc updated_moment with insertions / merges.
        updated_moment_ops: set[cirq.Operation] = set()
        for q in orig_circuit.all_qubits():
            op_at_q = moment.operation_at(q)
            remaining_pulled_through_gate = pulled_through.get(q)
            updated_op = op_at_q
            if op_at_q is None:  # insert into idle op
                if not _is_insertable_moment(moment, single_qubit_gate_moments_only):
                    continue
                if (
                    busy_moment_range_by_qubit[q][0] < moment_id < busy_moment_range_by_qubit[q][1]
                ):  # insert next pauli gate in the dd sequence
                    updated_op = _update_pulled_through(q, next(dd_iter_by_qubits[q]))
                elif (  # insert the remaining pulled through if beyond the ending busy moment
                    moment_id > busy_moment_range_by_qubit[q][1]
                    and remaining_pulled_through_gate is not None
                ):
                    updated_op = _update_pulled_through(q, remaining_pulled_through_gate)
            elif (
                remaining_pulled_through_gate is not None
            ):  # merge pulled-through of q to op_at_q if needed
                if _need_merge_pulled_through(
                    op_at_q, moment_id == busy_moment_range_by_qubit[q][1]
                ):
                    remaining_op = _update_pulled_through(q, remaining_pulled_through_gate)
                    updated_op = _merge_single_qubit_ops_to_phxz(q, (remaining_op, op_at_q))
            if updated_op is not None:
                updated_moment_ops.add(updated_op)

        if updated_moment_ops:
            updated_moment = circuits.Moment(updated_moment_ops)
            transformed_moments.append(updated_moment)

            # Step 3, update pulled through.
            # In detail: pulling current `pulled_through` through updated_moment.
            pulled_through = _calc_pulled_through(updated_moment, pulled_through)

    # Insert a new moment if there are remaining pulled-through operations.
    ending_moment_ops = []
    for affected_q, combined_op_in_pauli in pulled_through.items():
        ending_moment_ops.append(combined_op_in_pauli.on(affected_q))
    if ending_moment_ops:
        transformed_moments.extend(
            _try_merge_single_qubit_ops_of_two_moments(
                transformed_moments.pop(), circuits.Moment(ending_moment_ops)
            )
        )

    return circuits.Circuit(transformed_moments)
