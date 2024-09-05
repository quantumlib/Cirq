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

from functools import reduce
from typing import Dict, Optional, Tuple, Union
from itertools import cycle

from cirq.transformers import transformer_api
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
from cirq.transformers.analytical_decompositions import unitary_to_pauli_string
import cirq
import numpy as np


def _get_dd_sequence_from_schema_name(schema: str) -> Tuple['cirq.Gate', ...]:
    """Gets dynamical decoupling sequence from a schema name."""
    match schema:
        case 'DEFAULT':
            return (cirq.X, cirq.Y, cirq.X, cirq.Y)
        case 'XX_PAIR':
            return (cirq.X, cirq.X)
        case 'X_XINV':
            return (cirq.X, cirq.X**-1)
        case 'YY_PAIR':
            return (cirq.Y, cirq.Y)
        case 'Y_YINV':
            return (cirq.Y, cirq.Y**-1)
        case _:
            raise ValueError('Invalid schema name.')


def _pauli_up_to_global_phase(gate: 'cirq.Gate') -> Union['cirq.Pauli', None]:
    for pauli_gate in [cirq.X, cirq.Y, cirq.Z]:
        if cirq.equal_up_to_global_phase(gate, pauli_gate):
            return pauli_gate
    return None


def _validate_dd_sequence(dd_sequence: Tuple['cirq.Gate', ...]) -> None:
    """Validates a given dynamical decoupling sequence.

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
    matrices = [cirq.unitary(gate) for gate in dd_sequence]
    product = reduce(np.matmul, matrices)

    if not cirq.equal_up_to_global_phase(product, np.eye(2)):
        raise ValueError(
            'Invalid dynamical decoupling sequence. Expect sequence production equals'
            f' identity up to a global phase, got {product}.'.replace('\n', ' ')
        )


def _parse_dd_sequence(
    schema: Union[str, Tuple['cirq.Gate', ...]]
) -> Tuple[Tuple['cirq.Gate', ...], Dict['cirq.Gate', 'cirq.Pauli']]:
    """Parses and returns dynamical decoupling sequence and its associated pauli map from schema."""
    dd_sequence = None
    if isinstance(schema, str):
        dd_sequence = _get_dd_sequence_from_schema_name(schema)
    else:
        _validate_dd_sequence(schema)
        dd_sequence = schema

    # Map Gate to Puali gate.
    pauli_map: Dict['cirq.Gate', 'cirq.Pauli'] = {}
    for gate in dd_sequence:
        pauli_gate = _pauli_up_to_global_phase(gate)
        if pauli_gate is not None:
            pauli_map[gate] = pauli_gate

    return (dd_sequence, pauli_map)


def _is_single_qubit_operation(operation: 'cirq.Operation') -> bool:
    if len(operation.qubits) != 1:
        return False
    return True


def _is_single_qubit_gate_moment(moment: 'cirq.Moment') -> bool:
    for operation in moment:
        if not _is_single_qubit_operation(operation):
            return False
    return True


def _is_clifford_moment(moment: 'cirq.Moment') -> bool:
    for op in moment.operations:
        if op.gate is not None and isinstance(op.gate, cirq.MeasurementGate):
            return False
        if not cirq.has_stabilizer_effect(op):
            return False
    return True


def _get_clifford_pieces(circuit: 'cirq.AbstractCircuit') -> list[Tuple[int, int]]:
    """Returns all clifford pieces [l, r] for a Circuit."""
    clifford_pieces: list[Tuple[int, int]] = []
    left = 0
    for moment_id, moment in enumerate(circuit):
        if not _is_clifford_moment(moment):
            if moment_id > left:
                clifford_pieces.append((left, moment_id - 1))
            left = moment_id + 1
    if left < len(circuit):
        clifford_pieces.append((left, len(circuit) - 1))
    return clifford_pieces


def _is_insertable_moment(moment: 'cirq.Moment', single_qubit_gate_moments_only: bool) -> bool:
    return _is_single_qubit_gate_moment(moment) or not single_qubit_gate_moments_only


def _merge_single_qubit_ops_to_phxz(
    q: 'cirq.Qid', ops: Tuple['cirq.Operation', ...]
) -> 'cirq.Operation':
    """Merges [op1, op2, ...] and returns an equivalent op"""
    matrices = [cirq.unitary(op) for op in reversed(ops)]
    product = reduce(np.matmul, matrices)
    gate: Optional['cirq.Gate'] = single_qubit_decompositions.single_qubit_matrix_to_phxz(product)
    if gate is None:
        # gate is close to Identity.
        gate = cirq.I
    return gate.on(q)


def _calc_pulled_through(
    moment: 'cirq.Moment', input_pauli_ops: 'cirq.PauliString'
) -> 'cirq.PauliString':
    """Calculates the pulled_through after pulling through moment with the input.

    We assume that the moment is Clifford here. Then, pulling through is essentially
      decomposing a matrix into Pauli operations on each qubit.
    """
    pulled_through: 'cirq.PauliString' = cirq.PauliString()
    for affected_q, combined_gate_in_pauli in input_pauli_ops.items():
        op_at_moment = moment.operation_at(affected_q)
        if op_at_moment is None:
            pulled_through *= combined_gate_in_pauli.on(affected_q)
            continue
        prev_circuit = cirq.Circuit(cirq.Moment(op_at_moment))
        new_circuit = cirq.Circuit(
            cirq.Moment(combined_gate_in_pauli.on(affected_q)), cirq.Moment(op_at_moment)
        )
        qubit_order = op_at_moment.qubits
        pulled_through_pauli_ops = unitary_to_pauli_string(
            prev_circuit.unitary(qubit_order=qubit_order)
            @ new_circuit.unitary(qubit_order=qubit_order).conj().T
        )
        if pulled_through_pauli_ops is not None:
            for qid, gate in enumerate(pulled_through_pauli_ops):
                pulled_through *= gate.on(qubit_order[qid])
    return pulled_through


def _process_pulled_through(
    circuit: 'cirq.FrozenCircuit',
    pulled_through: 'cirq.PauliString',
    clifford_piece_range: Tuple[int, int],
    single_qubit_gate_moments_only: bool,
) -> Tuple[
    'cirq.PauliString',
    list[Tuple[int, 'cirq.Operation']],
    list[Tuple[int, 'cirq.Operation', 'cirq.Operation']],
]:
    """Merges pulled-through Pauli gates into the last single-qubit gate operation or the insert it
      into the first idle moment if idle moments exist.

    Args:
        circuit: a frozen circuit where pulled-through gates will be inserted / merged.
        pulled_through: Pauli gates to be merged.
        clifford_piece_range: Specifies the [l, r] moments within which pulled-through gate merging
          is to be performed.
        single_qubit_gate_moments_only: If set True, dynamical decoupling operation will only be
            added in single-qubit gate moments.

    Returns:
        The remaining pulled-through operations after merging.
    """
    insert_intos: list[Tuple[int, 'cirq.Operation']] = []
    batch_replaces: list[Tuple[int, 'cirq.Operation', 'cirq.Operation']] = []
    remaining_pulled_through = pulled_through
    for affected_q, combined_gate_in_pauli in pulled_through.items():
        moment_id = circuit.prev_moment_operating_on([affected_q], clifford_piece_range[1] + 1)
        if moment_id is not None:
            op = circuit.operation_at(affected_q, moment_id)
            # Try to merge op into the last active moment if it is single-qubit gate operation.
            if op is not None and _is_single_qubit_operation(op):
                updated_op = _merge_single_qubit_ops_to_phxz(
                    affected_q, (op, combined_gate_in_pauli.on(affected_q))
                )
                batch_replaces.append((moment_id, op, updated_op))
                remaining_pulled_through *= combined_gate_in_pauli.on(affected_q)
                continue
            # Insert into the first empty moment for the qubit if such moment exists.
            while moment_id <= clifford_piece_range[1]:
                if affected_q not in circuit.moments[moment_id].qubits and _is_insertable_moment(
                    circuit.moments[moment_id], single_qubit_gate_moments_only
                ):
                    insert_intos.append((moment_id, combined_gate_in_pauli.on(affected_q)))
                    remaining_pulled_through *= combined_gate_in_pauli.on(affected_q)
                    break
                moment_id += 1
    return remaining_pulled_through, insert_intos, batch_replaces


def _fill_for_each_clifford_piece(
    circuit: 'cirq.FrozenCircuit',
    base_dd_sequence_info: Tuple[Tuple['cirq.Gate', ...], Dict['cirq.Gate', 'cirq.Pauli']],
    single_qubit_gate_moments_only: bool,
) -> 'cirq.Circuit':
    """For each Clifford piece, insert if idle, pull through if busy.
    Note cross Clifford pieces dd sequence will not be added in this function."""

    base_dd_sequence, pauli_map = base_dd_sequence_info
    busy_moment_range_by_qubit: Dict['cirq.Qid', list[int]] = {
        q: [len(circuit), -1] for q in circuit.all_qubits()
    }
    for moment_id, moment in enumerate(circuit):
        for q in moment.qubits:
            busy_moment_range_by_qubit[q][0] = min(busy_moment_range_by_qubit[q][0], moment_id)
            busy_moment_range_by_qubit[q][1] = max(busy_moment_range_by_qubit[q][1], moment_id)

    clifford_pieces = _get_clifford_pieces(circuit)
    insert_moments = []
    mutable_circuit = circuit.unfreeze(copy=True)
    for l, r in clifford_pieces:  # [l, r]
        # A PauliString stores the result of 'pulling' Pauli gates past each operations
        # right before the current moment.
        pulled_through: 'cirq.PauliString' = cirq.PauliString()
        # Iterator of gate to be used in dd sequence for each qubit.
        dd_iter_by_qubits = {q: cycle(base_dd_sequence) for q in circuit.all_qubits()}
        insert_intos: list[Tuple[int, 'cirq.Operation']] = []
        batch_replaces: list[Tuple[int, 'cirq.Operation', 'cirq.Operation']] = []

        # Iterate over the Clifford piece.
        for moment_id in range(l, r + 1):
            moment = circuit.moments[moment_id]

            # Insert
            if _is_insertable_moment(moment, single_qubit_gate_moments_only):
                for q in circuit.all_qubits() - moment.qubits:
                    if (
                        busy_moment_range_by_qubit[q][0]
                        < moment_id
                        < busy_moment_range_by_qubit[q][1]
                    ):
                        insert_gate = next(dd_iter_by_qubits[q])
                        insert_intos.append((moment_id, insert_gate.on(q)))
                        pulled_through *= pauli_map[insert_gate].on(q)

            # Pull through
            pulled_through = _calc_pulled_through(moment, pulled_through)

        # Need to insert before processing pulled through.
        mutable_circuit.batch_insert_into(insert_intos)

        # For the pulled-through gates, fill / merge if possible.
        remaining_pulled_through, insert_intos, batch_replaces = _process_pulled_through(
            mutable_circuit.freeze(), pulled_through, (l, r), single_qubit_gate_moments_only
        )
        mutable_circuit.batch_insert_into(insert_intos)
        mutable_circuit.batch_replace(batch_replaces)

        # Insert a new moment if there are remaining pulled-through operations.
        new_moment_ops = []
        for affected_q, combined_op_in_pauli in remaining_pulled_through.items():
            new_moment_ops.append(combined_op_in_pauli.on(affected_q))
        if len(new_moment_ops) != 0:
            insert_moments.append((r + 1, cirq.Moment(new_moment_ops)))
    mutable_circuit.batch_insert(insert_moments)
    return mutable_circuit


def _fill_consecutive_idle_moments_for_each_qubit(
    circuit: 'cirq.FrozenCircuit',
    base_dd_sequence_info: Tuple[Tuple['cirq.Gate', ...], Dict['cirq.Gate', 'cirq.Pauli']],
    single_qubit_gate_moments_only: bool,
) -> 'cirq.Circuit':
    insert_intos: list[Tuple[int, 'cirq.Operation']] = []
    batch_replaces: list[Tuple[int, 'cirq.Operation', 'cirq.Operation']] = []
    base_dd_sequence, pauli_map = base_dd_sequence_info
    for q in circuit.all_qubits():
        prev_moment_id = circuit.next_moment_operating_on([q], 0)
        next_moment_id = None
        if prev_moment_id is not None:
            next_moment_id = circuit.next_moment_operating_on([q], prev_moment_id + 1)
            prev_op = circuit.operation_at(q, prev_moment_id)

        # Iterate over all idle pieces of this qubit.
        while prev_moment_id is not None and next_moment_id is not None:
            next_op = circuit.operation_at(q, next_moment_id)
            if next_moment_id - prev_moment_id > 1:  # idle operations exist
                dd_iter = cycle(base_dd_sequence)
                if prev_op is not None and next_op is not None:
                    insertable_moment_ids = [
                        moment_id
                        for moment_id in range(prev_moment_id + 1, next_moment_id)
                        if _is_insertable_moment(
                            circuit.moments[moment_id], single_qubit_gate_moments_only
                        )
                    ]
                    # If the prev op or the next op is single-qubit gate op (mergeable).
                    #   1. Insert dd sequence into all idle moments.
                    #   2. Merge the remaining of the dd_sequence into either prev op or next op
                    #      depends on the availability.
                    if _is_single_qubit_operation(prev_op) or _is_single_qubit_operation(next_op):
                        to_be_merged: 'cirq.PauliString' = cirq.PauliString()
                        for moment_id in insertable_moment_ids:
                            gate = next(dd_iter)
                            insert_intos.append((moment_id, gate.on(q)))
                            to_be_merged *= pauli_map[gate].on(q)
                        for q, combined_gate_in_pauli in to_be_merged.items():
                            if _is_single_qubit_operation(next_op):  # Merge into the next op.
                                updated_op = _merge_single_qubit_ops_to_phxz(
                                    q, (combined_gate_in_pauli.on(q), next_op)
                                )
                                batch_replaces.append((next_moment_id, next_op, updated_op))
                            else:  # Merge into the prev op.
                                updated_op = _merge_single_qubit_ops_to_phxz(
                                    q, (prev_op, combined_gate_in_pauli.on(q))
                                )
                                batch_replaces.append((prev_moment_id, prev_op, updated_op))
                    # Otherwise, insert whole pieces of base_dd_sequence until it is impossible to
                    # insert one more piece.
                    else:
                        for insert_moment_id in insertable_moment_ids[
                            0 : (len(insertable_moment_ids) // len(base_dd_sequence))
                            * len(base_dd_sequence)
                        ]:
                            insert_intos.append((insert_moment_id, next(dd_iter).on(q)))
            prev_moment_id, prev_op = next_moment_id, next_op
            next_moment_id = circuit.next_moment_operating_on([q], prev_moment_id + 1)
    mutable_circuit = circuit.unfreeze(copy=True)
    mutable_circuit.batch_insert_into(insert_intos)
    mutable_circuit.batch_replace(batch_replaces)
    return mutable_circuit


@transformer_api.transformer
def add_dynamical_decoupling(
    circuit: 'cirq.AbstractCircuit',
    *,
    context: Optional['cirq.TransformerContext'] = None,
    schema: Union[str, Tuple['cirq.Gate', ...]] = 'DEFAULT',
    single_qubit_gate_moments_only: bool = True,
) -> 'cirq.Circuit':
    """Adds dynamical decoupling gate operations to a given circuit.
    This transformer might add a new moment after each piece of Clifford moments, so the original
      moment structure could change.

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
    base_dd_sequence_info = _parse_dd_sequence(schema)

    # Step 1: for each Clifford piece, inserting dd sequence for idle operations, pulling through
    # for non-idle operations. (Note we split the circuit into Clifford pieces as we can pull Pauli
    # ops from dynamical decoupling base sequence through Clifford ops).
    updated_circuit = _fill_for_each_clifford_piece(
        circuit.freeze(), base_dd_sequence_info, single_qubit_gate_moments_only
    )

    # Step 2, for each qubit, filling consecutive empty ops and merging into single-qubit operations
    # if possible.
    updated_circuit = _fill_consecutive_idle_moments_for_each_qubit(
        updated_circuit.freeze(), base_dd_sequence_info, single_qubit_gate_moments_only
    )

    return updated_circuit
