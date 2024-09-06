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


def _is_clifford_op(op: 'cirq.Operation') -> bool:
    if op.gate is not None and isinstance(op.gate, cirq.MeasurementGate):
        return False
    return cirq.has_stabilizer_effect(op)


def _is_insertable_moment(moment: 'cirq.Moment', single_qubit_gate_moments_only: bool) -> bool:
    return _is_single_qubit_gate_moment(moment) or not single_qubit_gate_moments_only


def _merge_single_qubit_ops_to_phxz(
    q: 'cirq.Qid', ops: Tuple['cirq.Operation', ...]
) -> 'cirq.Operation':
    """Merges [op1, op2, ...] and returns an equivalent op"""
    if len(ops) == 1:
        return ops[0]
    matrices = [cirq.unitary(op) for op in reversed(ops)]
    product = reduce(np.matmul, matrices)
    gate: Optional['cirq.Gate'] = single_qubit_decompositions.single_qubit_matrix_to_phxz(product)
    if gate is None:
        # gate is close to Identity.
        gate = cirq.I
    return gate.on(q)


def _merge_single_qubit_ops_of_two_moments(
    m1: 'cirq.Moment', m2: 'cirq.Moment'
) -> Tuple['cirq.Moment', ...]:
    for q in m1.qubits | m2.qubits:
        op1 = m1.operation_at(q)
        op2 = m2.operation_at(q)
        if op1 is not None and op2 is not None:
            if any(
                not (_is_single_qubit_operation(op) and cirq.has_unitary(op)) for op in [op1, op2]
            ):
                return (m1, m2)
    ops: set['cirq.Operation'] = set()
    for q in m1.qubits | m2.qubits:
        op12 = [op for op in [m.operation_at(q) for m in [m1, m2]] if op is not None]
        ops.add(_merge_single_qubit_ops_to_phxz(q, tuple(op12)))
    return tuple([cirq.Moment(ops)])


def _calc_pulled_through(
    moment: 'cirq.Moment', input_pauli_ops: 'cirq.PauliString'
) -> 'cirq.PauliString':
    """Calculates the pulled_through after pulling through clifford ops in moment."""
    pulled_through: 'cirq.PauliString' = cirq.PauliString()
    for affected_q, combined_gate_in_pauli in input_pauli_ops.items():
        op_at_moment = moment.operation_at(affected_q)
        if op_at_moment is None:  # keep the pauli op through empty op
            pulled_through *= combined_gate_in_pauli.on(affected_q)
            continue
        if _is_clifford_op(op_at_moment):
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
    base_dd_sequence, pauli_map = _parse_dd_sequence(schema)

    busy_moment_range_by_qubit: Dict['cirq.Qid', list[int]] = {
        q: [len(circuit), -1] for q in circuit.all_qubits()
    }
    for moment_id, moment in enumerate(circuit):
        for q in moment.qubits:
            busy_moment_range_by_qubit[q][0] = min(busy_moment_range_by_qubit[q][0], moment_id)
            busy_moment_range_by_qubit[q][1] = max(busy_moment_range_by_qubit[q][1], moment_id)

    new_moments: list['cirq.Moment'] = []
    # A PauliString stores the result of 'pulling' Pauli gates past each operations
    # right before the current moment.
    pulled_through: 'cirq.PauliString' = cirq.PauliString()
    # Iterator of gate to be used in dd sequence for each qubit.
    dd_iter_by_qubits = {q: cycle(base_dd_sequence) for q in circuit.all_qubits()}

    # Insert and pull through remaining Puali ops through the whole circuit.
    for moment_id, moment in enumerate(circuit.moments):
        # Step 1, stop pulling through for multi-qubit non-Clifford ops or gates without
        # unitary representation (e.g., measure gates). If there are remaining pulled through ops,
        # insert into a new moment before current moment.
        insert_ops: list['cirq.Operation'] = (
            []
        )  # Stores the ops needs to be inserted before current moment
        stop_pulling_through_qubits = set()
        for op in moment:
            if (
                not _is_clifford_op(op) and not _is_single_qubit_operation(op)
            ) or not cirq.has_unitary(op):
                for q in op.qubits:
                    stop_pulling_through_qubits.add(q)
                    dd_iter_by_qubits[q] = cycle(base_dd_sequence)
                    to_be_merged_pauli = pulled_through.get(q)
                    if to_be_merged_pauli is not None:
                        insert_ops.append(to_be_merged_pauli.on(q))
                        pulled_through *= to_be_merged_pauli.on(q)
        if insert_ops:  # need to insert a moment before this moment
            # Fill insertable idle moments in the new moment
            for q in circuit.all_qubits() - stop_pulling_through_qubits:
                if busy_moment_range_by_qubit[q][0] < moment_id <= busy_moment_range_by_qubit[q][1]:
                    insert_gate = next(dd_iter_by_qubits[q])
                    insert_ops.append(insert_gate.on(q))
                    pulled_through *= pauli_map[insert_gate].on(q)
            new_moment = cirq.Moment(insert_ops)
            last_moment = new_moments.pop()
            new_moments.extend(_merge_single_qubit_ops_of_two_moments(last_moment, new_moment))

        # Step 2, update current moment with insertions / merges.
        updated_moment_ops: set['cirq.Operation'] = set()
        for q in circuit.all_qubits():
            op_at_q = moment.operation_at(q)
            is_insertable_moment = _is_insertable_moment(moment, single_qubit_gate_moments_only)
            if op_at_q is None:  # insert into idle op
                if (
                    is_insertable_moment
                    and busy_moment_range_by_qubit[q][0]
                    < moment_id
                    < busy_moment_range_by_qubit[q][1]
                ):  # insert next pauli gate in the dd sequence
                    insert_gate = next(dd_iter_by_qubits[q])
                    updated_moment_ops.add(insert_gate.on(q))
                    pulled_through *= pauli_map[insert_gate].on(q)
                elif (  # insert the remaining pulled through if beyond the ending busy moment
                    is_insertable_moment
                    and moment_id > busy_moment_range_by_qubit[q][1]
                    and pulled_through.get(q) is not None
                ):
                    remaining_pulled_through_gate = pulled_through.get(q)
                    if remaining_pulled_through_gate is not None:
                        updated_moment_ops.add(remaining_pulled_through_gate.on(q))
                        pulled_through *= remaining_pulled_through_gate.on(q)
            else:  # merge pulled-through of this qubit into the op / keep the orignal op
                remaining_pulled_through_gate = pulled_through.get(q)
                if (
                    remaining_pulled_through_gate is not None
                    and cirq.has_unitary(op_at_q)
                    and _is_single_qubit_operation(op_at_q)
                    and (
                        (not _is_clifford_op(op_at_q))  # single-qubit non-Clifford op
                        or (
                            moment_id == busy_moment_range_by_qubit[q][1]
                        )  # at the last busy moment of this qubit
                    )
                ):  # merge
                    updated_moment_ops.add(
                        _merge_single_qubit_ops_to_phxz(
                            q, (remaining_pulled_through_gate.on(q), op_at_q)
                        )
                    )
                    pulled_through *= remaining_pulled_through_gate.on(q)
                else:  # keep the original op
                    updated_moment_ops.add(op_at_q)
        updated_moment = cirq.Moment(updated_moment_ops)
        new_moments.append(updated_moment)

        # Step 3, pulled through current `pulled_through` through updated_moment.
        pulled_through = _calc_pulled_through(updated_moment, pulled_through)

    # Insert a new moment if there are remaining pulled-through operations.
    ending_moment_ops = []
    for affected_q, combined_op_in_pauli in pulled_through.items():
        ending_moment_ops.append(combined_op_in_pauli.on(affected_q))
    if len(ending_moment_ops) != 0:
        ending_moment = cirq.Moment(ending_moment_ops)
        last_moment = new_moments.pop()
        new_moments.extend(_merge_single_qubit_ops_of_two_moments(last_moment, ending_moment))

    return cirq.Circuit(new_moments)
