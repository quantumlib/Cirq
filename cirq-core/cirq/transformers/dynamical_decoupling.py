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


def _parse_dd_sequence(schema: Union[str, Tuple['cirq.Gate', ...]]) -> Tuple['cirq.Gate', ...]:
    """Parses and returns dynamical decoupling sequence from schema."""
    if isinstance(schema, str):
        return _get_dd_sequence_from_schema_name(schema)
    else:
        _validate_dd_sequence(schema)
        return schema


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
    clifford_pieces: list[Tuple[int, int]] = []
    left = 0
    for moment_id, moment in enumerate(circuit):
        if not _is_clifford_moment(moment):
            clifford_pieces.append((left, moment_id))
            left = moment_id + 1
    if left < len(circuit):
        clifford_pieces.append((left, len(circuit)))
    return clifford_pieces


def _is_insertable_moment(moment: 'cirq.Moment', single_qubit_gate_moments_only: bool) -> bool:
    return _is_single_qubit_gate_moment(moment) or not single_qubit_gate_moments_only


def _calc_pulled_through(
    moment: 'cirq.Moment', input_pauli_ops: 'cirq.PauliString'
) -> 'cirq.PauliString':
    """Calculates the pulled_through after pulling through moment with the input.

    We assume that the moment is Clifford here. Then, pulling through is essentially
      decomposing a matrix into Pauli operations on each qubit.
    """
    pulled_through: 'cirq.PauliString' = cirq.PauliString()
    for affected_q, combined_op_in_pauli in input_pauli_ops.items():
        op_at_moment = moment.operation_at(affected_q)
        if op_at_moment is None:
            pulled_through *= combined_op_in_pauli.on(affected_q)
            continue
        prev_circuit = cirq.Circuit(cirq.Moment(op_at_moment))
        new_circuit = cirq.Circuit(
            cirq.Moment(combined_op_in_pauli.on(affected_q)), cirq.Moment(op_at_moment)
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


def _merge_pulled_through(
    mutable_circuit: 'cirq.Circuit',
    pulled_through: 'cirq.PauliString',
    clifford_piece_range: Tuple[int, int],
    single_qubit_gate_moments_only: bool,
) -> 'cirq.PauliString':
    """Merges pulled through Pauli gates into the last single-qubit gate operation or the insert it
      into the first idle moment if idle moments exist.
    Args:
        mutable_circuit: Mutable circuit to transform.
        pulled_through: Pauli gates to be merged.
        clifford_piece_range: Specifies the [l, r) moments within which pulled-through gate merging
          is to be performed.
        single_qubit_gate_moments_only: If set True, dynamical decoupling operation will only be
            added in single-qubit gate moments.

    Returns:
        The remaining pulled through operations after merging.
    """
    insert_intos: list[Tuple[int, 'cirq.Operation']] = []
    batch_replaces: list[Tuple[int, 'cirq.Operation', 'cirq.Operation']] = []
    remaining_pulled_through = pulled_through
    for affected_q, combined_op_in_pauli in pulled_through.items():
        moment_id = mutable_circuit.prev_moment_operating_on([affected_q], clifford_piece_range[1])
        if moment_id is not None:
            op = mutable_circuit.operation_at(affected_q, moment_id)
            # Try to merge op into an existing single-qubit gate operation.
            if op is not None and _is_single_qubit_operation(op):
                updated_gate_mat = cirq.unitary(combined_op_in_pauli) @ cirq.unitary(op)
                updated_gate: Optional['cirq.Gate'] = (
                    single_qubit_decompositions.single_qubit_matrix_to_phxz(updated_gate_mat)
                )
                if updated_gate is None:
                    # updated_gate is close to Identity.
                    updated_gate = cirq.I
                batch_replaces.append((moment_id, op, updated_gate.on(affected_q)))
                remaining_pulled_through *= combined_op_in_pauli.on(affected_q)
                continue
            # Insert into the first empty moment for the qubit if such moment exists.
            while moment_id < clifford_piece_range[1]:
                if affected_q not in mutable_circuit.moments[
                    moment_id
                ].qubits and _is_insertable_moment(
                    mutable_circuit.moments[moment_id], single_qubit_gate_moments_only
                ):
                    insert_intos.append((moment_id, combined_op_in_pauli.on(affected_q)))
                    remaining_pulled_through *= combined_op_in_pauli.on(affected_q)
                    break
                moment_id += 1
    mutable_circuit.batch_insert_into(insert_intos)
    mutable_circuit.batch_replace(batch_replaces)
    return remaining_pulled_through


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
    base_dd_sequence: Tuple['cirq.Gate', ...] = _parse_dd_sequence(schema)
    mutable_circuit = circuit.unfreeze(copy=True)

    pauli_map: Dict['cirq.Gate', 'cirq.Pauli'] = {}
    for gate in base_dd_sequence:
        pauli_gate = _pauli_up_to_global_phase(gate)
        if pauli_gate is not None:
            pauli_map[gate] = pauli_gate

    busy_moment_range_by_qubit: Dict['cirq.Qid', list[int]] = {
        q: [len(circuit), -1] for q in circuit.all_qubits()
    }
    for moment_id, moment in enumerate(circuit):
        for q in moment.qubits:
            busy_moment_range_by_qubit[q][0] = min(busy_moment_range_by_qubit[q][0], moment_id)
            busy_moment_range_by_qubit[q][1] = max(busy_moment_range_by_qubit[q][1], moment_id)
    clifford_pieces = _get_clifford_pieces(circuit)

    insert_intos: list[Tuple[int, 'cirq.Operation']] = []
    insert_moments: list[Tuple[int, 'cirq.Moment']] = []
    for l, r in clifford_pieces:  # [l, r)
        # A PauliString stores the result of 'pulling' Pauli gates past each operations
        # right before the current moment.
        pulled_through: 'cirq.PauliString' = cirq.PauliString()
        iter_by_qubits = {q: cycle(base_dd_sequence) for q in circuit.all_qubits()}

        # Iterate over the Clifford piece.
        for moment_id in range(l, r):
            moment = circuit.moments[moment_id]

            # Insert
            if _is_insertable_moment(moment, single_qubit_gate_moments_only):
                for q in circuit.all_qubits() - moment.qubits:
                    if (
                        busy_moment_range_by_qubit[q][0]
                        < moment_id
                        < busy_moment_range_by_qubit[q][1]
                    ):
                        insert_gate = next(iter_by_qubits[q])
                        insert_intos.append((moment_id, insert_gate.on(q)))
                        pulled_through *= pauli_map[insert_gate].on(q)

            # Pull through
            pulled_through = _calc_pulled_through(moment, pulled_through)

        mutable_circuit.batch_insert_into(insert_intos)
        insert_intos.clear()

        pulled_through = _merge_pulled_through(
            mutable_circuit, pulled_through, (l, r), single_qubit_gate_moments_only
        )

        # Insert a new moment if there are remaining pulled through operations.
        new_moment_ops = []
        for affected_q, combined_op_in_pauli in pulled_through.items():
            new_moment_ops.append(combined_op_in_pauli.on(affected_q))
        if len(new_moment_ops) != 0:
            insert_moments.append((r, cirq.Moment(new_moment_ops)))

    mutable_circuit.batch_insert(insert_moments)
    return mutable_circuit
