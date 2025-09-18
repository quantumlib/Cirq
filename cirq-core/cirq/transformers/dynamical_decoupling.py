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
from enum import Enum
from typing import TYPE_CHECKING

from attrs import frozen

import numpy as np

from cirq import ops, protocols
from cirq.circuits import Circuit, FrozenCircuit, Moment
from cirq.protocols import unitary_protocol
from cirq.protocols.has_stabilizer_effect_protocol import has_stabilizer_effect
from cirq.protocols.has_unitary_protocol import has_unitary
from cirq.transformers import transformer_api
from cirq.transformers.analytical_decompositions import single_qubit_decompositions

if TYPE_CHECKING:
    import cirq


def _get_dd_sequence_from_schema_name(schema: str) -> tuple[ops.Gate, ...]:
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


def _pauli_up_to_global_phase(gate: ops.Gate) -> ops.Pauli | None:
    for pauli_gate in [ops.X, ops.Y, ops.Z]:
        if protocols.equal_up_to_global_phase(gate, pauli_gate):
            return pauli_gate
    return None


def _validate_dd_sequence(dd_sequence: tuple[ops.Gate, ...]) -> None:
    """Validates a given dynamical decoupling sequence.

    The sequence should only consist of Pauli gates and is essentially an identity gate.

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
            'Invalid dynamical decoupling sequence. Expect sequence product equals'
            f' identity up to a global phase, got {product}.'.replace('\n', ' ')
        )


def _parse_dd_sequence(
    schema: str | tuple[ops.Gate, ...],
) -> tuple[tuple[ops.Gate, ...], dict[ops.Gate, ops.Pauli]]:
    """Parses and returns dynamical decoupling sequence and its associated pauli map from schema."""
    dd_sequence = None
    if isinstance(schema, str):
        dd_sequence = _get_dd_sequence_from_schema_name(schema)
    else:
        _validate_dd_sequence(schema)
        dd_sequence = schema

    # Map gate to Pauli gate. This is necessary as dd sequence might contain gates like X^-1.
    pauli_map: dict[ops.Gate, ops.Pauli] = {}
    for gate in dd_sequence:
        pauli_gate = _pauli_up_to_global_phase(gate)
        if pauli_gate is not None:
            pauli_map[gate] = pauli_gate
    for gate in [ops.X, ops.Y, ops.Z]:
        pauli_map[gate] = gate

    return (dd_sequence, pauli_map)


def _is_single_qubit_operation(operation: ops.Operation) -> bool:
    return len(operation.qubits) == 1


def _is_single_qubit_gate_moment(moment: Moment) -> bool:
    return all(_is_single_qubit_operation(op) for op in moment)


def _is_clifford_op(op: ops.Operation) -> bool:
    return has_unitary(op) and has_stabilizer_effect(op)


def _calc_busy_moment_range_of_each_qubit(circuit: FrozenCircuit) -> dict[ops.Qid, list[int]]:
    busy_moment_range_by_qubit: dict[ops.Qid, list[int]] = {
        q: [len(circuit), -1] for q in circuit.all_qubits()
    }
    for moment_id, moment in enumerate(circuit):
        for q in moment.qubits:
            busy_moment_range_by_qubit[q][0] = min(busy_moment_range_by_qubit[q][0], moment_id)
            busy_moment_range_by_qubit[q][1] = max(busy_moment_range_by_qubit[q][1], moment_id)
    return busy_moment_range_by_qubit


def _merge_single_qubit_ops_to_phxz(
    q: ops.Qid, operations: tuple[ops.Operation, ...]
) -> ops.Operation:
    """Merges [op1, op2, ...] and returns an equivalent op"""
    if len(operations) == 1:
        return operations[0]
    matrices = [unitary_protocol.unitary(op) for op in reversed(operations)]
    product = reduce(np.matmul, matrices)
    gate = single_qubit_decompositions.single_qubit_matrix_to_phxz(product) or ops.I
    return gate.on(q)


@frozen
class _CircuitRepr:
    class _GateType(Enum):
        UNKOWN = 0
        WALL_GATE = 1
        DOOR_GATE = 2
        INSERTABLE_GATE = 3

    gate_types: dict[ops.Qid, dict[int, _CircuitRepr._GateType]]
    need_to_stop: dict[ops.Qid, dict[int, bool]]
    circuit: FrozenCircuit

    def __init__(self, circuit: cirq.FrozenCircuit, single_qubit_gate_moments_only: bool):
        object.__setattr__(self, 'circuit', circuit)

        gate_types: dict[ops.Qid, dict[int, _CircuitRepr._GateType]] = {
            q: {mid: _CircuitRepr._GateType.UNKOWN for mid in range(len(circuit))}
            for q in circuit.all_qubits()
        }
        mergable: dict[ops.Qid, dict[int, bool]] = {
            q: {mid: False for mid in range(len(circuit))} for q in circuit.all_qubits()
        }
        busy_moment_range_by_qubit = _calc_busy_moment_range_of_each_qubit(circuit)

        # Set gate types for each (q, mid)
        for mid, moment in enumerate(circuit):
            is_insertable_moment = (
                not single_qubit_gate_moments_only or _is_single_qubit_gate_moment(moment)
            )
            for q in circuit.all_qubits():
                if mid < busy_moment_range_by_qubit[q][0] or mid > busy_moment_range_by_qubit[q][1]:
                    gate_types[q][mid] = _CircuitRepr._GateType.WALL_GATE
                    continue
                op_at_q = moment.operation_at(q)
                if op_at_q is None:
                    if is_insertable_moment:
                        gate_types[q][mid] = _CircuitRepr._GateType.INSERTABLE_GATE
                        mergable[q][mid] = True
                    else:
                        gate_types[q][mid] = _CircuitRepr._GateType.DOOR_GATE
                else:
                    if _is_clifford_op(op_at_q):
                        gate_types[q][mid] = _CircuitRepr._GateType.DOOR_GATE
                        mergable[q][mid] = _is_single_qubit_operation(op_at_q)
                    else:
                        gate_types[q][mid] = _CircuitRepr._GateType.WALL_GATE
        object.__setattr__(self, 'gate_types', gate_types)

        need_to_stop: dict[ops.Qid, dict[int, bool]] = {
            q: {mid: False for mid in range(len(circuit))} for q in circuit.all_qubits()
        }
        # Reversely find the last mergeable gate of each qubit, set them as need_to_stop.
        for q in circuit.all_qubits():
            self._backward_set_stopping_slots(q, len(circuit) - 1, mergable, need_to_stop)
        # Reversely check for each wall gate, mark the closest mergeable gate as need_to_stop.
        for mid in range(len(circuit)):
            for q in circuit.all_qubits():
                if self.gate_types[q][mid] == _CircuitRepr._GateType.WALL_GATE:
                    self._backward_set_stopping_slots(q, mid - 1, mergable, need_to_stop)
        object.__setattr__(self, 'need_to_stop', need_to_stop)

    def _backward_set_stopping_slots(
        self,
        q: ops.Qid,
        from_mid: int,
        mergable: dict[ops.Qid, dict[int, bool]],
        need_to_stop: dict[ops.Qid, dict[int, bool]],
    ):
        affected_qubits: set[ops.Qid] = {q}
        for back_mid in range(from_mid, -1, -1):
            for back_q in set(affected_qubits):
                if self.gate_types[back_q][back_mid] == _CircuitRepr._GateType.WALL_GATE:
                    affected_qubits.remove(back_q)
                    continue
                if mergable[back_q][back_mid]:
                    need_to_stop[back_q][back_mid] = True
                    affected_qubits.remove(back_q)
                    continue
                op_at_q = self.circuit[back_mid].operation_at(back_q) or ops.I(q)
                affected_qubits.update(op_at_q.qubits)
            if not affected_qubits:
                break

    def __repr__(self) -> str:
        if not self.gate_types:
            return "CircuitRepr(empty)"

        qubits = sorted(list(self.gate_types.keys()))
        if not qubits:
            return "CircuitRepr(no qubits)"
        num_moments = len(self.gate_types[qubits[0]])

        type_map = {
            _CircuitRepr._GateType.WALL_GATE: 'w',
            _CircuitRepr._GateType.DOOR_GATE: 'd',
            _CircuitRepr._GateType.INSERTABLE_GATE: 'i',
            _CircuitRepr._GateType.UNKOWN: 'u',
        }

        max_qubit_len = max(len(str(q)) for q in qubits) if qubits else 0

        header = f"{'':>{max_qubit_len}} |"
        for i in range(num_moments):
            header += f" {i:^3} |"

        separator = f"{'-' * max_qubit_len}-+"
        separator += '-----+' * num_moments

        lines = ["CircuitRepr:", header, separator]

        for q in qubits:
            row_str = f"{str(q):>{max_qubit_len}} |"
            for mid in range(num_moments):
                gate_type = self.gate_types[q][mid]
                char = type_map.get(gate_type, '?')
                stop = self.need_to_stop[q][mid]
                cell = f"{char},s" if stop else f" {char} "
                row_str += f" {cell} |"
            lines.append(row_str)

        return "\n".join(lines)


@transformer_api.transformer
def add_dynamical_decoupling(
    circuit: cirq.AbstractCircuit,
    *,
    context: cirq.TransformerContext | None = None,
    schema: str | tuple[ops.Gate, ...] = 'DEFAULT',
    single_qubit_gate_moments_only: bool = True,
) -> cirq.Circuit:
    """Adds dynamical decoupling gate operations to a given circuit.
    This transformer preserves the structure of the original circuit.

    Args:
          circuit: Input circuit to transform.
          context: `cirq.TransformerContext` storing common configurable options for transformers.
          schema: Dynamical decoupling schema name or a dynamical decoupling sequence.
            If a schema is specified, the provided dynamical decoupling sequence will be used.
            Otherwise, customized dynamical decoupling sequence will be applied.
          single_qubit_gate_moments_only: If set True, dynamical decoupling operation will only be
            added in single-qubit gate moments.

    Returns:
          A copy of the input circuit with dynamical decoupling operations.
    """

    if context is not None and context.deep:
        raise ValueError("Deep transformation is not supported.")

    orig_circuit = circuit.freeze()

    repr = _CircuitRepr(orig_circuit, single_qubit_gate_moments_only)

    base_dd_sequence, pauli_map = _parse_dd_sequence(schema)
    # Stores all the moments of the output circuit chronologically.
    transformed_moments: list[Moment] = []
    # A PauliString stores the result of 'pulling' Pauli gates past each operations
    # right before the current moment.
    pulled_through: ops.PauliString = ops.PauliString()
    # Iterator of gate to be used in dd sequence for each qubit.
    dd_iter_by_qubits = {q: cycle(base_dd_sequence) for q in circuit.all_qubits()}

    for moment_id, moment in enumerate(orig_circuit.moments):
        updated_moment_ops: set[cirq.Operation] = set()
        for q in orig_circuit.all_qubits():
            new_op_at_q = moment.operation_at(q)
            if repr.gate_types[q][moment_id] == _CircuitRepr._GateType.INSERTABLE_GATE:
                new_gate = next(dd_iter_by_qubits[q])
                new_op_at_q = new_gate.on(q)
                pulled_through *= pauli_map[new_gate].on(q)
            if repr.need_to_stop[q][moment_id]:
                to_be_merged = pulled_through.get(q)
                if to_be_merged is not None:
                    new_op_at_q = _merge_single_qubit_ops_to_phxz(
                        q, [to_be_merged, new_op_at_q or ops.I(q)]
                    )
                    pulled_through *= to_be_merged.on(q)
            if new_op_at_q is not None:
                updated_moment_ops.add(new_op_at_q)

        updated_moment = Moment(updated_moment_ops)
        clifford_ops = [op for op in updated_moment if _is_clifford_op(op)]
        pulled_through = pulled_through.after(clifford_ops)
        transformed_moments.append(updated_moment)

    # DO NOT SUBMIT
    # if pulled_through.qubits() is not None:
    #     raise RuntimeError("Expect empty pulled through after propogating all moments.")

    return Circuit.from_moments(*transformed_moments)
