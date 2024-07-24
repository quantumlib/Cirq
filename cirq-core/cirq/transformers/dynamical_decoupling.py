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
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

from cirq.transformers import transformer_api
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
import cirq
import numpy as np


def _repeat_sequence(
    base_sequence: Sequence['cirq.Gate'], num_idle_moments: int
) -> Sequence['cirq.Gate']:
    """Returns the longest possible dynamical decoupling sequence."""
    repeat_times = num_idle_moments // len(base_sequence)
    return list(base_sequence) * repeat_times


def _get_dd_sequence_from_schema_name(schema: str) -> Sequence['cirq.Gate']:
    """Gets dynamical decoupling sequence from a schema name."""
    dd_sequence: Sequence['cirq.Gate']
    match schema:
        case 'XX_PAIR':
            dd_sequence = (cirq.X, cirq.X)
        case 'X_XINV':
            dd_sequence = (cirq.X, cirq.X**-1)
        case 'YY_PAIR':
            dd_sequence = (cirq.Y, cirq.Y)
        case 'Y_YINV':
            dd_sequence = (cirq.Y, cirq.Y**-1)
        case _:
            raise ValueError('Invalid schema name.')
    return dd_sequence


def _validate_dd_sequence(dd_sequence: Sequence['cirq.Gate']) -> None:
    """Validates a given dynamical decoupling sequence.

    Args:
        dd_sequence: Input dynamical sequence to be validated.

    Returns:
        A tuple containing:
            - is_valid (bool): True if the dd sequence is valid, False otherwise.
            - error_message (str): An error message if the dd sequence is invalid, else None.

    Raises:
        ValueError: If dd_sequence is not valid.
    """
    if len(dd_sequence) < 2:
        raise ValueError('Invalid dynamical decoupling sequence. Expect more than one gates.')
    matrices = [cirq.unitary(gate) for gate in dd_sequence]
    product = reduce(np.matmul, matrices)

    if not cirq.equal_up_to_global_phase(product, np.eye(2)):
        raise ValueError(
            'Invalid dynamical decoupling sequence. Expect sequence production equals'
            f' identity up to a global phase, got {product}.'.replace('\n', ' ')
        )


def _parse_dd_sequence(schema: Union[str, Sequence['cirq.Gate']]) -> Sequence['cirq.Gate']:
    """Parses and returns dynamical decoupling sequence from schema."""
    if isinstance(schema, str):
        dd_sequence = _get_dd_sequence_from_schema_name(schema)
    else:
        _validate_dd_sequence(schema)
        dd_sequence = schema
    return dd_sequence


def _is_single_qubit_operation(operation: 'cirq.Operation'):
    if len(operation.qubits) == 1:
        return True
    return False


def _is_single_qubit_gate_moment(moment: 'cirq.Moment'):
    for operation in moment:
        if not _is_single_qubit_operation(operation):
            return False
    return True


def _absorb_remaining_gates(
    base_dd_sequence: Sequence['cirq.Gate'], idx: int, gate: 'cirq.Gate'
) -> 'cirq.Gate':
    """Returns an equivlant PhasedXZ gate for [remaining dd gates] + [an existing gate]."""
    matrices = [cirq.unitary(gate) for gate in base_dd_sequence[idx:]]
    matrices.append(cirq.unitary(gate))
    product = reduce(np.matmul, matrices)
    ret = single_qubit_decompositions.single_qubit_matrix_to_phxz(product)
    if ret is None:
        raise ValueError(f"Can't convert {product} to PhasedXZ gate.")
    return ret


def _next_gate_id(lst: Iterable[Any], idx: int) -> int:
    return (idx + 1) % len(lst)


def _add_dynamical_decoupling_to_single_qubit_gate_moments(
    circuit: 'cirq.AbstractCircuit', base_dd_sequence: Sequence['cirq.Gate']
) -> Tuple[list[Tuple[int, 'cirq.OP_TREE']], list[Tuple[int, 'cirq.Operation', 'cirq.Operation']]]:
    insert_into: list[Tuple[int, 'cirq.OP_TREE']] = []
    batch_replace: list[Tuple[int, 'cirq.Operation', 'cirq.Operation']] = []

    # Iterate over the circuit and fetch single-qubit gate moments info.
    single_qubit_gate_moments: list[int] = []
    ending_single_qubit_moment_by_qubits: Dict['cirq.Qid', int] = {
        q: -1 for q in circuit.all_qubits()
    }
    idle_single_qubit_moments_by_qubits: Dict['cirq.Qid', list[int]] = {
        q: [] for q in circuit.all_qubits()
    }
    for moment_id, moment in enumerate(circuit):
        if _is_single_qubit_gate_moment(moment):
            single_qubit_gate_moments.append(moment_id)
            for q in circuit.all_qubits():
                if not q in moment.qubits:
                    idle_single_qubit_moments_by_qubits[q].append(moment_id)
                else:
                    ending_single_qubit_moment_by_qubits[q] = moment_id

    for q in circuit.all_qubits():
        first_active_moment = circuit.next_moment_operating_on([q])
        if first_active_moment is None:
            continue
        absorbing_moment = ending_single_qubit_moment_by_qubits[q]
        if absorbing_moment <= first_active_moment:
            continue

        # For each qubit, iterate over moments in set
        #  {moment: is_single_qubit_gate_moment, in range (first_active_moment, absorbing_moment)}.
        # 1. Insert gate operation if idle.
        # 2. Merge all remaining gates in the base dd sequence to the absorbing_moment.
        pointer_id_of_dd_sequence = 0
        for moment_id in idle_single_qubit_moments_by_qubits[q]:
            # Only insert after the first active moment.
            if moment_id < first_active_moment:
                continue
            # Ignore trailing idle moments.
            if moment_id > absorbing_moment:
                break
            # Insert the next gate in the base_dd_sequence to the idle moment.
            else:
                insert_into.append((moment_id, base_dd_sequence[pointer_id_of_dd_sequence].on(q)))
                pointer_id_of_dd_sequence = _next_gate_id(
                    base_dd_sequence, pointer_id_of_dd_sequence
                )

        # Absorb the inverse of all previous gates to the last single-qubit gate of this qubit.
        op = circuit.operation_at(q, absorbing_moment)
        if pointer_id_of_dd_sequence != 0:  # absorbing if added gates isn't equivalent to identity.
            if op is not None and op.gate is not None:
                absorbed_gate = _absorb_remaining_gates(
                    base_dd_sequence, pointer_id_of_dd_sequence, op.gate
                )
                batch_replace.append((absorbing_moment, op, absorbed_gate.on(q)))
    return insert_into, batch_replace


@transformer_api.transformer
def add_dynamical_decoupling(
    circuit: 'cirq.AbstractCircuit',
    *,
    context: Optional['cirq.TransformerContext'] = None,
    schema: Union[str, Sequence['cirq.Gate']] = 'X_XINV',
    single_qubit_gate_moments_only: bool = True,
) -> 'cirq.Circuit':
    """Adds dynamical decoupling gate operations to a given circuit.
    This transformer preserves the moment structure of the circuit.

    Args:
          circuit: Input circuit to transform.
          context: `cirq.TransformerContext` storing common configurable options for transformers.
          schema: Dynamical decoupling schema name or a dynamical decoupling sequence.
            If a schema is specified, provided dynamical decouping sequence will be used.
            Otherwise, customized dynamical decoupling sequence will be applied.
          single_qubit_gate_moments_only: Whether to add gate operations in moments with non
            single-qubit gates. If set True, dynamical decoupling operation will only be added in
            single-qubit gate moments and the last single-qubit operation of each qubit will absorb
            the inverse of all previously added operations.

    Returns:
          A copy of the input circuit with dynamical decoupling operations.
    """
    insert_into: list[Tuple[int, 'cirq.OP_TREE']] = []
    batch_replace: list[Tuple[int, 'cirq.Operation', 'cirq.Operation']] = []

    base_dd_sequence = _parse_dd_sequence(schema)

    if not single_qubit_gate_moments_only:
        # Fill operations on idle moments with pieces of base_dd_sequence, it's guaranteed that all
        # inserted gates cancel as each piece of base_dd_sequence is equivalent to identity.
        last_busy_moment_by_qubits: Dict['cirq.Qid', int] = {q: 0 for q in circuit.all_qubits()}
        for moment_id, moment in enumerate(circuit):
            for q in moment.qubits:
                insert_gates = _repeat_sequence(
                    base_dd_sequence, num_idle_moments=moment_id - last_busy_moment_by_qubits[q] - 1
                )
                for idx, gate in enumerate(insert_gates):
                    insert_into.append((last_busy_moment_by_qubits[q] + idx + 1, gate.on(q)))
                last_busy_moment_by_qubits[q] = moment_id
    else:
        insert_into, batch_replace = _add_dynamical_decoupling_to_single_qubit_gate_moments(
            circuit, base_dd_sequence
        )

    updated_circuit = circuit.unfreeze(copy=True)
    updated_circuit.batch_insert_into(insert_into)
    updated_circuit.batch_replace(batch_replace)
    return updated_circuit
