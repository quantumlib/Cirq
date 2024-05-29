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
from typing import Dict, Optional, Sequence, Tuple, Union

from cirq.transformers import transformer_api
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


@transformer_api.transformer
def add_dynamical_decoupling(
    circuit: 'cirq.AbstractCircuit',
    *,
    context: Optional['cirq.TransformerContext'] = None,
    schema: Union[str, Sequence['cirq.Gate']] = 'X_XINV',
) -> 'cirq.Circuit':
    """Adds dynamical decoupling gate operations to idle moments of a given circuit.
    This transformer preserves the moment structure of the circuit.

    Args:
          circuit: Input circuit to transform.
          context: `cirq.TransformerContext` storing common configurable options for transformers.
          schema: Dynamical decoupling schema name or a dynamical decoupling sequence.
            If a schema is specified, provided dynamical decouping sequence will be used.
            Otherwise, customized dynamical decoupling sequence will be applied.

    Returns:
          A copy of the input circuit with dynamical decoupling operations.
    """
    last_busy_moment_by_qubits: Dict['cirq.Qid', int] = {q: 0 for q in circuit.all_qubits()}
    insert_into: list[Tuple[int, 'cirq.OP_TREE']] = []

    base_dd_sequence = _parse_dd_sequence(schema)

    for moment_id, moment in enumerate(circuit):
        for q in moment.qubits:
            insert_gates = _repeat_sequence(
                base_dd_sequence, num_idle_moments=moment_id - last_busy_moment_by_qubits[q] - 1
            )
            for idx, gate in enumerate(insert_gates):
                insert_into.append((last_busy_moment_by_qubits[q] + idx + 1, gate.on(q)))
            last_busy_moment_by_qubits[q] = moment_id

    updated_circuit = circuit.unfreeze(copy=True)
    updated_circuit.batch_insert_into(insert_into)
    return updated_circuit
