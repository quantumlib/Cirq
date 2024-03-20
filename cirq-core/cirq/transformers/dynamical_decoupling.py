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

"""Transformer pass that adds dynamical decoupling moments to a circuit."""

import enum
from functools import reduce
from typing import Any, Dict, Optional, Tuple

import cirq
from cirq import value
import numpy as np


@enum.unique
class _DynamicalDecouplingSchema(enum.Enum):
    """Supported schemes of dynamical decoupling."""

    XX_PAIR = 'XX_PAIR'
    YY_PAIR = 'YY_PAIR'


def _repeat_sequence(base_sequence: list['cirq.Gate'], num_idle_moments: int):
    repeat_times = num_idle_moments // len(base_sequence)
    return base_sequence * repeat_times


def _generate_dd_sequence_from_schema(
    schema: _DynamicalDecouplingSchema, num_idle_moments: int = 2
) -> list['cirq.Gate']:
    match schema:
        case _DynamicalDecouplingSchema.XX_PAIR:
            return _repeat_sequence([cirq.XPowGate(), cirq.XPowGate()], num_idle_moments)
        case _DynamicalDecouplingSchema.YY_PAIR:
            return _repeat_sequence([cirq.YPowGate(), cirq.YPowGate()], num_idle_moments)


def _validate_dd_sequence(dd_sequence: list['cirq.Gate']) -> None:
    if len(dd_sequence) < 2:
        raise ValueError('Invalid dynamical decoupling sequence. Expect more than one gates.')
    matrices = [cirq.unitary(gate) for gate in dd_sequence]
    product = reduce(np.matmul, matrices)
    if not np.array_equal(product, np.eye(2)):
        raise ValueError(
            "Invalid dynamical decoupling sequence, sequence product doesn't equal" ' identity.'
        )


@value.value_equality
class DynamicalDecouplingModel:
    """Dynamical decoupling model that generates dynamical decoupling gate sequences."""

    def __init__(
        self,
        schema: Optional[_DynamicalDecouplingSchema] = None,
        base_dd_sequence: Optional[list['cirq.Gate']] = None,
    ):
        if not schema and not base_dd_sequence:
            raise ValueError(
                'Specify either schema or base_dd_sequence to construct a valid'
                ' DynamicalDecouplingModel.'
            )
        self.schema = schema
        self.base_dd_sequence = base_dd_sequence
        if base_dd_sequence:
            _validate_dd_sequence(base_dd_sequence)

    def generate_dd_sequence(self, num_idle_moments: int = 2) -> list['cirq.Gate']:
        """Returns the longest possible dynamical decoupling sequence."""
        if num_idle_moments <= 0:
            return []
        if self.schema:
            return _generate_dd_sequence_from_schema(self.schema, num_idle_moments)
        if self.base_dd_sequence:
            return _repeat_sequence(self.base_dd_sequence, num_idle_moments)
        return []

    @classmethod
    def from_schema(cls, schema: str):
        """Create dynamical decoupling model according to a given schema."""
        if not schema in _DynamicalDecouplingSchema.__members__:
            raise ValueError("Invalid schema name.")
        return cls(schema=_DynamicalDecouplingSchema[schema])

    @classmethod
    def from_base_dd_sequence(cls, base_dd_sequence: list['cirq.Gate']):
        """Create dynamical decoupling model according to a base sequence."""
        return cls(base_dd_sequence=base_dd_sequence)

    def _json_dict_(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.schema:
            d['schema'] = self.schema.name
        if self.base_dd_sequence:
            d['base_dd_sequence'] = self.base_dd_sequence
        return d

    @classmethod
    def _from_json_dict_(cls, schema=None, base_dd_sequence=None, **kwargs):
        if schema:
            return cls(schema=_DynamicalDecouplingSchema[schema])
        if base_dd_sequence:
            return cls(base_dd_sequence=base_dd_sequence)

    def _value_equality_values_(self) -> Any:
        return self.schema, self.base_dd_sequence


def add_dynamical_decoupling(
    circuit: 'cirq.AbstractCircuit', dd_model: DynamicalDecouplingModel
) -> 'cirq.Circuit':
    """Add dynamical decoupling gates in a given circuit.

    Args:
          circuit: Input circuit to transform.
          dd_model: Dynamical decoupling model that defines the schema to generate
            dynamical decoupling sequences.

    Return:
          A circuit with dynamical decoupling operations.
    """
    last_busy_moment_by_qubits: Dict['cirq.Qid', int] = {q: 0 for q in circuit.all_qubits()}
    insert_into: list[Tuple[int, 'cirq.OP_TREE']] = []
    for moment_id, moment in enumerate(circuit):
        for q in moment.qubits:
            insert_gates = dd_model.generate_dd_sequence(
                num_idle_moments=moment_id - last_busy_moment_by_qubits[q] - 1
            )
            for idx, gate in enumerate(insert_gates):
                insert_into.append((last_busy_moment_by_qubits[q] + idx + 1, gate.on(q)))
            last_busy_moment_by_qubits[q] = moment_id

    circuit.batch_insert_into(insert_into)

    return circuit
