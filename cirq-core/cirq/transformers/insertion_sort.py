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

"""Transformer that sorts commuting operations in increasing order of their `.qubits` tuple."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cirq import circuits, protocols
from cirq.transformers import transformer_api

if TYPE_CHECKING:
    import cirq


@transformer_api.transformer(add_deep_support=True)
def insertion_sort_transformer(
    circuit: cirq.AbstractCircuit, *, context: cirq.TransformerContext | None = None
) -> cirq.Circuit:
    """Sorts the operations using their sorted `.qubits` property as comparison key.

    Operations are swapped only if they commute.

    Args:
        circuit: input circuit.
        context: optional TransformerContext (not used),
    """
    final_operations: list[cirq.Operation] = []
    qubit_index: dict[cirq.Qid, int] = {
        q: idx for idx, q in enumerate(sorted(circuit.all_qubits()))
    }
    cached_qubit_indices: dict[int, list[int]] = {}
    cached_measurement_keys: dict[int, frozenset[cirq.MeasurementKey]] = {}
    cached_control_keys: dict[int, frozenset[cirq.MeasurementKey]] = {}
    for pos, op in enumerate(circuit.all_operations()):
        # here `pos` is at the append position of final_operations
        op_id = id(op)
        if (op_qubit_indices := cached_qubit_indices.get(op_id)) is None:
            op_qubit_indices = cached_qubit_indices[op_id] = sorted(
                qubit_index[q] for q in op.qubits
            )
        if (op_measurement_keys := cached_measurement_keys.get(op_id)) is None:
            op_measurement_keys = cached_measurement_keys[op_id] = protocols.measurement_key_objs(
                op
            )
        if (op_control_keys := cached_control_keys.get(op_id)) is None:
            op_control_keys = cached_control_keys[op_id] = protocols.control_keys(op)

        for tail_op in reversed(final_operations):
            tail_id = id(tail_op)
            tail_qubit_indices = cached_qubit_indices[tail_id]
            tail_measurement_keys = cached_measurement_keys[tail_id]
            tail_control_keys = cached_control_keys[tail_id]
            if (
                op_qubit_indices < tail_qubit_indices
                and op_measurement_keys.isdisjoint(tail_measurement_keys)
                and op_control_keys.isdisjoint(tail_measurement_keys)
                and tail_control_keys.isdisjoint(op_measurement_keys)
                and (
                    # special case for zero-qubit gates
                    not op_qubit_indices
                    # check if two sorted sequences are disjoint
                    or op_qubit_indices[-1] < tail_qubit_indices[0]
                    or set(op_qubit_indices).isdisjoint(tail_qubit_indices)
                    # fallback to more expensive commutation check
                    or protocols.commutes(op, tail_op, default=False)
                )
            ):
                pos -= 1
                continue
            break
        final_operations.insert(pos, op)
    return circuits.Circuit(final_operations)
