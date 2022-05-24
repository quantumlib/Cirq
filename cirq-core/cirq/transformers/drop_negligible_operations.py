# Copyright 2022 The Cirq Developers
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

"""Transformer pass that removes operations with tiny effects."""

from typing import Optional, TYPE_CHECKING
from cirq import protocols
from cirq.transformers import transformer_api, transformer_primitives

if TYPE_CHECKING:
    import cirq


@transformer_api.transformer
def drop_negligible_operations(
    circuit: 'cirq.AbstractCircuit',
    *,
    context: Optional['cirq.TransformerContext'] = None,
    atol: float = 1e-8,
) -> 'cirq.Circuit':
    """Removes operations with tiny effects.

    An operation `op` is considered to have a tiny effect if
    `cirq.trace_distance_bound(op) <= atol`.

    Args:
          circuit: Input circuit to transform.
          context: `cirq.TransformerContext` storing common configurable options for transformers.
          atol: Absolute tolerance to determine if an operation `op` is negligible --
                i.e. if `cirq.trace_distance_bound(op) <= atol`.

    Returns:
          Copy of the transformed input circuit.
    """
    if context is None:
        context = transformer_api.TransformerContext()

    def map_func(op: 'cirq.Operation', _: int) -> 'cirq.OP_TREE':
        return (
            op
            if protocols.num_qubits(op) > 10
            or protocols.is_measurement(op)
            or protocols.trace_distance_bound(op) > atol
            else []
        )

    return transformer_primitives.map_operations(
        circuit, map_func, tags_to_ignore=context.tags_to_ignore, deep=context.deep
    ).unfreeze(copy=False)
