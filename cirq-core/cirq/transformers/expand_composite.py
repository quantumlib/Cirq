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

"""Transformer pass that expands composite operations via `cirq.decompose`."""

from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING

from cirq import circuits, ops, protocols
from cirq.transformers import transformer_api, transformer_primitives

if TYPE_CHECKING:
    import cirq


@transformer_api.transformer
def expand_composite(
    circuit: cirq.AbstractCircuit,
    *,
    context: Optional[cirq.TransformerContext] = None,
    no_decomp: Callable[[ops.Operation], bool] = (lambda _: False),
):
    """A transformer that expands composite operations via `cirq.decompose`.

    For each operation in the circuit, this pass examines if the operation can
    be decomposed. If it can be, the operation is cleared out and replaced
    with its decomposition using a fixed insertion strategy.

    Transformation is applied using `cirq.map_operations_and_unroll`, which preserves the
    moment structure of the input circuit.

    Args:
          circuit: Input circuit to transform.
          context: `cirq.TransformerContext` storing common configurable options for transformers.
          no_decomp: A predicate that determines whether an operation should
                be decomposed or not. Defaults to decomposing everything.
    Returns:
          Copy of the transformed input circuit.
    """

    def map_func(op: cirq.Operation, _) -> cirq.OP_TREE:
        if context and context.deep and isinstance(op.untagged, circuits.CircuitOperation):
            return op
        return protocols.decompose(op, keep=no_decomp, on_stuck_raise=None)

    return transformer_primitives.map_operations_and_unroll(
        circuit,
        map_func,
        tags_to_ignore=context.tags_to_ignore if context else (),
        deep=context.deep if context else False,
    ).unfreeze(copy=False)
