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

"""Transformer pass that removes empty moments from a circuit."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from cirq.transformers import transformer_api, transformer_primitives

if TYPE_CHECKING:
    import cirq


@transformer_api.transformer
def drop_empty_moments(
    circuit: cirq.AbstractCircuit, *, context: Optional[cirq.TransformerContext] = None
) -> cirq.Circuit:
    """Removes empty moments from a circuit.

    Args:
          circuit: Input circuit to transform.
          context: `cirq.TransformerContext` storing common configurable options for transformers.

    Returns:
          Copy of the transformed input circuit.
    """
    if context is None:
        context = transformer_api.TransformerContext()
    return transformer_primitives.map_moments(
        circuit.unfreeze(False),
        lambda m, _: m if m else [],
        deep=context.deep,
        tags_to_ignore=context.tags_to_ignore,
    )
