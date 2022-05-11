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

"""Transformer passes which align operations to the left or right of the circuit."""

import dataclasses
from typing import Optional, TYPE_CHECKING
from cirq import circuits, ops
from cirq.transformers import transformer_api

if TYPE_CHECKING:
    import cirq


@transformer_api.transformer(add_deep_support=True)
def align_left(
    circuit: 'cirq.AbstractCircuit', *, context: Optional['cirq.TransformerContext'] = None
) -> 'cirq.Circuit':
    """Align gates to the left of the circuit.

    Note that tagged operations with tag in `context.tags_to_ignore` will continue to stay in their
    original position and will not be aligned.

    Args:
          circuit: Input circuit to transform.
          context: `cirq.TransformerContext` storing common configurable options for transformers.

    Returns:
          Copy of the transformed input circuit.
    """
    if context is None:
        context = transformer_api.TransformerContext()

    ret = circuits.Circuit()
    for i, moment in enumerate(circuit):
        for op in moment:
            if isinstance(op, ops.TaggedOperation) and set(op.tags).intersection(
                context.tags_to_ignore
            ):
                ret.append([circuits.Moment()] * (i + 1 - len(ret)))
                ret[i] = ret[i].with_operation(op)
            else:
                ret.append(op)
    return ret


@transformer_api.transformer(add_deep_support=True)
def align_right(
    circuit: 'cirq.AbstractCircuit', *, context: Optional['cirq.TransformerContext'] = None
) -> 'cirq.Circuit':
    """Align gates to the right of the circuit.

    Note that tagged operations with tag in `context.tags_to_ignore` will continue to stay in their
    original position and will not be aligned.

    Args:
          circuit: Input circuit to transform.
          context: `cirq.TransformerContext` storing common configurable options for transformers.

    Returns:
          Copy of the transformed input circuit.
    """
    if context is not None and context.deep is True:
        context = dataclasses.replace(context, deep=False)
    return align_left(circuit[::-1], context=context)[::-1]
