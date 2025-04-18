# Copyright 2025 The Cirq Developers
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

import itertools
from typing import Hashable, Optional, TYPE_CHECKING


from cirq.transformers import transformer_api, transformer_primitives

if TYPE_CHECKING:
    import cirq


@transformer_api.transformer
def index_tags(
    circuit: 'cirq.AbstractCircuit',
    *,
    target_tags: set[Hashable],
    skip_op_fn: Optional[callable] = None,
    context: Optional['cirq.TransformerContext'] = None,
) -> 'cirq.Circuit':
    """Indexes all the tags in target_tags tag_0, tag_1, ....

    Args:
        target_tags: Tags to be indexed.

    Returns:
        Copy of the transformed input circuit.
    """
    if context and target_tags.intersection(context.tags_to_ignore or set()):
        raise ValueError("Can't index tags in context.tags_to_ignore.")
    tag_iter_by_tags = {tag: itertools.count(start=0, step=1) for tag in target_tags}

    def _map_func(op: 'cirq.Operation', _) -> 'cirq.OP_TREE':
        if skip_op_fn and skip_op_fn(op):
            return op
        nonlocal tag_iter_by_tags
        updated_tags = set(op.tags)
        for tag in target_tags.intersection(op.tags):
            updated_tags.remove(tag)
            updated_tags.add(f"{tag}_{next(tag_iter_by_tags[tag])}")

        return op.untagged.with_tags(*sorted(updated_tags))

    return transformer_primitives.map_operations(
        circuit, _map_func, deep=context.deep if context else False
    )


@transformer_api.transformer
def remove_tags(
    circuit: 'cirq.AbstractCircuit',
    *,
    context: Optional['cirq.TransformerContext'] = None,
    target_tags: set[Hashable],
) -> 'cirq.Circuit':
    """Remove target_tags from the tags of all operations.

    Args:
        target_tags: Tags to be indexed.

    Returns:
        Copy of the transformed input circuit.
    """
    if context and target_tags.intersection(context.tags_to_ignore or set()):
        raise ValueError("Can't remove tags in context.tags_to_ignore.")

    def _map_func(op: 'cirq.Operation', _) -> 'cirq.OP_TREE':
        return op.untagged.with_tags(*set(op.tags).difference(target_tags))

    return transformer_primitives.map_operations(
        circuit, _map_func, deep=context.deep if context else False
    )
