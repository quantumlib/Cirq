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
from typing import Callable, Hashable, Optional, TYPE_CHECKING

from cirq.transformers import transformer_api, transformer_primitives

if TYPE_CHECKING:
    import cirq


@transformer_api.transformer
def index_tags(
    circuit: 'cirq.AbstractCircuit',
    *,
    context: Optional['cirq.TransformerContext'] = None,
    target_tags: Optional[set[Hashable]] = None,
) -> 'cirq.Circuit':
    """Indexes tags in target_tags as tag_0, tag_1, ... per tag.

    Args:
        circuit: Input circuit to apply the transformations on. The input circuit is not mutated.
        context: `cirq.TransformerContext` storing common configurable options for transformers.
        target_tags: Tags to be indexed.

    Returns:
        Copy of the transformed input circuit.
    """
    if context and context.tags_to_ignore:
        raise ValueError("index_tags doesn't support tags_to_ignore, use function args instead.")
    if not target_tags:
        return circuit.unfreeze(copy=False)
    tag_iter_by_tags = {tag: itertools.count(start=0, step=1) for tag in target_tags}

    def _map_func(op: 'cirq.Operation', _) -> 'cirq.OP_TREE':
        tag_set = set(op.tags)
        nonlocal tag_iter_by_tags
        for tag in target_tags.intersection(op.tags):
            tag_set.remove(tag)
            tag_set.add(f"{tag}_{next(tag_iter_by_tags[tag])}")

        return op.untagged.with_tags(*tag_set)

    return transformer_primitives.map_operations(
        circuit, _map_func, deep=context.deep if context else False
    ).unfreeze(copy=False)


@transformer_api.transformer
def remove_tags(
    circuit: 'cirq.AbstractCircuit',
    *,
    context: Optional['cirq.TransformerContext'] = None,
    target_tags: Optional[set[Hashable]] = None,
    remove_if: Callable[[Hashable], bool] = lambda _: False,
) -> 'cirq.Circuit':
    """Removes tags from the operations based on the input args.

    Args:
        circuit: Input circuit to apply the transformations on. The input circuit is not mutated.
        context: `cirq.TransformerContext` storing common configurable options for transformers.
        target_tags: Tags to be removed.
        remove_if: A callable(tag) that returns True if the tag should be removed.
          Defaults to False.

    Returns:
        Copy of the transformed input circuit.
    """
    if context and context.tags_to_ignore:
        raise ValueError("remove_tags doesn't support tags_to_ignore, use function args instead.")
    target_tags = target_tags or set()

    def _map_func(op: 'cirq.Operation', _) -> 'cirq.OP_TREE':
        remaing_tags = set()
        for tag in op.tags:
            if not remove_if(tag) and tag not in target_tags:
                remaing_tags.add(tag)

        return op.untagged.with_tags(*remaing_tags)

    return transformer_primitives.map_operations(
        circuit, _map_func, deep=context.deep if context else False
    ).unfreeze(copy=False)
