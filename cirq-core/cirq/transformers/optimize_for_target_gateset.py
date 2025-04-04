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

"""Transformers to rewrite a circuit using gates from a given target gateset."""

from typing import Callable, Hashable, Optional, Sequence, TYPE_CHECKING, Union

from cirq import circuits
from cirq.protocols import decompose_protocol as dp
from cirq.transformers import transformer_api, transformer_primitives

if TYPE_CHECKING:
    import cirq


def _create_on_stuck_raise_error(gateset: 'cirq.Gateset'):
    def _value_error_describing_bad_operation(op: 'cirq.Operation') -> ValueError:
        return ValueError(f"Unable to convert {op} to target gateset {gateset!r}")

    return _value_error_describing_bad_operation


@transformer_api.transformer
def _decompose_operations_to_target_gateset(
    circuit: 'cirq.AbstractCircuit',
    *,
    context: Optional['cirq.TransformerContext'] = None,
    gateset: Optional['cirq.Gateset'] = None,
    decomposer: Callable[['cirq.Operation', int], dp.DecomposeResult] = lambda *_: NotImplemented,
    ignore_failures: bool = True,
    tags_to_decompose: Sequence[Hashable] = (),
) -> 'cirq.Circuit':
    """Decomposes every operation to `gateset` using `cirq.decompose` and `decomposer`.

    This transformer attempts to decompose every operation `op` in the given circuit to `gateset`
    using `cirq.decompose` protocol with `decomposer` used as an intercepting decomposer. This
    ensures that `op` is recursively decomposed using implicitly defined known decompositions
    (eg: in `_decompose_` magic method on the gaet class) till either `decomposer` knows how to
    decompose the given operation or the given operation belongs to `gateset`.

    Args:
        circuit: Input circuit to transform. It will not be modified.
        context: `cirq.TransformerContext` storing common configurable options for transformers.
        gateset: Target gateset, which the decomposed operations should belong to.
        decomposer: A callable type which accepts an (operation, moment_index) and returns
            - An equivalent `cirq.OP_TREE` implementing `op` using gates from `gateset`.
            - `None` or `NotImplemented` if does not know how to decompose a given `op`.
        ignore_failures: If set, operations that fail to convert are left unchanged. If not set,
            conversion failures raise a ValueError.
        tags_to_decompose: `cirq.CircuitOperation`s tagged with any of `tags_to_decompose` will
            be decomposed even if context.deep is True.

    Returns:
        An equivalent circuit containing gates accepted by `gateset`.

    Raises:
        ValueError: If any input operation fails to convert and `ignore_failures` is False.
    """

    def map_func(op: 'cirq.Operation', moment_index: int):
        if (
            context
            and context.deep
            and isinstance(op.untagged, circuits.CircuitOperation)
            and set(op.tags).isdisjoint(tags_to_decompose)
        ):
            return op
        return dp.decompose(
            op,
            intercepting_decomposer=lambda o: decomposer(o, moment_index),
            keep=gateset.validate if gateset else None,
            on_stuck_raise=(
                None
                if ignore_failures or gateset is None
                else _create_on_stuck_raise_error(gateset)
            ),
        )

    return transformer_primitives.map_operations_and_unroll(
        circuit,
        map_func,
        tags_to_ignore=context.tags_to_ignore if context else (),
        deep=context.deep if context else False,
    ).unfreeze(copy=False)


@transformer_api.transformer
def optimize_for_target_gateset(
    circuit: 'cirq.AbstractCircuit',
    *,
    context: Optional['cirq.TransformerContext'] = None,
    gateset: Optional['cirq.CompilationTargetGateset'] = None,
    ignore_failures: bool = True,
    max_num_passes: Union[int, None] = 1,
) -> 'cirq.Circuit':
    """Transforms the given circuit into an equivalent circuit using gates accepted by `gateset`.

    Repeat max_num_passes times or when `max_num_passes=None` until no further changes can be done
    1. Run all `gateset.preprocess_transformers`
    2. Convert operations using built-in cirq decompose + `gateset.decompose_to_target_gateset`.
    3. Run all `gateset.postprocess_transformers`

    Note:
        The optimizer is a heuristic and may not produce optimal results even with
        max_num_passes=None. The preprocessors and postprocessors of the gate set
        as well as their order yield different results.


    Args:
        circuit: Input circuit to transform. It will not be modified.
        context: `cirq.TransformerContext` storing common configurable options for transformers.
        gateset: Target gateset, which should be an instance of `cirq.CompilationTargetGateset`.
        ignore_failures: If set, operations that fail to convert are left unchanged. If not set,
            conversion failures raise a ValueError.
        max_num_passes: The maximum number of passes to do. A value of `None` means to keep
            iterating until no more changes happen to the number of moments or operations.

    Returns:
        An equivalent circuit containing gates accepted by `gateset`.

    Raises:
        ValueError: If any input operation fails to convert and `ignore_failures` is False.
    """
    if gateset is None:
        return _decompose_operations_to_target_gateset(
            circuit, context=context, ignore_failures=ignore_failures
        )
    if isinstance(max_num_passes, int):
        _outerloop = lambda: range(max_num_passes)
    else:

        def _outerloop():
            while True:
                yield 0

    initial_num_moments, initial_num_ops = len(circuit), sum(1 for _ in circuit.all_operations())
    for _ in _outerloop():
        for transformer in gateset.preprocess_transformers:
            circuit = transformer(circuit, context=context)
        circuit = _decompose_operations_to_target_gateset(
            circuit,
            context=context,
            gateset=gateset,
            decomposer=gateset.decompose_to_target_gateset,
            ignore_failures=ignore_failures,
            tags_to_decompose=(gateset._intermediate_result_tag,),
        )
        for transformer in gateset.postprocess_transformers:
            circuit = transformer(circuit, context=context)

        num_moments, num_ops = len(circuit), sum(1 for _ in circuit.all_operations())
        if (num_moments, num_ops) == (initial_num_moments, initial_num_ops):
            # Stop early. No further optimizations can be done.
            break
        initial_num_moments, initial_num_ops = num_moments, num_ops
    return circuit.unfreeze(copy=False)
