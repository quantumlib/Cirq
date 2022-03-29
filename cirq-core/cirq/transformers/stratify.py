# Copyright 2020 The Cirq Developers
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

"""Transformer pass to repack circuits avoiding simultaneous operations with different classes."""

import itertools
from typing import (
    TYPE_CHECKING,
    Type,
    Callable,
    Optional,
    Union,
    Iterable,
    Sequence,
    List,
    Tuple,
)

from cirq import ops, circuits, _import
from cirq.transformers import transformer_api, transformer_primitives

drop_empty_moments = _import.LazyLoader('drop_empty_moments', globals(), 'cirq.transformers')

if TYPE_CHECKING:
    import cirq

# A function that decides based on an operation
# whether it belongs to a class or not
Classifier = Callable[['cirq.Operation'], bool]

# Any of the possible operation categories that we can stratify on.
Category = Union[
    'cirq.Gate', 'cirq.Operation', Type['cirq.Gate'], Type['cirq.Operation'], Classifier
]


@transformer_api.transformer(add_deep_support=True)
def stratified_circuit(
    circuit: 'cirq.AbstractCircuit',
    *,
    context: Optional['cirq.TransformerContext'] = None,
    categories: Iterable[Category] = (),
) -> 'cirq.Circuit':
    """Repacks avoiding simultaneous operations with different classes.

    This transforms the given circuit to ensure that no operations of different categories are
    found in the same moment. Makes no optimality guarantees.
    Tagged Operations marked with any of `context.tags_to_ignore` will be treated as a separate
    category will be left in their original moments without stratification.

    Args:
        circuit: The circuit whose operations should be re-arranged. Will not be modified.
        context: `cirq.TransformerContext` storing common configurable options for transformers.
        categories: A list of classifiers picking out certain operations. There are several ways
            to specify a classifier. You can pass in a gate instance (e.g. `cirq.X`),
            a gate type (e.g. `cirq.XPowGate`), an operation instance
            (e.g. `cirq.X(cirq.LineQubit(0))`), an operation type (e.g.`cirq.CircuitOperation`),
            or an arbitrary operation predicate (e.g. `lambda op: len(op.qubits) == 2`).

    Returns:
        A copy of the original circuit, but with re-arranged operations.
    """

    # Normalize categories into classifier functions.
    classifiers = [_category_to_classifier(category) for category in categories]
    # Make the classifiers exhaustive by adding an "everything else" bucket.
    and_the_rest = lambda op: all(not classifier(op) for classifier in classifiers)
    classifiers_and_the_rest = [*classifiers, and_the_rest]

    # Try the algorithm with each permutation of the classifiers.
    classifiers_permutations = list(itertools.permutations(classifiers_and_the_rest))
    reversed_circuit = circuit[::-1]
    solutions = []
    for c in classifiers_permutations:
        solutions.append(
            _stratify_circuit(
                circuit,
                classifiers=list(c),
                context=context or transformer_api.TransformerContext(),
            )
        )
        # Do the same thing, except this time in reverse. This helps for some
        # circuits because it inserts operations at the end instead of at the
        # beginning.
        solutions.append(
            _stratify_circuit(
                reversed_circuit,
                classifiers=list(c),
                context=context or transformer_api.TransformerContext(),
            )[::-1]
        )

    # Return the shortest circuit.
    return min(solutions, key=lambda c: len(c))


def _stratify_circuit(
    circuit: circuits.AbstractCircuit,
    *,
    context: 'cirq.TransformerContext',
    classifiers: Sequence[Classifier],
) -> 'cirq.Circuit':
    """Performs the stratification by iterating through the operations in the
    circuit and using the given classifiers to align them.

    Tagged Operations marked with any of `context.tags_to_ignore` are treated as separate
    categories and left in their original moments without stratification.

    Args:
        circuit: The circuit to break out into homogeneous moments. Will not be edited.
        context: `cirq.TransformerContext` storing common configurable options for transformers.
        classifiers: A list of rules to align the circuit. Must be exhaustive, i.e. all operations
                    will be caught by one of the processors.

    Returns:
        The stratified circuit.
    """
    num_categories = len(classifiers) + 1

    def map_func(m: 'cirq.Moment', _) -> Sequence['cirq.Moment']:
        stratified_ops: List[List['cirq.Operation']] = [[] for _ in range(num_categories)]
        for op in m:
            if set(op.tags) & set(context.tags_to_ignore):
                stratified_ops[0].append(op)
                continue
            for i, classifier in enumerate(classifiers):
                if classifier(op):
                    stratified_ops[i + 1].append(op)
                    break
        return [circuits.Moment(op_list) for op_list in stratified_ops]

    stratified_circuit = transformer_primitives.map_moments(circuit, map_func).unfreeze(copy=False)
    assert len(stratified_circuit) == len(circuit) * num_categories

    # Try to move operations to the left to reduce circuit depth, preserving stratification.
    for curr_idx, moment in enumerate(stratified_circuit):
        curr_category = curr_idx % num_categories
        if curr_category == 0:
            # Moment containing tagged operations to be ignored.
            continue
        batch_removals: List[Tuple[int, 'cirq.Operation']] = []
        batch_inserts: List[Tuple[int, 'cirq.Operation']] = []
        for op in moment:
            prv_idx = stratified_circuit.earliest_available_moment(op, end_moment_index=curr_idx)
            prv_category = prv_idx % num_categories
            should_move_to_next_batch = curr_category < prv_category
            prv_idx += curr_category - prv_category + num_categories * should_move_to_next_batch
            assert prv_idx <= curr_idx and prv_idx % num_categories == curr_idx % num_categories
            if prv_idx < curr_idx:
                batch_inserts.append((prv_idx, op))
                batch_removals.append((curr_idx, op))
        stratified_circuit.batch_remove(batch_removals)
        stratified_circuit.batch_insert_into(batch_inserts)
    return drop_empty_moments.drop_empty_moments(stratified_circuit)


# No type for `category` because mypy does not keep the return type when
# returning a callback.
def _category_to_classifier(category) -> Classifier:
    """Normalizes the given category into a classifier function."""
    if isinstance(category, ops.Gate):
        return lambda op: op.gate == category
    if isinstance(category, ops.Operation):
        return lambda op: op == category
    elif isinstance(category, type) and issubclass(category, ops.Gate):
        return lambda op: isinstance(op.gate, category)
    elif isinstance(category, type) and issubclass(category, ops.Operation):
        return lambda op: isinstance(op, category)
    elif callable(category):
        return lambda op: category(op)
    else:
        raise TypeError(
            f'Unrecognized classifier type '
            f'{type(category)} ({category!r}).\n'
            f'Expected a cirq.Gate, cirq.Operation, '
            f'Type[cirq.Gate], Type[cirq.Operation], '
            f'or Callable[[cirq.Operation], bool].'
        )
