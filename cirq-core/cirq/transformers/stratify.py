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
from typing import TYPE_CHECKING, Type, Callable, Optional, Union, Iterable, Sequence, List

from cirq import ops, circuits, _import
from cirq.transformers import transformer_api

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
    classifiers = _get_classifiers(circuit, categories)

    # Try the algorithm with each permutation of the classifiers.
    classifiers_permutations = list(itertools.permutations(classifiers))
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
    num_classes = len(classifiers) + 1  # include one "extra" category for ignored operations
    new_moments: List[List['cirq.Operation']] = []

    # Keep track of the earliest time index that can accomodate a new operation on a given qubit.
    min_time_index = {qubit: 0 for qubit in circuit.all_qubits()}

    # The minimum time index for operations with a tag in context.tags_to_ignore.
    min_time_index_for_ignored_op = 0

    for moment in circuit:
        # Identify the new time indices that operations should be moved into.
        ignored_ops = []
        op_time_indices = {}
        for op in moment:
            ignored_op = any(tag in op.tags for tag in context.tags_to_ignore)

            # Get the index of the earliest moment that can accomodate this operation,
            # and identify the "class" of this operation (by index).
            min_time_index_for_op = max(min_time_index[qubit] for qubit in op.qubits)
            if ignored_op:
                min_time_index_for_op = max(min_time_index_for_op, min_time_index_for_ignored_op)
                op_class = len(classifiers)
            else:
                op_class = _get_op_class(op, classifiers)

            # Identify the time index to place this operation into.
            time_index = (min_time_index_for_op // num_classes) * num_classes + op_class
            if time_index < min_time_index_for_op:
                time_index += num_classes

            if ignored_op:
                ignored_ops.append(op)
                min_time_index_for_ignored_op = max(min_time_index_for_ignored_op, time_index)
            else:
                op_time_indices[op] = time_index

        # Assign ignored operations to the same moment.
        for op in ignored_ops:
            op_time_indices[op] = min_time_index_for_ignored_op
        min_time_index_for_ignored_op += 1

        # Move the operations into their assigned moments.
        for op, time_index in op_time_indices.items():
            for qubit in op.qubits:
                min_time_index[qubit] = time_index + 1
            if time_index >= len(new_moments):
                new_moments += [[] for _ in range(num_classes)]
            new_moments[time_index].append(op)

    return circuits.Circuit(circuits.Moment(moment) for moment in new_moments if moment)


def _get_classifiers(
    circuit: circuits.AbstractCircuit, categories: Iterable[Category]
) -> List[Classifier]:
    """Convert a collection of categories into a list of classifiers.

    The returned list of classifiers is:
    - Exhaustive, meaning every operation in the circuit is classified by at least one classifier.
    - Minimal, meaning unused classifiers are forgotten.
    """
    # Convert all categories into classifiers, and make the list exhaustive by adding a dummy
    # classifier for otherwise unclassified ops.
    classifiers = [_category_to_classifier(cat) for cat in categories] + [_dummy_classifier]

    # Figure out which classes are actually used in the circuit.
    class_is_used = [False for _ in classifiers]
    for op in circuit.all_operations():
        class_is_used[_get_op_class(op, classifiers)] = True
        if all(class_is_used):
            break

    # Return only the classifiers that are used.
    return [classifier for classifier, is_used in zip(classifiers, class_is_used) if is_used]


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


def _dummy_classifier(op: 'cirq.Operation') -> bool:
    """Dummy classifier, used to "complete" a collection of classifiers and make it exhaustive."""
    pass


def _get_op_class(op: 'cirq.Operation', classifiers: Sequence[Classifier]) -> int:
    """Get the "class" of an operator, by index."""
    for class_index, classifier in enumerate(classifiers):
        if classifier is _dummy_classifier:
            dummy_classifier_index = class_index
        elif classifier(op):
            return class_index
    # If we got this far, the operation did not match any "actual" classifier,
    # so return the index of the dummy classifer.
    try:
        return dummy_classifier_index
    except NameError:
        raise ValueError(f"Operation {op} not identified by any classifier")
