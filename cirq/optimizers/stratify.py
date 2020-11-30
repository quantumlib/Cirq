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
import itertools
from typing import TYPE_CHECKING, Type, Callable, Union, Iterable, Set

from cirq import ops, circuits

if TYPE_CHECKING:
    import cirq

# A function that decides based on an operation
# whether it belongs to a class or not
Classifier = Callable[['cirq.Operation'], bool]

# Any of the possible operation categories that we can stratify on.
Category = Union[
    'cirq.Gate', 'cirq.Operation', Type['cirq.Gate'], Type['cirq.Operation'], Classifier
]


def stratified_circuit(
    circuit: 'cirq.Circuit', *, categories: Iterable[Category]
) -> 'cirq.Circuit':
    """Repacks avoiding simultaneous operations with different classes.

    Sometimes, certain operations should not be done at the same time. For
    example, the physical hardware may not be capable of doing certain
    operations at the same time. Or it may have worse noise characteristics
    when certain operations are done at the same time. In these cases, it
    would be good to rearrange the circuit so that these operations always
    occur in different moments.

    (As a secondary effect, this may make the circuit easier to read.)

    This methods takes a series of classifiers identifying categories of
    operations and then ensures operations from each category only overlap
    with operations from the same category. There is no guarantee that the
    resulting circuit will be optimally packed under this constraint.

    Args:
        circuit: The circuit whose operations should be re-arranged.
        categories: A list of classifiers picking out certain operations.
            There are several ways to specify a classifier. You can pass
            in a gate instance (e.g. `cirq.X`), a gate type (e.g.
            `cirq.XPowGate`), an operation instance (e.g.
            `cirq.X(cirq.LineQubit(0))`), an operation type (e.g.
            `cirq.GlobalPhaseOperation`), or an arbitrary operation
            predicate (e.g. `lambda op: len(op.qubits) == 2`).

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
        solutions.append(stratify_circuit(list(c), circuit))
        # Do the same thing, except this time in reverse. This helps for some
        # circuits because it inserts operations at the end instead of at the
        # beginning.
        solutions.append(stratify_circuit(list(c), reversed_circuit)[::-1])

    # Return the shortest circuit.
    return min(solutions, key=lambda c: len(c))


def stratify_circuit(classifiers: Iterable[Classifier], circuit: circuits.Circuit):
    """Performs the stratification by iterating through the operations in the
    circuit and using the given classifiers to align them.

    Args:
        classifiers: A list of rules to align the circuit. Must be exhaustive,
            i.e. all operations will be caught by one of the processors.
        circuit: The circuit to break out into homogeneous moments. Will not be
            edited.

    Returns:
        The stratified circuit.
    """
    solution = circuits.Circuit()
    circuit_copy = circuit.copy()
    while len(circuit_copy.all_qubits()) > 0:
        for classifier in classifiers:
            current_moment = ops.Moment()
            blocked_qubits: Set[ops.Qid] = set()
            for moment_idx, moment in enumerate(circuit_copy.moments):
                for op in moment.operations:
                    can_insert = classifier(op)
                    if not can_insert:
                        blocked_qubits.update(op.qubits)
                    else:
                        # Ensure that all the qubits for this operation are
                        # still available.
                        if not any(qubit in blocked_qubits for qubit in op.qubits):
                            # Add the operation to the current moment and
                            # remove it from the circuit.
                            current_moment = current_moment.with_operation(op)
                            blocked_qubits.update(op.qubits)
                            circuit_copy.batch_remove([(moment_idx, op)])

                # Short-circuit: If all the qubits are blocked, go on to the
                # next moment.
                if blocked_qubits.issuperset(circuit_copy.all_qubits()):
                    break

            if len(current_moment) > 0:
                solution.append(current_moment)
    return solution


# No type for `category` because MyPy does not keep the return type when
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
