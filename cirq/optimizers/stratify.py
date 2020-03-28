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

from typing import TYPE_CHECKING, Type, Callable, Union, Iterable

from cirq import ops, circuits

if TYPE_CHECKING:
    import cirq


def stratified_circuit(circuit: 'cirq.Circuit', *, categories: Iterable[
        Union['cirq.Gate', 'cirq.Operation', Type['cirq.Gate'],
              Type['cirq.Operation'], Callable[['cirq.Operation'], bool]]]
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

    # Convert classifiers into arguments for `reachable_frontier_from`.
    blockers = [_category_to_blocker(classifier) for classifier in categories]
    and_the_rest = lambda op: not all(blocker(op) for blocker in blockers)
    blockers_and_the_rest = [*blockers, and_the_rest]

    # Identify transition frontiers between categories.
    prev_frontier = {q: 0 for q in circuit.all_qubits()}
    frontiers = [prev_frontier]
    while True:
        cur_frontier = prev_frontier

        # Attempt to advance the frontier using each possible category.
        for b in blockers_and_the_rest:
            next_frontier = circuit.reachable_frontier_from(
                start_frontier=cur_frontier, is_blocker=b)
            if next_frontier == cur_frontier:
                continue  # No operations from this category at frontier.
            frontiers.append(next_frontier)
            cur_frontier = next_frontier

        # If the frontier didn't move, we should be at the end of the circuit.
        if cur_frontier == prev_frontier:
            assert set(cur_frontier.values()).issubset({len(circuit)})
            break
        prev_frontier = cur_frontier

    # Re-pack operations within each section, then concatenate into result.
    result = circuits.Circuit()
    for f1, f2 in zip(frontiers, frontiers[1:]):
        result += circuits.Circuit(
            op for _, op in circuit.findall_operations_between(f1, f2))

    return result


def _category_to_blocker(classifier):
    if isinstance(classifier, ops.Gate):
        return lambda op: op.gate != classifier
    if isinstance(classifier, ops.Operation):
        return lambda op: op != classifier
    elif isinstance(classifier, type) and issubclass(classifier, ops.Gate):
        return lambda op: not isinstance(op.gate, classifier)
    elif isinstance(classifier, type) and issubclass(classifier, ops.Operation):
        return lambda op: not isinstance(op, classifier)
    elif callable(classifier):
        return lambda op: not classifier(op)
    else:
        raise TypeError(f'Unrecognized classifier type '
                        f'{type(classifier)} ({classifier!r}).\n'
                        f'Expected a cirq.Gate, cirq.Operation, '
                        f'Type[cirq.Gate], Type[cirq.Operation], '
                        f'or Callable[[cirq.Operation], bool].')
