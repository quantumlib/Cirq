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
from enum import Enum
from typing import Callable, List

import cirq
from cirq import circuits, ops


def is_two_qubit_op(op: cirq.Operation) -> bool:
    """ Returns true if the given operation operates on 2 qubits."""
    return len(op.qubits) == 2


def is_z_op(op: cirq.Operation) -> bool:
    """Returns true if the given operation holds a Z Gate."""
    return isinstance(op.gate, ops.ZPowGate)


def is_single_qubit_op(op: cirq.Operation) -> bool:
    """Returns true if the given operation operates on a single qubit and
    does not hold a Z gate. """
    return not is_two_qubit_op(op) and not is_z_op(op)


class InteractionType(Enum):
    """The different types of interactions that a given operation can have in
    the context of aligning it in a homogeneous moment."""

    # The operation can be placed in the current moment.
    MATCH = 1,
    # The operation cannot be placed in the current moment and we cannot
    # search any further on this qubit
    BLOCKER = 2,
    # The operation cannot be placed in the current moment but we can check
    # subsequent operations on this qubit.
    CONTINUE = 3


def process_single_qubit_non_z_op(op: cirq.Operation) -> InteractionType:
    """Returns the InteractionType for an operation in the context of a
    single-qubit non-Z moment. """
    if is_two_qubit_op(op):
        return InteractionType.BLOCKER
    elif is_single_qubit_op(op):
        return InteractionType.MATCH
    return InteractionType.CONTINUE


def process_z_op(op: cirq.Operation) -> InteractionType:
    """Returns the InteractionType for an operation in the context of a Z
    moment. """
    if is_two_qubit_op(op):
        return InteractionType.BLOCKER
    elif is_single_qubit_op(op):
        return InteractionType.CONTINUE
    return InteractionType.MATCH


def process_two_qubit_op(op: cirq.Operation) -> InteractionType:
    """Returns the InteractionType for an operation in the context of a
    two-qubit moment. """
    if is_two_qubit_op(op):
        return InteractionType.MATCH
    return InteractionType.BLOCKER


# A function that checks how the given operation would interact with the
# current moment.
OperationProcessor = Callable[[cirq.Operation], InteractionType]


def align_circuit(circuit: circuits.Circuit,
                  processors: List[OperationProcessor]) -> circuits.Circuit:
    """
    Performs the alignment by iterating through the operations in the circuit
    and using the given processors to align them.

    Args:
        circuit: The circuit to break out into homogeneous moments. Will not be
            edited.
        processors: A list of rules to align the circuit. Must be exhaustive,
            i.e. all operations will be caught by one of the processors.

    Returns:
        The aligned circuit.
    """
    solution = cirq.Circuit()
    circuit_copy = circuit.copy()

    while len(circuit_copy.all_qubits()) > 0:
        for processor in processors:
            current_moment = cirq.Moment()
            blocked_qubits = set()
            for moment_idx, moment in enumerate(circuit_copy.moments):
                for op in moment.operations:
                    interaction_type = processor(op)
                    if interaction_type == InteractionType.BLOCKER:
                        blocked_qubits.update(op.qubits)
                    elif interaction_type == InteractionType.MATCH:
                        # Ensure that all the qubits for this operation are
                        # still available.
                        if not any(
                                qubit in blocked_qubits for qubit in op.qubits):
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


class AlignInteractions:
    """Aligns two-qubit gates, single-qubit gates, and Z-gates into their own
    moments.  It is recommended to run a Merge optimizer before this one in
    order to merge all single-qubit gates.

    This optimizer does not guarantee that the resulting circuit will be
    faster - in fact, it may result in a circuit with more moments than the
    original. Instead, it attempts to split Z operations, non-Z operations,
    and 2-qubit operations into their own homogeneous moments, with the
    restriction that single qubit gates cannot move past 2 qubit gates.

    Note that this optimizer uses a fairly simple algorithm that is known not
    to be optimal - optimal alignment is a CSP problem with high
    dimensionality that quickly becomes intractable. See
    https://github.com/quantumlib/Cirq/pull/2772/ for some discussion on
    this, as well as a more optimal but much more complex and slow solution. """

    def __init__(self):
        pass

    def optimize_circuit(self, circuit: cirq.Circuit) -> None:
        """Runs the actual optimization algorithm. The strategy is to
        greedily search for each type of operation one at a time in order to
        partition them into moments. As an added optimization, we will run
        all permutations of the operation types' orders - for example,
        we will partition into [X, Z, 2-qubit], [Z, 2-qubit, X], [2-qubit, Z,
        X], and so on. Finally, we will do this in reverse order. These
        variations allow for certain circuits to be optimized correctly - see
        the tests for some examples.

        Args:
            circuit: The circuit to be aligned. Will be modified inline.
        """

        solutions = []
        processors_permutations = list(
            itertools.permutations([
                process_single_qubit_non_z_op, process_z_op,
                process_two_qubit_op
            ]))
        for processor in processors_permutations:
            solutions.append(align_circuit(circuit, list(processor)))

        reversed_circuit = circuit[::-1]
        for processor in processors_permutations:
            solutions.append(
                align_circuit(reversed_circuit, list(processor))[::-1])

        solution = min(solutions, key=lambda c: len(c))

        del circuit[:]
        for moment in solution:
            circuit.append(moment)
