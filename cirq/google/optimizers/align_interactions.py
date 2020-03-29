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

from typing import Callable, List, Set

import cirq
from cirq import circuits

# A function that checks how the given operation would interact with the
# current moment. A value of True means that the operation can be inserted,
# where False means that the operation can't be inserted and that we cannot
# check any subsequent operations on that qubit until the next moment.
OperationProcessor = Callable[[cirq.Operation], bool]


def process_single_qubit_op(op: cirq.Operation) -> bool:
    """Returns true if the operation is on a single qubit."""
    return len(op.qubits) == 2


def process_two_qubit_op(op: cirq.Operation) -> bool:
    """Returns true if the operation is on two qubits."""
    return len(op.qubits) == 1


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
            blocked_qubits: Set[cirq.Qid] = set()
            for moment_idx, moment in enumerate(circuit_copy.moments):
                for op in moment.operations:
                    can_insert = processor(op)
                    if can_insert:
                        blocked_qubits.update(op.qubits)
                    else:
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
    """Aligns two-qubit gates and single-qubit gates into their own moments.
    It is recommended to run a Merge optimizer before this one in order to
    merge all single-qubit gates, otherwise Z gates will mix with X gates,
    which is sub-optimal.

    This optimizer does not guarantee that the resulting circuit will be
    faster - in fact, it may result in a circuit with more moments than the
    original. Instead, it attempts to split 1-qubit and  2-qubit operations
    into their own homogeneous moments, with the restriction that 1-qubit gates
    cannot move past 2 qubit gates.

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
        the operation types in both orders (i.e. [1-qubit, 2-qubit] followed
        by [2-qubit, 1-qubit]). Finally, we will do this with the circuit in
        reverse order. These variations allow for certain circuits to be
        optimized correctly - see the tests for some examples.

        Args:
            circuit: The circuit to be aligned. Will be modified inline.
        """

        solutions = []
        processors_permutations = [[
            process_single_qubit_op, process_two_qubit_op
        ], [
            process_two_qubit_op,
            process_single_qubit_op,
        ]]

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
