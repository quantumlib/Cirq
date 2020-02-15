import random
from enum import Enum
from typing import List, Set
import numpy as np

import cirq
from cirq import circuits, ops, InsertStrategy

from cirq.google.line.placement import optimization


class AlignInteractions():
    """Aligns two-qubit gates, single-qubit gates, and Z-gates into their own
    moments. Note that this optimizer does not guarantee that the resulting
    circuit will be faster - in fact, it may result in a circuit with more
    moments than the original. Instead, it attempts to split Z operations, non-Z
    operations, and 2-qubit operations into their own homogeneous moments, with
    the restriction that single qubit gates cannot move past 2 qubit gates.

    This function's algorithm is non-trivial because it boils down to the
    following Constraint Satisfiability Problem: given a set of operations which
    each have ranges where they can be applied, how can they be reorganized in
    such a way as to minimize the number of heterogeneous and total moments? See
    the tests for some cases that quickly become tricky without using a full CSP
    solver.

    Because of the intractable complexity of brute-forcing CSPs, we instead use
    Simulated Annealing to search for an optimized circuit. An adjacent circuit
    is defined as one where a single operation is moved to a brand new moment or
    swapped with an existing operation in another moment.

    Args:
        num_repetitions: The number of repetitions per temperature in the
        Simulated Annealing algorithm. The lower this number is, the quicker the
        algorithm will run, but the greater chance that it ends up with a
        non-optimal or partially heterogeneous circuit. Usually, the default is
        fine.
        random_state: A seed for random number generation.
       """

    num_repetitions: int
    random_state: np.random.RandomState
    max_num_moments: int  # The largest the circuit is allowed to grow to.

    def __init__(self,
                 *,
                 num_repetitions=10,
                 random_state: cirq.value.RANDOM_STATE_LIKE = None) -> None:
        self.random_state = cirq.value.parse_random_state(random_state)
        self.num_repetitions = num_repetitions

    def optimize_circuit(self, circuit: circuits.Circuit) -> None:
        """Optimize the circuit inline according to the algorithm specified in
        the class comment.

        This algorithm is currently slow - it takes about 5 seconds to run on an
        8x8 circuit. It could easily be optimized in the following ways:

          - Currently, circuits are used as the model during the simulated
            annealing, which causes a lot of wasted effort around copying
            circuits and moving operations around. Converting the initial
            circuit to a simpler Data Structure before optimizing and then back
            after would save a lot of time.
          - We can detect when we are at a local maximum by checking if any
            operations can be moved to a location that decreases our score. This
            could be implemented into our search and could potentially be used
            to end the search early.

        Args:
            circuit: The circuit to optimize, which will be mutated.
        """
        self.max_num_moments = len(circuit) * 3
        solution = optimization.anneal_minimize(
            initial=circuit.copy(),
            cost_func=self.cost_func,
            move_func=self.move_func,
            random_sample=self.random_state.random_sample,
            repeat=self.num_repetitions)

        del circuit[:]
        for moment in solution:
            circuit.append(moment)

    def cost_func(self, current: cirq.Circuit) -> float:
        """Scores the given circuit. It penalizes heterogeneous moments
        as well as the total number of moments, although the former incurs a
        much greater penalty than the latter. This is to allow the annealing
        algorithm to create new moments if necessary."""
        num_heterogeneous = 0
        for moment in current:
            kinds = set([operation_kind(op) for op in moment])
            if len(kinds) > 1:
                # 3-type moments get doubly penalized
                num_heterogeneous += len(kinds) - 1
        num_total = len(current)
        return num_heterogeneous + num_total / 100.0

    def move_func(self, current: cirq.Circuit) -> cirq.Circuit:
        """Returns a new circuit that is a neighbor of the given circuit.

        Two circuits are adjacent if one can be constructed from the other
        either by swapping two operations or by moving a single operation to a
        new moment.

        Args:
            current: The circuit to find a neighbor of. Will not be mutated.

        Returns: A neighbor of the given circuit.
        """
        # Pick a random qubit in the circuit and a random moment that operates
        # on it.
        rand_qubit = random.sample(current.all_qubits(), 1)[0]
        indices_for_qubit = []
        for moment_idx, moment in enumerate(current):
            if moment.operates_on_single_qubit(rand_qubit):
                indices_for_qubit.append(moment_idx)
        rand_moment_idx = random.sample(indices_for_qubit, 1)[0]
        rand_op = current.operation_at(rand_qubit, rand_moment_idx)
        if rand_op is None:
            # This should never happen, and is here to appease mypy.
            # coverage: ignore
            raise RuntimeError('Error when finding  random operation')

        # Move this operation, as long as it doesn't cross any 2-qubit gate
        # boundaries.
        return self.move_operation(current, rand_moment_idx, rand_op)

    def move_operation(self, current: cirq.Circuit, moment_idx: int,
                       op: cirq.Operation) -> cirq.Circuit:
        """Returns a new circuit that is constructed from copying the given
        circuit and moving the given operation to a valid location. This
        location is randomly chosen from any moments that would not cause this
        operation to jump past a two-qubit gate.  Note that this may require
        swapping the operation with another one.

        Args:
            current: The circuit to move the operation in. Not mutated.
            moment_idx: The moment index of the given operation.
            op: The operation to move.

        Returns: A new circuit with the given operation moved to a new moment.
        """
        # Get all the possible places to move the operation.
        valid_locations = get_possible_move_locations(current, moment_idx, op)

        # If we can't move the operation at all and the circuit is already too
        # large, just return the original circuit.
        if len(current) >= self.max_num_moments and len(valid_locations) == 0:
            return current

        # Remove the operation.
        copy = current.copy()
        copy.batch_remove([(moment_idx, op)])

        # With a certain probability OR if this operation can't be moved, insert
        # it at a new moment. Don't do this if the circuit has already reached
        # its max size.
        if len(current) < self.max_num_moments and (random.randint(0, 10) < 3 or
                                                    len(valid_locations) == 0):
            valid_locations.add(moment_idx)
            new_moment_idx = random.sample(valid_locations, 1)[0]
            copy.insert(new_moment_idx,
                        cirq.Moment([op]),
                        strategy=InsertStrategy.NEW)
            return copy

        # Otherwise, move/swap the operation to a random valid location.
        new_moment_idx = random.sample(valid_locations, 1)[0]

        # If there is an operation at this location, swap it.
        for qubit in op.qubits:
            if copy[new_moment_idx].operates_on([qubit]):
                old_op = copy.operation_at(qubit, new_moment_idx)
                if old_op is None:
                    # Should never happen - here to appease mypy.
                    # coverage: ignore
                    continue
                # Get rid of the old operation.
                copy[new_moment_idx] = copy[
                    new_moment_idx].without_operations_touching([qubit])
                # Put it in the place of the operation we are going to move.
                copy[moment_idx] = copy[moment_idx].with_operation(old_op)

        # Finally, put the operation in its new home.
        copy[new_moment_idx] = copy[new_moment_idx].with_operation(op)
        # Delete the operation's old moment if it is now empty.
        if len(copy[moment_idx]) == 0:
            del copy[moment_idx]

        return copy


def get_possible_move_locations(current: cirq.Circuit, moment_idx: int,
                                op: cirq.Operation) -> Set[int]:
    """Returns all the possible moment indices that the given operation could be
     moved to within the given circuit.

     Args:
        current: The circuit to check for possible move locations.
        moment_idx: The moment index of the given operation.
        op: The operation to move.

     Returns: A set of moment indices where the given operation could be moved
        to within the given circuit.
    """
    # Find all the valid locations for each qubit of the operation.
    all_valid_locations: List[Set[int]] = []
    for qubit in op.qubits:
        qubit_valid_locations: Set[int] = set()

        # Go from the current moment to the end of the circuit, or until a
        # 2-qubit gate is found.
        idx = moment_idx + 1
        while idx < len(current):
            new_op = current.operation_at(qubit, idx)
            if new_op is not None and (len(new_op.qubits) > 1 or
                                       len(new_op.qubits) != len(op.qubits)):
                break
            if moment_homogeneous_for_op(current[idx], op):
                qubit_valid_locations.add(idx)
            idx += 1

        # Go from the current moment to the beginning of the circuit, or until a
        # 2-qubit gate is found.
        idx = moment_idx - 1
        while idx >= 0:
            new_op = current.operation_at(qubit, idx)
            if new_op is not None and (len(new_op.qubits) > 1 or
                                       len(new_op.qubits) != len(op.qubits)):
                break
            if moment_homogeneous_for_op(current[idx], op):
                qubit_valid_locations.add(idx)
            idx -= 1
        all_valid_locations.append(qubit_valid_locations)

    # Take the intersection of the possible moments for each qubit of the
    # operation.
    return set.intersection(*all_valid_locations)


def moment_homogeneous_for_op(moment: cirq.Moment, op: cirq.Operation) -> bool:
    """Returns true iff all of the operations in the given moment have the same
    type as the given operation."""
    for moment_op in moment:
        if operation_kind(moment_op) != operation_kind(op):
            return False
    return True


class OperationKind(Enum):
    """The different kinds of operations that we want to separate out."""
    Z = 1
    SINGLE_QUBIT = 2
    TWO_QUBIT = 3


def operation_kind(op: ops.Operation) -> OperationKind:
    """Returns the kind of operation for use in sorting it into the correct
    moment."""
    if len(op.qubits) > 1:
        return OperationKind.TWO_QUBIT
    if isinstance(op.gate, ops.ZPowGate):
        return OperationKind.Z
    return OperationKind.SINGLE_QUBIT
