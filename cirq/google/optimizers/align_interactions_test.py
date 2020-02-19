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

from typing import Callable, List, Dict

import numpy as np
import cirq
import cirq.google.optimizers.align_interactions as cgoai


def assert_all_homogeneous(circuit: cirq.Circuit):
    """Check that the given circuit has only homogeneous moments."""
    for moment in circuit:
        kinds = set([cgoai.operation_kind(op) for op in moment])
        assert len(kinds) <= 1, f"Moment {moment} was not homogeneous, but " \
        f" had {len(kinds)} different types of operations"


def get_qubit_to_operations(circuit: cirq.Circuit
                           ) -> Dict[cirq.Qid, List[cirq.Operation]]:
    """Creates a dictionary from qubit to all the operations that operate on
    that qubit, in order."""
    qubit_to_operations: Dict[cirq.Qid, List[cirq.Operation]] = {}
    for qubit in circuit.all_qubits():
        operations: List[cirq.Operation] = []
        moment_idx = 0
        while True:
            n = circuit.next_moment_operating_on([qubit], moment_idx)
            if n is not None:
                op = circuit.operation_at(qubit, n)
                if op is None:
                    # Should never happen, and is here to appease mypy.
                    # coverage: ignore
                    raise RuntimeError('Unexpected error gathering operations')
                operations.append(op)
                moment_idx = n + 1
            else:
                break
        qubit_to_operations[qubit] = operations
    return qubit_to_operations


def assert_optimizes(circuit: cirq.Circuit, num_moments: int):
    """Check that the given circuit has no heterogeneous moments and that the
    unitary matrix for the input circuit is the same (up to global phase and
    rounding error) as the unitary matrix for the optimized circuit.
    Additionally, check that the number of X, Z, and two-qubit moments is
    correct, and that no gates jumped over any two-qubit gates."""
    # Store a map from the Qubit ID to the order of the operations for that
    # qubit.
    before_qubit_to_operations = get_qubit_to_operations(circuit)
    circuit_before = circuit.copy()
    u_before = circuit.unitary()
    # Perform the optimization
    cgoai.AlignInteractions(
        random_state=np.random.RandomState(1)).optimize_circuit(circuit)
    # Get the same information as before now that the circuit is optimized.
    u_after = circuit.unitary()
    after_qubit_to_operations = get_qubit_to_operations(circuit)

    # Ensure that the operations did not cross any 2-qubit operation boundaries.
    for qubit, before_operations in before_qubit_to_operations.items():
        after_operations = after_qubit_to_operations[qubit]
        assert len(before_operations) == len(after_operations)
        for idx, before_op in enumerate(before_operations):
            assert len(before_op.qubits) == len(after_operations[idx].qubits), \
            f"Moment order in the following circuit was violated:\n {circuit}"

    # Ignore differences that would be caught by follow-up optimizations.
    followup_optimizations: List[Callable[[cirq.Circuit], None]] = [
        cirq.DropEmptyMoments().optimize_circuit
    ]
    for post in followup_optimizations:
        post(circuit)

    assert_all_homogeneous(circuit)
    cirq.testing.assert_allclose_up_to_global_phase(
        u_before,
        u_after,
        atol=1e-8,
        err_msg=f"The following circuits do not match. Initial: \n" \
        f"{circuit_before}\nAfter: \n{circuit}"
    )
    assert len(circuit) <= num_moments, \
    f"Expected the following circuit to have {num_moments} or fewer moments" \
        f", but had {len(circuit)} moments:\n {circuit}"


def assert_optimizes_exact(circuit: cirq.Circuit, expected: cirq.Circuit):
    """Check that the given circuit optimizes to the given expected circuit, and
    that the unitary matrix for the input circuit is the same (up to global
    phase and rounding error) as the unitary matrix of the optimized circuit.
    """
    cgoai.AlignInteractions(
        random_state=np.random.RandomState(1)).optimize_circuit(circuit)

    assert circuit == expected


def test_noop():
    """Tests that nothing happens on circuits that are already homogeneous."""
    q1, q2, q3, q4 = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(cirq.X(q1), cirq.Y(q2), cirq.X(q3))
    assert_optimizes_exact(circuit, circuit.copy())

    circuit = cirq.Circuit(
        cirq.Moment([cirq.X(q1), cirq.Y(q2), cirq.X(q3)]),
        cirq.Moment([cirq.ISWAP(q1, q2), cirq.ISWAP(q3, q4)]),
        cirq.Moment([cirq.Z(q1), cirq.Z(q2)]))
    assert_optimizes_exact(circuit, circuit.copy())


def test_simple():
    """Tests that a simple circuit maintains homogeneity."""
    q1, q2, q3 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.Moment([cirq.X(q1),
                                        cirq.Y(q2),
                                        cirq.X(q3)]), cirq.Moment([cirq.X(q1)]))
    assert_optimizes(circuit, 2)


def test_move_adjacent_gates():
    """Tests that we can move gates between adjacent moments."""
    q1, q2, q3, q4 = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(
        cirq.Moment([cirq.X(q3), cirq.X(q4)]),
        cirq.Moment([cirq.X(q1), cirq.X(q2),
                     cirq.ISWAP(q3, q4)]))
    # The circuit should have a moment of Xs and SWAPs.
    assert_optimizes(circuit, 2)


def test_eject_z():
    """Tests that a Z gate is ejected from a moment with other single-qubit
    gates."""
    q1, q2 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.Moment([cirq.ISWAP(q1, q2)]),
                           cirq.Moment([cirq.Z(q1), cirq.X(q2)]),
                           cirq.Moment([cirq.X(q1)]))
    # The circuit should have a moment of SWAPs, Xs, and Zs.
    assert_optimizes(circuit, 3)


def test_sparse_circuit():
    """Tests that a sparse circuit is still optimized correctly."""
    q1, q2, q3 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.Moment([cirq.ISWAP(q1, q2),
                                        cirq.X(q3)]), cirq.Moment([]),
                           cirq.Moment([]),
                           cirq.Moment([cirq.Z(q1),
                                        cirq.X(q2)]), cirq.Moment([]),
                           cirq.Moment([]), cirq.Moment([cirq.X(q1)]))
    # The circuit should have a moment of SWAPs, Xs, and Zs.
    assert_optimizes(circuit, 3)


def test_x_gates_pushed_back():
    """Tests that x gates are pushed back to the end if necessary."""
    q1, q2, q3, q4, q5 = cirq.LineQubit.range(5)
    circuit = cirq.Circuit(cirq.Moment([cirq.ISWAP(q1, q2),
                                        cirq.X(q5)]),
                           cirq.Moment([cirq.ISWAP(q3, q4)]),
                           cirq.Moment([cirq.X(q1), cirq.X(q2)]))
    # This circuit should have a moment of SWAPs and a moment of Xs.
    assert_optimizes(circuit, 2)


def test_z_gates_pushed_back():
    """Tests that z gates are pushed back to the end if necessary."""
    q1, q2, q3, q4, q5 = cirq.LineQubit.range(5)
    circuit = cirq.Circuit(cirq.Moment([cirq.ISWAP(q1, q2),
                                        cirq.Z(q5)]),
                           cirq.Moment([cirq.ISWAP(q3, q4)]),
                           cirq.Moment([cirq.Z(q1), cirq.Z(q2)]))
    # This circuit should have a moment of SWAPs and a moment of Zs.
    assert_optimizes(circuit, 2)


def test_two_qubit_gates_pushed_back():
    """Tests that two qubit gates are pushed back to the end if necessary."""
    q1, q2, q3, q4, q5 = cirq.LineQubit.range(5)
    circuit = cirq.Circuit(
        cirq.Moment([cirq.X(q1),
                     cirq.X(q2),
                     cirq.ISWAP(q3, q4),
                     cirq.X(q5)]), cirq.Moment([cirq.X(q1),
                                                cirq.X(q2)]),
        cirq.Moment([cirq.ISWAP(q1, q2)]))
    # This circuit should have the SWAP gate pushed to the end of the circuit,
    # and start with two X moments.
    assert_optimizes(circuit, 3)


def test_greedy_merging():
    """Tests a tricky situation where the algorithm of "Merge single-qubit gates,
    greedily search for single-qubit then 2-qubit operations" doesn't work."""
    q1, q2, q3, q4 = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(cirq.Moment([cirq.X(q1)]),
                           cirq.Moment([cirq.SWAP(q1, q2),
                                        cirq.SWAP(q3, q4)]),
                           cirq.Moment([cirq.X(q3)]),
                           cirq.Moment([cirq.SWAP(q3, q4)]))

    assert_optimizes(circuit, 3)


def test_complex_circuit():
    """Tests that a complex circuit is correctly optimized."""
    q1, q2, q3, q4, q5 = cirq.LineQubit.range(5)
    circuit = cirq.Circuit(
        cirq.Moment([cirq.X(q1), cirq.ISWAP(q2, q3),
                     cirq.Z(q5)]), cirq.Moment([cirq.X(q1),
                                                cirq.ISWAP(q4, q5)]),
        cirq.Moment([cirq.ISWAP(q1, q2), cirq.X(q4)]))
    assert_optimizes(circuit, 5)


def test_heterogeneous_circuit():
    """Tests that a circuit that is very heterogeneous is correctly optimized"""
    q1, q2, q3, q4, q5, q6 = cirq.LineQubit.range(6)
    circuit = cirq.Circuit(
        cirq.Moment(
            [cirq.X(q1),
             cirq.X(q2),
             cirq.ISWAP(q3, q4),
             cirq.ISWAP(q5, q6)]),
        cirq.Moment(
            [cirq.ISWAP(q1, q2),
             cirq.ISWAP(q3, q4),
             cirq.X(q5),
             cirq.X(q6)]),
        cirq.Moment([
            cirq.X(q1),
            cirq.Z(q2),
            cirq.X(q3),
            cirq.Z(q4),
            cirq.X(q5),
            cirq.Z(q6)
        ]))
    # Ideally, this would have 5 moments, but its complicated to get there
    # and the algorithm doesn't always pick up on it.
    assert_optimizes(circuit, 6)


def test_trapped_single_qubit_circuit():
    """Tests that a circuit containing single-qubit gates that are trapped
    between two-qubit gates are correctly separated."""
    q1, q2, q3, q4 = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(
        cirq.Moment([cirq.ISWAP(q1, q2), cirq.ISWAP(q3, q4)]),
        cirq.Moment([cirq.X(q1), cirq.Z(q2),
                     cirq.X(q3), cirq.Z(q4)]),
        cirq.Moment([cirq.ISWAP(q1, q2), cirq.ISWAP(q3, q4)]))
    assert_optimizes(circuit, 4)


def test_large_circuit():
    """Tests that a large circuit is correctly optimized."""
    q1, q2, q3, q4, q5, q6, q7, q8 = cirq.LineQubit.range(8)
    circuit = cirq.Circuit(
        cirq.Moment([
            cirq.X(q1),
            cirq.ISWAP(q2, q3),
            cirq.Z(q5),
            cirq.Z(q6),
            cirq.ISWAP(q7, q8)
        ]),
        cirq.Moment([
            cirq.ISWAP(q1, q2),
            cirq.ISWAP(q3, q8),
            cirq.ISWAP(q4, q6),
            cirq.X(q5),
            cirq.X(q7)
        ]),
        cirq.Moment([
            cirq.X(q1),
            cirq.X(q2),
            cirq.Z(q3),
            cirq.Z(q4),
            cirq.X(q5),
            cirq.X(q6)
        ]), cirq.Moment([cirq.Z(q7)]),
        cirq.Moment(
            [cirq.Z(q1),
             cirq.Z(q2),
             cirq.Z(q3),
             cirq.Z(q4),
             cirq.Z(q5)]),
        cirq.Moment(
            [cirq.ISWAP(q1, q4),
             cirq.ISWAP(q5, q8),
             cirq.ISWAP(q2, q3)]), cirq.Moment(),
        cirq.Moment([
            cirq.X(q1),
            cirq.ISWAP(q2, q3),
            cirq.Z(q5),
            cirq.Z(q6),
            cirq.ISWAP(q7, q8)
        ]))
    assert_optimizes(circuit, 11)
