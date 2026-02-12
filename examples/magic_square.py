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

"""An implementation of the Mermin–Peres magic square game.

The game is described on
[Wikipedia](https://en.wikipedia.org/wiki/Quantum_pseudo-telepathy#The_magic_square_game).
There are two players, Alice and Bob, who are allowed to share two Bell pairs but otherwise cannot
communicate. Alice is given a row number and asked to fill in three numbers, each either +1 or -1,
such that their product is +1, and Bob is given a column number and asked to fill in three numbers,
also either +1 or -1, such that their product is -1. Alice and Bob win if they agree at the
intersection of their row and column. The best classical strategy wins 8/9 of the time, but with a
quantum strategy, they can win 100% of the time.

In the quantum strategy, Alice and Bob share two Bell pairs and agree ahead of time on the
following table:

     I ⊗ Z | Z ⊗ I | Z ⊗ Z
     X ⊗ I | I ⊗ X | X ⊗ X
    -X ⊗ Z |-Z ⊗ X | Y ⊗ Y

Alice then measures the operators in the row number that she is given and returns the three values.
Similarly, Bob measures the three operators in his assigned column and returns those numbers. By
construction, Alice and Bob's numbers follow the rules of the game and agree at the intersection.

In the implementation of this strategy here, Alice and Bob only measure the first two operators in
their assigned row or column, resulting in a shallower circuit. For the first two rows and first
two columns, only single-qubit measurements are required. However, the bottom row and rightmost
column require non-destructive measurements of two-qubit Paulis, so we require ancillas. The qubit
layout that we use is as follows;

  measure ---- data ---- data ---- measure   (Alice)
                |          |
  measure ---- data ---- data ---- measure   (Bob)

the measure qubits are only used when row 3 or column 3 is requested.



=== EXAMPLE OUTPUT ===
(row, col) = (0, 0)
               ┌──┐
(0, 1): ───H────@─────────────M('alice')───
                │             │
(0, 2): ───H────┼@────────────M────────────
                ││
(1, 1): ───H────@┼────H───H───M('bob')─────
                 │            │
(1, 2): ───H─────@────H───────M────────────
               └──┘


(row, col) = (0, 1)
               ┌──┐
(0, 1): ───H────@─────────────M('alice')───
                │             │
(0, 2): ───H────┼@────────────M────────────
                ││
(1, 1): ───H────@┼────H───────M('bob')─────
                 │            │
(1, 2): ───H─────@────H───H───M────────────
               └──┘


(row, col) = (0, 2)
               ┌──┐
(0, 1): ───H────@─────────────────────────────M('alice')───
                │                             │
(0, 2): ───H────┼@────────────────────────────M────────────
                ││
(1, 0): ────────┼┼────────────────H───@───H───M('bob')─────
                ││                    │       │
(1, 1): ───H────@┼────H───H───@───H───@───────┼────────────
                 │            │               │
(1, 2): ───H─────@────H───────@───H───@───────┼────────────
                                      │       │
(1, 3): ──────────────────────────H───@───H───M────────────
               └──┘


(row, col) = (1, 0)
               ┌──┐
(0, 1): ───H────@─────────H───M('alice')───
                │             │
(0, 2): ───H────┼@────────H───M────────────
                ││
(1, 1): ───H────@┼────H───H───M('bob')─────
                 │            │
(1, 2): ───H─────@────H───────M────────────
               └──┘


(row, col) = (1, 1)
               ┌──┐
(0, 1): ───H────@─────────H───M('alice')───
                │             │
(0, 2): ───H────┼@────────H───M────────────
                ││
(1, 1): ───H────@┼────H───────M('bob')─────
                 │            │
(1, 2): ───H─────@────H───H───M────────────
               └──┘


(row, col) = (1, 2)
               ┌──┐
(0, 1): ───H────@─────────────────────────H───M('alice')───
                │                             │
(0, 2): ───H────┼@────────────────────────H───M────────────
                ││
(1, 0): ────────┼┼────────────────H───@───H───M('bob')─────
                ││                    │       │
(1, 1): ───H────@┼────H───H───@───H───@───────┼────────────
                 │            │               │
(1, 2): ───H─────@────H───────@───H───@───────┼────────────
                                      │       │
(1, 3): ──────────────────────────H───@───H───M────────────
               └──┘


(row, col) = (2, 0)
               ┌──┐
(0, 0): ──────────────────────H───@───H───M('alice')───
                                  │       │
(0, 1): ───H────@─────────@───H───@───────┼────────────
                │         │               │
(0, 2): ───H────┼@────────@───H───@───────┼────────────
                ││                │       │
(0, 3): ────────┼┼────────────H───@───H───M────────────
                ││
(1, 1): ───H────@┼────H───────────────H───M('bob')─────
                 │                        │
(1, 2): ───H─────@────H───────────────────M────────────
               └──┘


(row, col) = (2, 1)
               ┌──┐
(0, 0): ──────────────────────H───@───H───M('alice')───
                                  │       │
(0, 1): ───H────@─────────@───H───@───────┼────────────
                │         │               │
(0, 2): ───H────┼@────────@───H───@───────┼────────────
                ││                │       │
(0, 3): ────────┼┼────────────H───@───H───M────────────
                ││
(1, 1): ───H────@┼────H───────────────────M('bob')─────
                 │                        │
(1, 2): ───H─────@────H───────────────H───M────────────
               └──┘


(row, col) = (2, 2)
               ┌──┐
(0, 0): ──────────────────────────H───@───H───M('alice')───
                                      │       │
(0, 1): ───H────@─────────────@───H───@───────┼────────────
                │             │               │
(0, 2): ───H────┼@────────────@───H───@───────┼────────────
                ││                    │       │
(0, 3): ────────┼┼────────────────H───@───H───M────────────
                ││
(1, 0): ────────┼┼────────────────H───@───H───M('bob')─────
                ││                    │       │
(1, 1): ───H────@┼────H───H───@───H───@───────┼────────────
                 │            │               │
(1, 2): ───H─────@────H───────@───H───@───────┼────────────
                                      │       │
(1, 3): ──────────────────────────H───@───H───M────────────
               └──┘


Win rate for all 9 possible inputs, out of 10,000 repetitions each:
[[1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]]

"""

from __future__ import annotations

import dataclasses

import numpy as np

import cirq


def bell_pair_prep_circuit(q0: cirq.GridQubit, q1: cirq.GridQubit) -> cirq.Circuit:
    """Prepare the Bell state |00> + |11> between qubits q0 and q1.

    Args:
        q0: One qubit.
        q1: The other qubit.

    Returns:
        A circuit creating a Bell state.
    """
    return cirq.Circuit.from_moments(cirq.H.on_each(q0, q1), cirq.CZ(q0, q1), cirq.H(q1))


def state_prep_circuit(
    alice_data_qubits: tuple[cirq.GridQubit, cirq.GridQubit],
    bob_data_qubits: tuple[cirq.GridQubit, cirq.GridQubit],
) -> cirq.Circuit:
    """Construct a circuit to produce the initial Bell pairs.

    Args:
        alice_data_qubits: Alice's data qubits.
        bob_data_qubits: Bob's data qubits.

    Returns:
        A circuit preparing the Bell states.
    """
    a1, a2 = alice_data_qubits
    b1, b2 = bob_data_qubits
    return bell_pair_prep_circuit(a1, b1).zip(bell_pair_prep_circuit(a2, b2))


def construct_measure_circuit(
    alice_qubits: list[cirq.GridQubit],
    bob_qubits: list[cirq.GridQubit],
    mermin_row: int,
    mermin_col: int,
) -> cirq.Circuit:
    """Construct a circuit to implement the measurement.

    We use the conventions described in https://en.wikipedia.org/wiki/Quantum_pseudo-telepathy.
    In particular, the Mermin-Peres table is

     I ⊗ Z | Z ⊗ I | Z ⊗ Z
     X ⊗ I | I ⊗ X | X ⊗ X
    -X ⊗ Z |-Z ⊗ X | Y ⊗ Y

    Args:
        alice_qubits: The line of qubits to use for Alice (measure, data, data, measure).
        bob_qubits: The line of qubits to use for Bob (measure, data, data, measure).
        mermin_row: The row of the Mermin-Peres square to measure.
        mermin_col: The column of the Mermin-Peres square to measure.

    Returns:
        A circuit implementing the measurement.
    """

    q = alice_qubits[1:3]  # data qubits
    m = (alice_qubits[0], alice_qubits[3])  # measure qubits
    if mermin_row == 0:
        alice_circuit = cirq.Circuit.from_moments(cirq.M(*q, key="alice"))
    elif mermin_row == 1:
        alice_circuit = cirq.Circuit.from_moments(cirq.H.on_each(*q), cirq.M(*q, key="alice"))
    elif mermin_row == 2:
        alice_circuit = cirq.Circuit.from_moments(
            cirq.CZ(*q),
            cirq.H.on_each(*q, *m),
            cirq.CZ.on_each(*zip(m, q)),
            cirq.H.on_each(*m),
            cirq.M(*m, key="alice"),
        )

    q = bob_qubits[1:3]  # data qubits
    m = (bob_qubits[0], bob_qubits[3])  # measure qubits
    if mermin_col == 0:
        bob_circuit = cirq.Circuit.from_moments(cirq.H(q[0]), cirq.M(*q, key="bob"))
    elif mermin_col == 1:
        bob_circuit = cirq.Circuit.from_moments(cirq.H(q[1]), cirq.M(*q, key="bob"))
    elif mermin_col == 2:
        bob_circuit = cirq.Circuit.from_moments(
            cirq.H(q[0]),
            cirq.CZ(*q),
            cirq.H.on_each(*q, *m),
            cirq.CZ.on_each(*zip(m, q)),
            cirq.H.on_each(*m),
            cirq.M(*m, key="bob"),
        )

    last_moment_circuit = cirq.Circuit.from_moments(
        cirq.Moment(alice_circuit[-1] + bob_circuit[-1])
    )
    if mermin_row == 0:
        circuit = bob_circuit[:-1] + last_moment_circuit
    elif mermin_row == 1:
        if mermin_col == 2:
            circuit = bob_circuit[:-2] + alice_circuit.zip(bob_circuit[-2:])
        else:
            circuit = alice_circuit[:-1].zip(bob_circuit[:-1]) + last_moment_circuit
    elif mermin_row == 2:
        if mermin_col == 2:
            circuit = (
                cirq.Circuit.from_moments(bob_circuit[0])
                + alice_circuit[:-1].zip(bob_circuit[1:-1])
                + last_moment_circuit
            )
        else:
            circuit = (
                alice_circuit[0:3]
                + cirq.Circuit.from_moments(cirq.Moment(alice_circuit[3] + bob_circuit[0]))
                + last_moment_circuit
            )
    return circuit


def construct_magic_square_circuit(
    alice_qubits: list[cirq.GridQubit],
    bob_qubits: list[cirq.GridQubit],
    mermin_row: int,
    mermin_col: int,
    add_dd: bool,
    dd_scheme: tuple[cirq.Gate, ...] = (cirq.X, cirq.Y, cirq.X, cirq.Y),
) -> cirq.Circuit:
    """Construct a circuit to implement the magic square game.

    We use the conventions described in https://en.wikipedia.org/wiki/Quantum_pseudo-telepathy.
    In particular, the Mermin-Peres table is

     I ⊗ Z | Z ⊗ I | Z ⊗ Z
     X ⊗ I | I ⊗ X | X ⊗ X
    -X ⊗ Z |-Z ⊗ X | Y ⊗ Y

    Args:
        alice_qubits: The line of qubits to use for Alice (measure, data, data, measure).
        bob_qubits: The line of qubits to use for Bob (measure, data, data, measure).
        mermin_row: The row of the Mermin-Peres square to measure.
        mermin_col: The column of the Mermin-Peres square to measure.
        add_dd: Whether to add dynamical decoupling.
        dd_scheme: The dynamical decoupling sequence to use if doing DD.

    Returns:
        A circuit implementing the game.
    """
    assert len(alice_qubits) == 4, f"Expected 4 qubits, but got {len(alice_qubits)}"
    assert len(bob_qubits) == 4, f"Expected 4 qubits, but got {len(bob_qubits)}"
    alice_data_qubits = (alice_qubits[1], alice_qubits[2])
    bob_data_qubits = (bob_qubits[1], bob_qubits[2])
    prep_circuit = state_prep_circuit(alice_data_qubits, bob_data_qubits)
    measure_circuit = construct_measure_circuit(alice_qubits, bob_qubits, mermin_row, mermin_col)
    circuit = prep_circuit + measure_circuit
    if add_dd:
        circuit = cirq.add_dynamical_decoupling(
            circuit, single_qubit_gate_moments_only=True, schema=dd_scheme
        )
    return circuit


@dataclasses.dataclass
class MagicSquareResult:
    """Result of the magic square game.

    Attributes:
        alice_measurements: Alice's measurements. Shape is (mermin_row_alice, mermin_col_bob,
            repetition, mermin_col).
        bob_measurements: Bob's measurements. Shape is (mermin_row_alice, mermin_col_bob,
            repetition, mermin_row).
    """

    alice_measurements: np.ndarray
    bob_measurements: np.ndarray

    def _generate_choices_from_rules(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate choices from Alice and Bob's measurements by inferring the third number from the
        first two.

        Returns:
            Alice and Bob's choices in the game.
        """
        repetitions = self.alice_measurements.shape[2]

        # the following two arrays have indices signifying
        # [query_row, query_column, repetition, index_of_output]
        alice_choices = np.zeros((3, 3, repetitions, 3), dtype=bool)
        bob_choices = np.zeros((3, 3, repetitions, 3), dtype=bool)

        # the following manipulations implment the Mermin-Peres square from the
        # docstring of `construct_magic_square_circuit`
        alice_choices[0, :, :, 0] = self.alice_measurements[0, :, :, 1] # I ⊗ Z
        alice_choices[0, :, :, 1] = self.alice_measurements[0, :, :, 0] # Z ⊗ I
        alice_choices[1, :, :, :2] = self.alice_measurements[1, :, :, :2] # X ⊗ I | I ⊗ X
        alice_choices[2, :, :, :2] = 1 - self.alice_measurements[2, :, :, :2] # -X ⊗ Z |-Z ⊗ X
        bob_choices[:, 0, :, 0] = self.bob_measurements[:, 0, :, 1] # I ⊗ Z
        bob_choices[:, 0, :, 1] = self.bob_measurements[:, 0, :, 0] # X ⊗ I
        bob_choices[:, 1:, :, :2] = self.bob_measurements[:, 1:, :, :2] # Z ⊗ I | I ⊗ X
        alice_choices[:, :, :, 2] = np.sum(alice_choices, axis=3) % 2 # infer from rule
        bob_choices[:, :, :, 2] = 1 - (np.sum(bob_choices, axis=3) % 2) # infer from rule
        assert np.all((np.sum(alice_choices, axis=3) % 2) == 0) # check rule
        assert np.all((np.sum(bob_choices, axis=3) % 2) == 1) # check rule
        return alice_choices, bob_choices

    def generate_choices(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Generate choices from Alice and Bob's measurements.

        Args:
            seed: A seed for the random number generator.

        Returns:
            Alice and Bob's choices in the game. The two numpy arrays have indices
            signifying [query_row, query_column, repetition, index_of_output].

        Raises:
            NotImplementedError: If Alice and Bob measure unequal numbers of Paulis.
        """
        return self._generate_choices_from_rules()

    def get_win_matrix(self, seed: int | None = None) -> np.ndarray:
        """Find the fraction of the time that Alice and Bob win.

        Args:
            seed: The seed for the random number generator.

        Returns:
            The fraction of the time that they win.
        """
        alice_choices, bob_choices = self.generate_choices(seed)
        win_matrix = np.zeros((3, 3))
        for row in range(3):
            for col in range(3):
                win_matrix[row, col] = np.mean(
                    alice_choices[row, col, :, col] == bob_choices[row, col, :, row]
                )
        return win_matrix


def run_magic_square_game(
    sampler: cirq.Sampler,
    alice_qubits: list[cirq.GridQubit],
    bob_qubits: list[cirq.GridQubit],
    repetitions: int = 10_000,
    add_dd: bool = True,
    dd_scheme: tuple[cirq.Gate, ...] = (cirq.X, cirq.Y, cirq.X, cirq.Y),
) -> MagicSquareResult:
    """Run the magic square game.

    Args:
        sampler: The hardware sampler or simulator.
        alice_qubits: Alice's qubits, order is (measure, measure, data, data, measure) or (measure,
            data, data, measure).
        bob_qubits: Bob's qubits, (order is measure, measure, data, data, measure) or (measure,
            data, data, measure).
        repetitions: The number of repetitions for each row and column of the Mermin-Peres square.
        add_dd: Whether to add dynamical decoupling.
        dd_scheme: The dynamical decoupling sequence to use if doing DD.

    Returns:
        A MagicSquareResult object containing the experiment results.
    """

    all_circuits = []
    for mermin_row in range(3):
        for mermin_col in range(3):
            all_circuits.append(
                construct_magic_square_circuit(
                    alice_qubits, bob_qubits, mermin_row, mermin_col, add_dd, dd_scheme
                )
            )

    results = sampler.run_batch(all_circuits, repetitions=repetitions)

    alice_measurements = np.zeros((3, 3, repetitions, len(alice_qubits) - 2), dtype=bool)
    bob_measurements = np.zeros((3, 3, repetitions, len(bob_qubits) - 2), dtype=bool)
    idx = 0
    for row in range(3):
        for col in range(3):
            alice_measurements[row, col] = results[idx][0].measurements["alice"]
            bob_measurements[row, col] = results[idx][0].measurements["bob"]
            idx += 1

    return MagicSquareResult(alice_measurements, bob_measurements)


def main():
    sampler = cirq.Simulator()
    alice_qubits = cirq.GridQubit.rect(1, 4, 0, 0)
    bob_qubits = cirq.GridQubit.rect(1, 4, 1, 0)
    add_dd = False

    # print out the 9 circuits:
    for row in range(3):
        for col in range(3):
            print(f'(row, col) = ({row}, {col})')
            print(construct_magic_square_circuit(alice_qubits, bob_qubits, row, col, add_dd))
            print('\n')

    # run the experiment
    result = run_magic_square_game(sampler, alice_qubits, bob_qubits, add_dd=add_dd)

    # print out the win rate:
    print('Win rate for all 9 possible inputs, out of 10,000 repetitions each:')
    print(result.get_win_matrix())


if __name__ == '__main__':
    main()
