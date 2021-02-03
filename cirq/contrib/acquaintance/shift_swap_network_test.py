# Copyright 2018 The Cirq Developers
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
import random

import pytest

import cirq
import cirq.contrib.acquaintance as cca


def random_part_lens(max_n_parts, max_part_size):
    return tuple(random.randint(1, max_part_size) for _ in range(random.randint(1, max_n_parts)))


@pytest.mark.parametrize(
    'left_part_lens,right_part_lens',
    [tuple(random_part_lens(7, 2) for _ in ('left', 'right')) for _ in range(5)],
)
def test_shift_swap_network_gate_acquaintance_opps(left_part_lens, right_part_lens):

    gate = cca.ShiftSwapNetworkGate(left_part_lens, right_part_lens)
    n_qubits = gate.qubit_count()
    qubits = cirq.LineQubit.range(n_qubits)
    strategy = cirq.Circuit(gate(*qubits), device=cca.UnconstrainedAcquaintanceDevice)

    # actual_opps
    initial_mapping = {q: i for i, q in enumerate(qubits)}
    actual_opps = cca.get_logical_acquaintance_opportunities(strategy, initial_mapping)

    # expected opps
    i = 0
    sides = ('left', 'right')
    parts = {side: [] for side in sides}
    for side, part_lens in zip(sides, (left_part_lens, right_part_lens)):
        for part_len in part_lens:
            parts[side].append(set(range(i, i + part_len)))
            i += part_len

    expected_opps = set(
        frozenset(left_part | right_part)
        for left_part, right_part in itertools.product(parts['left'], parts['right'])
    )
    assert actual_opps == expected_opps


circuit_diagrams = {
    (
        'undecomposed',
        (1,) * 3,
        (1,) * 3,
    ): """
0: ───(0, 0, 0)↦(1, 0, 0)───
      │
1: ───(0, 1, 0)↦(1, 1, 0)───
      │
2: ───(0, 2, 0)↦(1, 2, 0)───
      │
3: ───(1, 0, 0)↦(0, 0, 0)───
      │
4: ───(1, 1, 0)↦(0, 1, 0)───
      │
5: ───(1, 2, 0)↦(0, 2, 0)───
    """,
    (
        'decomposed',
        (1,) * 3,
        (1,) * 3,
    ): """
0: ───────────────────────█───╲0╱───────────────────────
                          │   │
1: ─────────────█───╲0╱───█───╱1╲───█───╲0╱─────────────
                │   │               │   │
2: ───█───╲0╱───█───╱1╲───█───╲0╱───█───╱1╲───█───╲0╱───
      │   │               │   │               │   │
3: ───█───╱1╲───█───╲0╱───█───╱1╲───█───╲0╱───█───╱1╲───
                │   │               │   │
4: ─────────────█───╱1╲───█───╲0╱───█───╱1╲─────────────
                          │   │
5: ───────────────────────█───╱1╲───────────────────────
    """,
    (
        'undecomposed',
        (2,) * 3,
        (2,) * 3,
    ): """
0: ────(0, 0, 0)↦(1, 0, 0)───
       │
1: ────(0, 0, 1)↦(1, 0, 1)───
       │
2: ────(0, 1, 0)↦(1, 1, 0)───
       │
3: ────(0, 1, 1)↦(1, 1, 1)───
       │
4: ────(0, 2, 0)↦(1, 2, 0)───
       │
5: ────(0, 2, 1)↦(1, 2, 1)───
       │
6: ────(1, 0, 0)↦(0, 0, 0)───
       │
7: ────(1, 0, 1)↦(0, 0, 1)───
       │
8: ────(1, 1, 0)↦(0, 1, 0)───
       │
9: ────(1, 1, 1)↦(0, 1, 1)───
       │
10: ───(1, 2, 0)↦(0, 2, 0)───
       │
11: ───(1, 2, 1)↦(0, 2, 1)───
    """,
    (
        'decomposed',
        (2,) * 3,
        (2,) * 3,
    ): """
0: ────────────────────────█───╲0╱───────────────────────
                           │   │
1: ────────────────────────█───╲1╱───────────────────────
                           │   │
2: ──────────────█───╲0╱───█───╱2╲───█───╲0╱─────────────
                 │   │     │   │     │   │
3: ──────────────█───╲1╱───█───╱3╲───█───╲1╱─────────────
                 │   │               │   │
4: ────█───╲0╱───█───╱2╲───█───╲0╱───█───╱2╲───█───╲0╱───
       │   │     │   │     │   │     │   │     │   │
5: ────█───╲1╱───█───╱3╲───█───╲1╱───█───╱3╲───█───╲1╱───
       │   │               │   │               │   │
6: ────█───╱2╲───█───╲0╱───█───╱2╲───█───╲0╱───█───╱2╲───
       │   │     │   │     │   │     │   │     │   │
7: ────█───╱3╲───█───╲1╱───█───╱3╲───█───╲1╱───█───╱3╲───
                 │   │               │   │
8: ──────────────█───╱2╲───█───╲0╱───█───╱2╲─────────────
                 │   │     │   │     │   │
9: ──────────────█───╱3╲───█───╲1╱───█───╱3╲─────────────
                           │   │
10: ───────────────────────█───╱2╲───────────────────────
                           │   │
11: ───────────────────────█───╱3╲───────────────────────
    """,
    (
        'undecomposed',
        (1, 2, 2),
        (2, 1, 2),
    ): """
0: ───(0, 0, 0)↦(1, 0, 0)───
      │
1: ───(0, 1, 0)↦(1, 1, 0)───
      │
2: ───(0, 1, 1)↦(1, 1, 1)───
      │
3: ───(0, 2, 0)↦(1, 2, 0)───
      │
4: ───(0, 2, 1)↦(1, 2, 1)───
      │
5: ───(1, 0, 0)↦(0, 0, 0)───
      │
6: ───(1, 0, 1)↦(0, 0, 1)───
      │
7: ───(1, 1, 0)↦(0, 1, 0)───
      │
8: ───(1, 2, 0)↦(0, 2, 0)───
      │
9: ───(1, 2, 1)↦(0, 2, 1)───
    """,
    (
        'decomposed',
        (1, 2, 2),
        (2, 1, 2),
    ): """
0: ───────────────────────█───╲0╱───────────────────────
                          │   │
1: ─────────────█───╲0╱───█───╱1╲───────────────────────
                │   │     │   │
2: ─────────────█───╲1╱───█───╱2╲───█───╲0╱─────────────
                │   │               │   │
3: ───█───╲0╱───█───╱2╲───█───╲0╱───█───╱1╲───█───╲0╱───
      │   │     │   │     │   │               │   │
4: ───█───╲1╱───█───╱3╲───█───╲1╱───█───╲0╱───█───╱1╲───
      │   │               │   │     │   │     │   │
5: ───█───╱2╲───█───╲0╱───█───╱2╲───█───╲1╱───█───╱2╲───
      │   │     │   │               │   │
6: ───█───╱3╲───█───╲1╱───█───╲0╱───█───╱2╲─────────────
                │   │     │   │     │   │
7: ─────────────█───╱2╲───█───╲1╱───█───╱3╲─────────────
                          │   │
8: ───────────────────────█───╱2╲───────────────────────
                          │   │
9: ───────────────────────█───╱3╲───────────────────────
    """,
}


@pytest.mark.parametrize('left_part_lens,right_part_lens', set(key[1:] for key in circuit_diagrams))
def test_shift_swap_network_gate_diagrams(left_part_lens, right_part_lens):

    gate = cca.ShiftSwapNetworkGate(left_part_lens, right_part_lens)
    n_qubits = gate.qubit_count()
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit(gate(*qubits))

    diagram = circuit_diagrams['undecomposed', left_part_lens, right_part_lens]
    cirq.testing.assert_has_diagram(circuit, diagram)

    cca.expose_acquaintance_gates(circuit)
    diagram = circuit_diagrams['decomposed', left_part_lens, right_part_lens]
    cirq.testing.assert_has_diagram(circuit, diagram)


def test_shift_swap_network_gate_bad_part_lens():
    with pytest.raises(ValueError):
        cca.ShiftSwapNetworkGate((0, 1, 1), (2, 2))
    with pytest.raises(ValueError):
        cca.ShiftSwapNetworkGate((-1, 1, 1), (2, 2))
    with pytest.raises(ValueError):
        cca.ShiftSwapNetworkGate((1, 1), (2, 0, 2))
    with pytest.raises(ValueError):
        cca.ShiftSwapNetworkGate((1, 1), (2, -3))


@pytest.mark.parametrize(
    'left_part_lens,right_part_lens',
    [tuple(random_part_lens(2, 2) for _ in ('left', 'right')) for _ in range(5)],
)
def test_shift_swap_network_gate_repr(left_part_lens, right_part_lens):
    gate = cca.ShiftSwapNetworkGate(left_part_lens, right_part_lens)
    cirq.testing.assert_equivalent_repr(gate)

    gate = cca.ShiftSwapNetworkGate(left_part_lens, right_part_lens, cirq.ZZ)
    cirq.testing.assert_equivalent_repr(gate)


@pytest.mark.parametrize(
    'left_part_lens,right_part_lens',
    [tuple(random_part_lens(2, 2) for _ in ('left', 'right')) for _ in range(5)],
)
def test_shift_swap_network_gate_permutation(left_part_lens, right_part_lens):
    gate = cca.ShiftSwapNetworkGate(left_part_lens, right_part_lens)
    n_qubits = gate.qubit_count()
    cca.testing.assert_permutation_decomposition_equivalence(gate, n_qubits)
