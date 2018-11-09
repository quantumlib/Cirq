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

from itertools import product
from string import ascii_lowercase as alphabet
from typing import Sequence, Tuple

from numpy.random import poisson
import pytest

import cirq
from cirq.contrib.acquaintance.gates import (
        ACQUAINT, SwapNetworkGate)
from cirq.contrib.acquaintance.devices import (
        get_acquaintance_size)
from cirq.contrib.acquaintance.shift import CircularShiftGate
from cirq.contrib.acquaintance.permutation import (
        update_mapping, LinearPermutationGate)


def test_acquaintance_gate_repr():
    assert repr(ACQUAINT) == 'Acq'

def test_acquaintance_gate_text_diagram_info():
    qubits = [cirq.NamedQubit(s) for s in 'xyz']
    circuit = cirq.Circuit([cirq.Moment([ACQUAINT(*qubits)])])
    actual_text_diagram = circuit.to_text_diagram().strip()
    expected_text_diagram = """
x: ───█───
      │
y: ───█───
      │
z: ───█───
    """.strip()
    assert actual_text_diagram == expected_text_diagram


def test_acquaintance_gate_unknown_qubit_count():
    assert cirq.circuit_diagram_info(ACQUAINT, default=None) is None


def test_swap_network_gate():
    qubits = tuple(cirq.NamedQubit(s) for s in alphabet)

    acquaintance_size = 3
    n_parts = 3
    part_lens = (acquaintance_size - 1,) * n_parts
    n_qubits = sum(part_lens)
    swap_network_op = SwapNetworkGate(part_lens,
        acquaintance_size=acquaintance_size)(*qubits[:n_qubits])
    swap_network = cirq.Circuit.from_ops(swap_network_op)
    actual_text_diagram = swap_network.to_text_diagram().strip()
    expected_text_diagram = """
a: ───×(0,0)───
      │
b: ───×(0,1)───
      │
c: ───×(1,0)───
      │
d: ───×(1,1)───
      │
e: ───×(2,0)───
      │
f: ───×(2,1)───
    """.strip()
    assert actual_text_diagram == expected_text_diagram

    no_decomp = lambda op: isinstance(op.gate,
            (CircularShiftGate, LinearPermutationGate))
    expander = cirq.ExpandComposite(no_decomp=no_decomp)
    expander(swap_network)
    actual_text_diagram = swap_network.to_text_diagram().strip()
    expected_text_diagram = """
a: ───█───────╲0╱───█─────────────────█───────────╲0╱───█───────0↦1───
      │       │     │                 │           │     │       │
b: ───█───█───╲1╱───█───█─────────────█───█───────╲1╱───█───█───1↦0───
      │   │   │     │   │             │   │       │     │   │
c: ───█───█───╱2╲───█───█───█───╲0╱───█───█───█───╱2╲───█───█───0↦1───
          │   │         │   │   │         │   │   │         │   │
d: ───────█───╱3╲───█───█───█───╲1╱───█───█───█───╱3╲───────█───1↦0───
                    │       │   │     │       │
e: ─────────────────█───────█───╱2╲───█───────█───0↦1─────────────────
                    │           │     │           │
f: ─────────────────█───────────╱3╲───█───────────1↦0─────────────────
    """.strip()
    assert actual_text_diagram == expected_text_diagram

    no_decomp = lambda op: isinstance(op.gate, CircularShiftGate)
    expander = cirq.ExpandComposite(no_decomp=no_decomp)

    acquaintance_size = 3
    n_parts = 6
    part_lens = (1,) * n_parts
    n_qubits = sum(part_lens)
    swap_network_op = SwapNetworkGate(part_lens,
        acquaintance_size=acquaintance_size)(*qubits[:n_qubits])
    swap_network = cirq.Circuit.from_ops(swap_network_op)

    expander(swap_network)
    actual_text_diagram = swap_network.to_text_diagram().strip()
    expected_text_diagram = """
a: ───╲0╱─────────╲0╱─────────╲0╱─────────
      │           │           │
b: ───╱1╲───╲0╱───╱1╲───╲0╱───╱1╲───╲0╱───
            │           │           │
c: ───╲0╱───╱1╲───╲0╱───╱1╲───╲0╱───╱1╲───
      │           │           │
d: ───╱1╲───╲0╱───╱1╲───╲0╱───╱1╲───╲0╱───
            │           │           │
e: ───╲0╱───╱1╲───╲0╱───╱1╲───╲0╱───╱1╲───
      │           │           │
f: ───╱1╲─────────╱1╲─────────╱1╲─────────
    """.strip()
    assert actual_text_diagram == expected_text_diagram

@pytest.mark.parametrize('part_lens, acquaintance_size',
    list(((part_len,) * n_parts, acquaintance_size) for
         part_len, acquaintance_size, n_parts in
         product(range(1, 5), range(5), range(2, 5)))
    )
def test_swap_network_gate_permutation(part_lens, acquaintance_size):
    n_qubits = sum(part_lens)
    qubits = cirq.LineQubit.range(n_qubits)
    swap_network_gate = SwapNetworkGate(part_lens, acquaintance_size)
    operations = cirq.decompose_once_with_qubits(swap_network_gate, qubits)
    operations = list(cirq.flatten_op_tree(operations))
    mapping = {q: i for i, q in enumerate(qubits)}
    update_mapping(mapping, operations)
    assert mapping == {q: i for i, q in enumerate(reversed(qubits))}

def test_swap_network_gate_from_ops():
    n_qubits = 10
    qubits = cirq.LineQubit.range(n_qubits)
    part_lens = (1, 2, 1, 3, 3)
    operations = [cirq.Z(qubits[0]),
                  cirq.CZ(*qubits[1:3]),
                  cirq.CCZ(*qubits[4:7]),
                  cirq.CCZ(*qubits[7:])]
    acquaintance_size = 3
    swap_network = SwapNetworkGate.from_operations(
            qubits, operations, acquaintance_size)
    assert swap_network.acquaintance_size == acquaintance_size
    assert swap_network.part_lens == part_lens


def test_swap_network_decomposition():
    qubits = cirq.LineQubit.range(8)
    swap_network_gate = SwapNetworkGate((4, 4), 5)
    operations = cirq.decompose_once_with_qubits(swap_network_gate, qubits)
    circuit = cirq.Circuit.from_ops(operations)
    actual_text_diagram = circuit.to_text_diagram()
    expected_text_diagram = """
0: ───█─────────────█─────────────╲0╱─────────────█─────────█───────0↦2───
      │             │             │               │         │       │
1: ───█─────────────█─────────────╲1╱─────────────█─────────█───────1↦3───
      │             │             │               │         │       │
2: ───█─────────────█───1↦0───────╲2╱───────1↦0───█─────────█───────2↦0───
      │             │   │         │         │     │         │       │
3: ───█───█─────────█───0↦1───█───╲3╱───█───0↦1───█─────────█───█───3↦1───
      │   │         │         │   │     │         │         │   │
4: ───█───█───0↦1───█─────────█───╱4╲───█─────────█───0↦1───█───█───0↦2───
          │   │               │   │     │             │         │   │
5: ───────█───1↦0─────────────█───╱5╲───█─────────────1↦0───────█───1↦3───
          │                   │   │     │                       │   │
6: ───────█───────────────────█───╱6╲───█───────────────────────█───2↦0───
          │                   │   │     │                       │   │
7: ───────█───────────────────█───╱7╲───█───────────────────────█───3↦1───
    """.strip()
    assert actual_text_diagram == expected_text_diagram

def test_swap_network_init_error():
    with pytest.raises(ValueError):
        SwapNetworkGate(())
    with pytest.raises(ValueError):
        SwapNetworkGate((3,))

@pytest.mark.parametrize('part_lens, acquaintance_size', [
    [[l + 1 for l in poisson(size=n_parts, lam=lam)], poisson(4)]
    for n_parts, lam in product(range(2, 20, 3), range(1, 4))
     ])
def test_swap_network_permutation(part_lens, acquaintance_size):
    n_qubits = sum(part_lens)
    gate = SwapNetworkGate(part_lens, acquaintance_size)

    expected_permutation = {i: j for i, j in
            zip(range(n_qubits), reversed(range(n_qubits)))}
    assert gate.permutation(n_qubits) == expected_permutation

def test_swap_network_permutation_error():
    gate = SwapNetworkGate((1, 1))
    with pytest.raises(ValueError):
        gate.permutation(1)

class OtherOperation(cirq.Operation):
    def __init__(self, qubits: Sequence[cirq.QubitId]) -> None:
        self._qubits = tuple(qubits)

    @property
    def qubits(self) -> Tuple[cirq.QubitId, ...]:
        return self._qubits

    def with_qubits(self, *new_qubits: cirq.QubitId) -> 'OtherOperation':
        return type(self)(self._qubits)

    def __eq__(self, other):
        return (isinstance(other, type(self)) and
                self.qubits == other.qubits)

def test_get_acquaintance_size():
    qubits = cirq.LineQubit.range(5)
    op = OtherOperation(qubits)
    assert op.with_qubits(qubits) == op
    assert get_acquaintance_size(op) == 0

    for s, _ in enumerate(qubits):
        op = ACQUAINT(*qubits[:s + 1])
        assert get_acquaintance_size(op) == s + 1

    part_lens = (2, 2, 2, 2)
    acquaintance_size = 3
    gate = SwapNetworkGate(part_lens, acquaintance_size)
    op = gate(*qubits[:sum(part_lens)])
    assert get_acquaintance_size(op) == 3

    part_lens = (2, 2, 2, 2)
    acquaintance_size = 4
    gate = SwapNetworkGate(part_lens, acquaintance_size)
    op = gate(*qubits[:sum(part_lens)])
    assert get_acquaintance_size(op) == 0

    part_lens = (2, 2, 2, 2)
    acquaintance_size = 1
    gate = SwapNetworkGate(part_lens, acquaintance_size)
    op = gate(*qubits[:sum(part_lens)])
    assert get_acquaintance_size(op) == 0

    part_lens = (2, 2, 2, 2)
    acquaintance_size = 1
    gate = SwapNetworkGate(part_lens, acquaintance_size)
    op = gate(*qubits[:sum(part_lens)])
    assert get_acquaintance_size(op) == 0
