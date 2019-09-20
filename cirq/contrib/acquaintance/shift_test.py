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

import cirq
import cirq.contrib.acquaintance as cca


def test_circular_shift_gate_init():
    g = cca.CircularShiftGate(4, 2)
    assert g.num_qubits() == 4
    assert g.shift == 2

    g = cca.CircularShiftGate(4, 1, swap_gate = cirq.CZ)
    assert g.swap_gate == cirq.CZ


def test_circular_shift_gate_eq():
    equals_tester = cirq.testing.EqualsTester()
    equals_tester.add_equality_group(cca.CircularShiftGate(4, 1),
                                     cca.CircularShiftGate(4, 1))
    equals_tester.add_equality_group(
        cca.CircularShiftGate(4, 1, swap_gate=cirq.CZ))
    equals_tester.add_equality_group(cca.CircularShiftGate(4, 2))
    equals_tester.add_equality_group(cca.CircularShiftGate(3, 2))
    equals_tester.add_equality_group(
        cca.CircularShiftGate(3, 2, swap_gate=cirq.CZ))


def test_circular_shift_gate_permutation():
    assert (cca.CircularShiftGate(3, 4).permutation() ==
            {0: 2, 1: 0, 2: 1})
    assert (cca.CircularShiftGate(4, 0).permutation() ==
            {0: 0, 1: 1, 2: 2, 3: 3})

    assert (cca.CircularShiftGate(5, 2).permutation() ==
            {0:3, 1: 4, 2: 0, 3: 1, 4: 2})


def test_circular_shift_gate_repr():
    g = cca.CircularShiftGate(3, 2)
    cirq.testing.assert_equivalent_repr(g)


def test_circular_shift_gate_decomposition():
    qubits = [cirq.NamedQubit(q) for q in 'abcdef']

    expander = cirq.ExpandComposite()
    circular_shift = cca.CircularShiftGate(2, 1, cirq.CZ)(*qubits[:2])
    circuit = cirq.Circuit(circular_shift)
    expander.optimize_circuit(circuit)
    expected_circuit = cirq.Circuit(
            (cirq.Moment((cirq.CZ(*qubits[:2]),)),))
    assert circuit == expected_circuit

    no_decomp = lambda op: (isinstance(op, cirq.GateOperation) and
                            op.gate == cirq.SWAP)
    expander = cirq.ExpandComposite(no_decomp=no_decomp)

    circular_shift = cca.CircularShiftGate(6, 3)(*qubits)
    circuit = cirq.Circuit(circular_shift)
    expander.optimize_circuit(circuit)
    actual_text_diagram = circuit.to_text_diagram().strip()
    expected_text_diagram = """
a: ───────────×───────────
              │
b: ───────×───×───×───────
          │       │
c: ───×───×───×───×───×───
      │       │       │
d: ───×───×───×───×───×───
          │       │
e: ───────×───×───×───────
              │
f: ───────────×───────────
    """.strip()
    assert actual_text_diagram == expected_text_diagram

    circular_shift = cca.CircularShiftGate(6, 2)(*qubits)
    circuit = cirq.Circuit(circular_shift)
    expander.optimize_circuit(circuit)
    actual_text_diagram = circuit.to_text_diagram().strip()
    expected_text_diagram = """
a: ───────×───────────────
          │
b: ───×───×───×───────────
      │       │
c: ───×───×───×───×───────
          │       │
d: ───────×───×───×───×───
              │       │
e: ───────────×───×───×───
                  │
f: ───────────────×───────
    """.strip()
    assert actual_text_diagram == expected_text_diagram


def test_circular_shift_gate_wire_symbols():
    qubits = [cirq.NamedQubit(q) for q in 'xyz']
    circuit = cirq.Circuit(cca.CircularShiftGate(3, 2)(*qubits))
    actual_text_diagram = circuit.to_text_diagram().strip()
    expected_text_diagram = """
x: ───╲0╱───
      │
y: ───╲1╱───
      │
z: ───╱2╲───
    """.strip()
    assert actual_text_diagram == expected_text_diagram

    actual_text_diagram = circuit.to_text_diagram(use_unicode_characters=False)
    expected_text_diagram = r"""
x: ---\0/---
      |
y: ---\1/---
      |
z: ---/2\---
    """.strip()
    assert actual_text_diagram.strip() == expected_text_diagram
