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
from cirq.circuits import ExpandComposite
from cirq.contrib.acquaintance.shift import CircularShiftGate
from cirq.ops import SWAP, TextDiagramInfoArgs

def test_circular_shift_gate_init():

    g = CircularShiftGate(2)
    assert g.shift == 2

    g = CircularShiftGate(1, swap_gate = cirq.CZ)
    assert g.swap_gate == cirq.CZ

def test_circular_shift_gate_unknown_qubit_count():
    g = CircularShiftGate(2)
    args = TextDiagramInfoArgs.UNINFORMED_DEFAULT
    assert g.text_diagram_info(args) == NotImplemented

def test_circular_shift_gate_eq():
    a = CircularShiftGate(1)
    b = CircularShiftGate(1)
    c = CircularShiftGate(2)
    assert a == b
    assert a != c

def test_circular_shift_gate_repr():
    g = CircularShiftGate(2)
    assert repr(g) == 'CircularShiftGate'

def test_circular_shift_gate_decomposition():
    qubits = [cirq.NamedQubit(q) for q in 'abcdef']

    expander = ExpandComposite()
    circular_shift = CircularShiftGate(1, cirq.CZ)(*qubits[:2])
    circuit = cirq.Circuit.from_ops(circular_shift)
    expander.optimize_circuit(circuit)
    expected_circuit = cirq.Circuit(
            (cirq.Moment((cirq.CZ(*qubits[:2]),)),))
    assert circuit == expected_circuit

    stopper = lambda op: (op.gate == SWAP)
    expander = ExpandComposite(stopper=stopper)

    circular_shift = CircularShiftGate(3)(*qubits)
    circuit = cirq.Circuit.from_ops(circular_shift)
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

    circular_shift = CircularShiftGate(2)(*qubits)
    circuit = cirq.Circuit.from_ops(circular_shift)
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
    circuit = cirq.Circuit.from_ops(CircularShiftGate(2)(*qubits))
    actual_text_diagram = circuit.to_text_diagram().strip()
    expected_text_diagram = """
x: ───╲0╱───
      │
y: ───╲1╱───
      │
z: ───╱2╲───
    """.strip()
    assert actual_text_diagram == expected_text_diagram

    actual_text_diagram = circuit.to_text_diagram(use_unicode_characters=0)
    expected_text_diagram = r"""
x: ---\0/---
      |
y: ---\1/---
      |
z: ---/2\---
    """.strip()
    assert actual_text_diagram.strip() == expected_text_diagram
