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

import pytest

import cirq
from cirq.circuits import ExpandComposite
from cirq.contrib.swaps.multiswap import MultiswapGate
from cirq.ops import SWAP

def test_multiswap_gate_init():

    with pytest.raises(ValueError):
        MultiswapGate((3,))
    with pytest.raises(ValueError):
        MultiswapGate((1, 2, 3))
    with pytest.raises(ValueError):
        MultiswapGate((0, 2))

    g = MultiswapGate(2)
    assert g.multiplicities == (2, 2)

    g = MultiswapGate((1, 2), swap_gate = cirq.CZ)
    assert g.swap_gate == cirq.CZ

def test_multiswap_gate_eq():
    a = MultiswapGate((2, 2))
    b = MultiswapGate((2, 2))
    c = MultiswapGate((1, 2))
    assert a == b
    assert a != c

def test_multiswap_gate_repr():
    g = MultiswapGate((2, 2))
    assert repr(g) == 'multiSWAP'

def test_multiswap_gate_decomposition():
    qubits = [cirq.NamedQubit(q) for q in 'abcdef']

    expander = ExpandComposite()
    multiswap = MultiswapGate((1, 1), cirq.CZ)(*qubits[:2])
    circuit = cirq.Circuit.from_ops(multiswap)
    expander.optimize_circuit(circuit)
    expected_circuit = cirq.Circuit(
            (cirq.Moment((cirq.CZ(*qubits[:2]),)),))
    assert circuit == expected_circuit

    stopper = lambda op: (op.gate == SWAP)
    expander = ExpandComposite(stopper=stopper)

    multiswap = MultiswapGate((3, 3))(*qubits)
    circuit = cirq.Circuit.from_ops(multiswap)
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

    multiswap = MultiswapGate((2, 4))(*qubits)
    circuit = cirq.Circuit.from_ops(multiswap)
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
    
    
def test_multiswap_gate_wire_symbols():
    qubits = [cirq.NamedQubit(q) for q in 'xyz']
    circuit = cirq.Circuit.from_ops(MultiswapGate((2, 1))(*qubits))
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
