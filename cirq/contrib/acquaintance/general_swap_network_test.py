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

from string import ascii_lowercase as alphabet

from cirq import NamedQubit
from cirq.circuits import Circuit, ExpandComposite
from cirq.contrib.acquaintance.gates import (
        SwapNetworkGate, CircularShiftGate)

def test_swap_network_gate():
    qubits = tuple(NamedQubit(s) for s in alphabet)

    acquaintance_size = 3
    n_parts = 3
    part_lens = (acquaintance_size - 1,) * n_parts
    n_qubits = sum(part_lens)
    swap_network_op = SwapNetworkGate(part_lens,
        acquaintance_size=acquaintance_size)(*qubits[:n_qubits])
    swap_network = Circuit.from_ops(swap_network_op)
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

    no_decomp = lambda op: isinstance(op.gate, CircularShiftGate)
    expander = ExpandComposite(no_decomp=no_decomp)
    expander(swap_network)
    actual_text_diagram = swap_network.to_text_diagram().strip()
    expected_text_diagram = """
a: ───█───────╲0╱───█─────────────────█───────────╲0╱───█───────╲0╱───
      │       │     │                 │           │     │       │
b: ───█───█───╲1╱───█───█─────────────█───█───────╲1╱───█───█───╱1╲───
      │   │   │     │   │             │   │       │     │   │
c: ───█───█───╱2╲───█───█───█───╲0╱───█───█───█───╱2╲───█───█───╲0╱───
          │   │         │   │   │         │   │   │         │   │
d: ───────█───╱3╲───█───█───█───╲1╱───█───█───█───╱3╲───────█───╱1╲───
                    │       │   │     │       │
e: ─────────────────█───────█───╱2╲───█───────█───╲0╱─────────────────
                    │           │     │           │
f: ─────────────────█───────────╱3╲───█───────────╱1╲─────────────────
    """.strip()
    assert actual_text_diagram == expected_text_diagram

    no_decomp = lambda op: isinstance(op.gate, CircularShiftGate)
    expander = ExpandComposite(no_decomp=no_decomp)

    acquaintance_size = 3
    n_parts = 6
    part_lens = (1,) * n_parts
    n_qubits = sum(part_lens)
    swap_network_op = SwapNetworkGate(part_lens,
        acquaintance_size=acquaintance_size)(*qubits[:n_qubits])
    swap_network = Circuit.from_ops(swap_network_op)

    expander(swap_network)
    actual_text_diagram = swap_network.to_text_diagram().strip()
    print(actual_text_diagram)
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
