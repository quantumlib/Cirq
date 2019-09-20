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

import pytest

import cirq
import cirq.contrib.acquaintance as cca


def test_complete_acquaintance_strategy():
    qubits = [cirq.NamedQubit(s) for s in alphabet]

    with pytest.raises(ValueError):
        _ = cca.complete_acquaintance_strategy(qubits, -1)

    empty_strategy = cca.complete_acquaintance_strategy(qubits)
    assert empty_strategy._moments == []

    trivial_strategy = cca.complete_acquaintance_strategy(qubits[:4], 1)
    actual_text_diagram = trivial_strategy.to_text_diagram().strip()
    expected_text_diagram = """
a: ───█───

b: ───█───

c: ───█───

d: ───█───
    """.strip()
    assert actual_text_diagram == expected_text_diagram
    assert cca.get_acquaintance_size(trivial_strategy) == 1

    is_shift_or_lin_perm = lambda op: isinstance(op.gate,
            (cca.CircularShiftGate, cca.LinearPermutationGate))
    expand = cirq.ExpandComposite(no_decomp=is_shift_or_lin_perm)

    quadratic_strategy = cca.complete_acquaintance_strategy(qubits[:8], 2)
    actual_text_diagram = quadratic_strategy.to_text_diagram().strip()
    expected_text_diagram = """
a: ───×(0,0)───
      │
b: ───×(1,0)───
      │
c: ───×(2,0)───
      │
d: ───×(3,0)───
      │
e: ───×(4,0)───
      │
f: ───×(5,0)───
      │
g: ───×(6,0)───
      │
h: ───×(7,0)───
    """.strip()
    assert actual_text_diagram == expected_text_diagram
    assert cca.get_acquaintance_size(quadratic_strategy) == 2

    expand(quadratic_strategy)
    actual_text_diagram = quadratic_strategy.to_text_diagram(
            transpose=True).strip()
    expected_text_diagram = '\n'.join((
        "a   b   c   d   e   f   g   h        ".strip(),
        "│   │   │   │   │   │   │   │        ".strip(),
        "█───█   █───█   █───█   █───█        ".strip(),
        "│   │   │   │   │   │   │   │        ".strip(),
        "╲0╱─╱1╲ ╲0╱─╱1╲ ╲0╱─╱1╲ ╲0╱─╱1╲      ".strip(),
        "│   │   │   │   │   │   │   │        ".strip(),
        "│   █───█   █───█   █───█   │        ".strip(),
        "│   │   │   │   │   │   │   │        ".strip(),
        "│   ╲0╱─╱1╲ ╲0╱─╱1╲ ╲0╱─╱1╲ │        ".strip(),
        "│   │   │   │   │   │   │   │        ".strip(),
        "█───█   █───█   █───█   █───█        ".strip(),
        "│   │   │   │   │   │   │   │        ".strip(),
        "╲0╱─╱1╲ ╲0╱─╱1╲ ╲0╱─╱1╲ ╲0╱─╱1╲      ".strip(),
        "│   │   │   │   │   │   │   │        ".strip(),
        "│   █───█   █───█   █───█   │        ".strip(),
        "│   │   │   │   │   │   │   │        ".strip(),
        "│   ╲0╱─╱1╲ ╲0╱─╱1╲ ╲0╱─╱1╲ │        ".strip(),
        "│   │   │   │   │   │   │   │        ".strip(),
        "█───█   █───█   █───█   █───█        ".strip(),
        "│   │   │   │   │   │   │   │        ".strip(),
        "╲0╱─╱1╲ ╲0╱─╱1╲ ╲0╱─╱1╲ ╲0╱─╱1╲      ".strip(),
        "│   │   │   │   │   │   │   │        ".strip(),
        "│   █───█   █───█   █───█   │        ".strip(),
        "│   │   │   │   │   │   │   │        ".strip(),
        "│   ╲0╱─╱1╲ ╲0╱─╱1╲ ╲0╱─╱1╲ │        ".strip(),
        "│   │   │   │   │   │   │   │        ".strip(),
        "█───█   █───█   █───█   █───█        ".strip(),
        "│   │   │   │   │   │   │   │        ".strip(),
        "╲0╱─╱1╲ ╲0╱─╱1╲ ╲0╱─╱1╲ ╲0╱─╱1╲      ".strip(),
        "│   │   │   │   │   │   │   │        ".strip(),
        "│   █───█   █───█   █───█   │        ".strip(),
        "│   │   │   │   │   │   │   │        ".strip(),
        "│   ╲0╱─╱1╲ ╲0╱─╱1╲ ╲0╱─╱1╲ │        ".strip(),
        "│   │   │   │   │   │   │   │        ".strip()
        ))
    assert actual_text_diagram == expected_text_diagram
    assert cca.get_acquaintance_size(quadratic_strategy) == 2

    cubic_strategy = cca.complete_acquaintance_strategy(qubits[:4], 3)
    actual_text_diagram = cubic_strategy.to_text_diagram(
            transpose=True).strip()
    expected_text_diagram = """
a      b      c      d
│      │      │      │
×(0,0)─×(0,1)─×(1,0)─×(1,1)
│      │      │      │
╱1╲────╲0╱    ╱1╲────╲0╱
│      │      │      │
×(0,0)─×(1,0)─×(1,1)─×(2,0)
│      │      │      │
│      ╲0╱────╱1╲    │
│      │      │      │
×(0,0)─×(0,1)─×(1,0)─×(1,1)
│      │      │      │
╱1╲────╲0╱    ╱1╲────╲0╱
│      │      │      │
×(0,0)─×(1,0)─×(1,1)─×(2,0)
│      │      │      │
│      ╲0╱────╱1╲    │
│      │      │      │
    """.strip()
    assert actual_text_diagram == expected_text_diagram
    assert cca.get_acquaintance_size(cubic_strategy) == 3

def test_rectification():
    qubits = cirq.LineQubit.range(4)

    with pytest.raises(TypeError):
        cca.rectify_acquaintance_strategy(cirq.Circuit())

    perm_gate = cca.SwapPermutationGate()
    operations = [
        perm_gate(*qubits[:2]), cca.acquaint(*qubits[2:]),
        cca.acquaint(*qubits[:2]), perm_gate(*qubits[2:])
        ]

    strategy = cirq.Circuit(operations,
                            device=cca.UnconstrainedAcquaintanceDevice)
    cca.rectify_acquaintance_strategy(strategy)
    actual_text_diagram = strategy.to_text_diagram().strip()
    expected_text_diagram = """
0: ───────0↦1─────────█───
          │           │
1: ───────1↦0─────────█───

2: ───█─────────0↦1───────
      │         │
3: ───█─────────1↦0───────
    """.strip()
    assert actual_text_diagram == expected_text_diagram

    strategy = cirq.Circuit(operations,
                            device=cca.UnconstrainedAcquaintanceDevice)
    cca.rectify_acquaintance_strategy(strategy, False)
    actual_text_diagram = strategy.to_text_diagram()
    expected_text_diagram = """
0: ───0↦1───────█─────────
      │         │
1: ───1↦0───────█─────────

2: ─────────█───────0↦1───
            │       │
3: ─────────█───────1↦0───
    """.strip()
    assert actual_text_diagram == expected_text_diagram

def test_replace_acquaintance_with_swap_network():
    with pytest.raises(TypeError):
        qubits = cirq.LineQubit.range(3)
        cca.replace_acquaintance_with_swap_network(cirq.Circuit(), qubits)
