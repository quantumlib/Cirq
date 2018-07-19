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

from itertools import combinations
from typing import Sequence

import pytest

from cirq import LineQubit
from cirq.circuits import Circuit
from cirq.ops import Gate, gate_features, X
from cirq.contrib.acquaintance.strategy import (
    complete_acquaintance_strategy, AcquaintanceStrategy)
from cirq.contrib.acquaintance.executor import (
        StrategyExecutor, GreedyExecutionStrategy)

class ExampleGate(Gate, gate_features.TextDiagrammable):
    def __init__(self, wire_symbols: Sequence[str]) -> None:
        self._wire_symbols = tuple(wire_symbols)

    def text_diagram_info(self, args: gate_features.TextDiagramInfoArgs
                          ) -> gate_features.TextDiagramInfo:
        return gate_features.TextDiagramInfo(
                wire_symbols=self._wire_symbols)


def test_executor():
    n_qubits = 8
    qubits = LineQubit.range(n_qubits)
    circuit = complete_acquaintance_strategy(qubits, 2)

    gates = {(i, j): ExampleGate([str(k) for k in ij])
             for ij in combinations(range(n_qubits), 2)
             for i, j in (ij, ij[::-1])}
    initial_mapping = {q: i for i, q in enumerate(qubits)}
    execution_strategy = GreedyExecutionStrategy(gates, initial_mapping)
    executor = StrategyExecutor(execution_strategy)

    with pytest.raises(NotImplementedError):
        bad_gates = {(0,): ExampleGate(['0']), (0, 1): ExampleGate(['0', '1'])}
        GreedyExecutionStrategy(bad_gates, initial_mapping)

    with pytest.raises(ValueError):
        executor(Circuit())

    with pytest.raises(TypeError):
        bad_strategy = AcquaintanceStrategy(Circuit())
        bad_strategy.append(X(qubits[0]))
        executor(bad_strategy)

    executor(circuit)
    actual_text_diagram = circuit.to_text_diagram().strip()
    expected_text_diagram = """
0: ───0───1───╲0╱─────────────────1───3───╲0╱─────────────────3───5───╲0╱─────────────────5───7───╲0╱─────────────────
      │   │   │                   │   │   │                   │   │   │                   │   │   │
1: ───1───0───╱1╲───0───3───╲0╱───3───1───╱1╲───1───5───╲0╱───5───3───╱1╲───3───7───╲0╱───7───5───╱1╲───5───6───╲0╱───
                    │   │   │                   │   │   │                   │   │   │                   │   │   │
2: ───2───3───╲0╱───3───0───╱1╲───0───5───╲0╱───5───1───╱1╲───1───7───╲0╱───7───3───╱1╲───3───6───╲0╱───6───5───╱1╲───
      │   │   │                   │   │   │                   │   │   │                   │   │   │
3: ───3───2───╱1╲───2───5───╲0╱───5───0───╱1╲───0───7───╲0╱───7───1───╱1╲───1───6───╲0╱───6───3───╱1╲───3───4───╲0╱───
                    │   │   │                   │   │   │                   │   │   │                   │   │   │
4: ───4───5───╲0╱───5───2───╱1╲───2───7───╲0╱───7───0───╱1╲───0───6───╲0╱───6───1───╱1╲───1───4───╲0╱───4───3───╱1╲───
      │   │   │                   │   │   │                   │   │   │                   │   │   │
5: ───5───4───╱1╲───4───7───╲0╱───7───2───╱1╲───2───6───╲0╱───6───0───╱1╲───0───4───╲0╱───4───1───╱1╲───1───2───╲0╱───
                    │   │   │                   │   │   │                   │   │   │                   │   │   │
6: ───6───7───╲0╱───7───4───╱1╲───4───6───╲0╱───6───2───╱1╲───2───4───╲0╱───4───0───╱1╲───0───2───╲0╱───2───1───╱1╲───
      │   │   │                   │   │   │                   │   │   │                   │   │   │
7: ───7───6───╱1╲─────────────────6───4───╱1╲─────────────────4───2───╱1╲─────────────────2───0───╱1╲─────────────────
    """.strip()
    assert actual_text_diagram == expected_text_diagram
