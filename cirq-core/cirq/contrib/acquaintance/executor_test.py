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
from string import ascii_lowercase
from typing import Sequence, Dict, Tuple

import numpy as np
import pytest

import cirq
import cirq.testing as ct
import cirq.contrib.acquaintance as cca


class ExampleGate(cirq.Gate):
    def __init__(self, wire_symbols: Sequence[str]) -> None:
        self._num_qubits = len(wire_symbols)
        self._wire_symbols = tuple(wire_symbols)

    def num_qubits(self) -> int:
        return self._num_qubits

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs):
        return self._wire_symbols


def test_executor_explicit():
    num_qubits = 8
    qubits = cirq.LineQubit.range(num_qubits)
    circuit = cca.complete_acquaintance_strategy(qubits, 2)

    gates = {
        (i, j): ExampleGate([str(k) for k in ij])
        for ij in combinations(range(num_qubits), 2)
        for i, j in (ij, ij[::-1])
    }
    initial_mapping = {q: i for i, q in enumerate(sorted(qubits))}
    execution_strategy = cca.GreedyExecutionStrategy(gates, initial_mapping)
    executor = cca.StrategyExecutor(execution_strategy)

    with pytest.raises(NotImplementedError):
        bad_gates = {(0,): ExampleGate(['0']), (0, 1): ExampleGate(['0', '1'])}
        cca.GreedyExecutionStrategy(bad_gates, initial_mapping)

    with pytest.raises(TypeError):
        bad_strategy = cirq.Circuit(cirq.X(qubits[0]))
        executor(bad_strategy)

    with pytest.raises(TypeError):
        op = cirq.X(qubits[0])
        bad_strategy = cirq.Circuit(op)
        executor.optimization_at(bad_strategy, 0, op)

    executor(circuit)
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
    ct.assert_has_diagram(circuit, expected_text_diagram)


def random_diagonal_gates(
    num_qubits: int, acquaintance_size: int
) -> Dict[Tuple[cirq.Qid, ...], cirq.Gate]:

    return {
        Q: cirq.DiagonalGate(np.random.random(2 ** acquaintance_size))
        for Q in combinations(cirq.LineQubit.range(num_qubits), acquaintance_size)
    }


@pytest.mark.parametrize(
    'num_qubits, acquaintance_size, gates',
    [
        (num_qubits, acquaintance_size, random_diagonal_gates(num_qubits, acquaintance_size))
        for acquaintance_size, num_qubits in (
            [(2, n) for n in range(2, 9)] + [(3, n) for n in range(3, 8)] + [(4, 4), (4, 6), (5, 5)]
        )
        for _ in range(2)
    ],
)
def test_executor_random(
    num_qubits: int, acquaintance_size: int, gates: Dict[Tuple[cirq.Qid, ...], cirq.Gate]
):
    qubits = cirq.LineQubit.range(num_qubits)
    circuit = cca.complete_acquaintance_strategy(qubits, acquaintance_size)

    logical_circuit = cirq.Circuit([g(*Q) for Q, g in gates.items()])
    expected_unitary = logical_circuit.unitary()

    initial_mapping = {q: q for q in qubits}
    final_mapping = cca.GreedyExecutionStrategy(gates, initial_mapping)(circuit)
    permutation = {q.x: qq.x for q, qq in final_mapping.items()}
    circuit.append(cca.LinearPermutationGate(num_qubits, permutation)(*qubits))
    actual_unitary = circuit.unitary()

    np.testing.assert_allclose(actual=actual_unitary, desired=expected_unitary, verbose=True)


def test_acquaintance_operation():
    n = 5
    physical_qubits = tuple(cirq.LineQubit.range(n))
    logical_qubits = tuple(cirq.NamedQubit(s) for s in ascii_lowercase[:n])
    int_indices = tuple(range(n))
    with pytest.raises(ValueError):
        cca.AcquaintanceOperation(physical_qubits[:3], int_indices[:4])
    for logical_indices in (logical_qubits, int_indices):
        op = cca.AcquaintanceOperation(physical_qubits, logical_indices)
        assert op.logical_indices == logical_indices
        assert op.qubits == physical_qubits
        wire_symbols = tuple(f'({i})' for i in logical_indices)
        assert cirq.circuit_diagram_info(op) == cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)
