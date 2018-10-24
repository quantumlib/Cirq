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
from typing import Sequence, Dict, Tuple

import numpy as np
import pytest

import cirq
from cirq.contrib.acquaintance.strategy import (
    complete_acquaintance_strategy)
from cirq.contrib.acquaintance.permutation import (
        LinearPermutationGate)
from cirq.contrib.acquaintance.executor import (
        StrategyExecutor, GreedyExecutionStrategy)


class ExampleGate(cirq.Gate):
    def __init__(self, wire_symbols: Sequence[str]) -> None:
        self._wire_symbols = tuple(wire_symbols)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs):
        return self._wire_symbols


def test_executor_explicit():
    n_qubits = 8
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = complete_acquaintance_strategy(qubits, 2)

    gates = {(i, j): ExampleGate([str(k) for k in ij])
             for ij in combinations(range(n_qubits), 2)
             for i, j in (ij, ij[::-1])}
    initial_mapping = {q: i for i, q in enumerate(sorted(qubits))}
    execution_strategy = GreedyExecutionStrategy(gates, initial_mapping)
    executor = StrategyExecutor(execution_strategy)

    with pytest.raises(NotImplementedError):
        bad_gates = {(0,): ExampleGate(['0']), (0, 1): ExampleGate(['0', '1'])}
        GreedyExecutionStrategy(bad_gates, initial_mapping)

    with pytest.raises(TypeError):
        executor(cirq.Circuit())

    with pytest.raises(TypeError):
        bad_strategy = cirq.Circuit.from_ops(cirq.X(qubits[0]))
        executor(bad_strategy)

    with pytest.raises(TypeError):
        op = cirq.X(qubits[0])
        bad_strategy = cirq.Circuit.from_ops(op)
        executor.optimization_at(bad_strategy, 0, op)

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
    print(actual_text_diagram)
    assert actual_text_diagram == expected_text_diagram


class DiagonalGate(cirq.Gate):
    def __init__(self, n_qubits: int, diagonal: np.ndarray) -> None:
        dimension = 2 ** n_qubits
        if (diagonal.shape != (dimension,) or not
            np.allclose(
                np.absolute(diagonal), np.ones(dimension))):
            raise ValueError('Diagonal must be an (2**n_qubits)-dimensional '
                    'vector with unit-norm entries.')
        self.n_qubits = n_qubits
        self.diagonal = diagonal

    def _unitary_(self) -> np.ndarray:
        return np.diag(self.diagonal)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs):
        qubit_count = len(self.diagonal).bit_length() - 1
        assert (args.known_qubit_count is None or
                args.known_qubit_count == qubit_count)
        return ('Diag',) * qubit_count

    @staticmethod
    def random(n_qubits: int):
        dimension = 2 ** n_qubits
        diagonal = np.exp(2j * np.pi * np.random.random(dimension))
        return DiagonalGate(n_qubits, diagonal)


def test_diagonal_gate():
    with pytest.raises(ValueError):
        diagonal = np.exp(2j * np.pi * np.random.random(5))
        DiagonalGate(2, diagonal)
    with pytest.raises(ValueError):
        diagonal = np.ndarray(range(4))
        DiagonalGate(2, diagonal)
    gate = DiagonalGate.random(2)
    assert cirq.circuit_diagram_info(gate) == cirq.CircuitDiagramInfo(
        wire_symbols=('Diag', 'Diag'))

    qubits = cirq.LineQubit.range(2)
    gate = DiagonalGate.random(2)
    circuit = cirq.Circuit.from_ops([gate(*qubits)])
    actual_text_diagram = circuit.to_text_diagram()
    expected_text_diagram = """
0: ───Diag───
      │
1: ───Diag───
    """.strip()
    assert actual_text_diagram == expected_text_diagram


def random_diagonal_gates(n_qubits: int,
                 acquaintance_size: int
                 ) -> Dict[Tuple[cirq.QubitId, ...], cirq.Gate]:

    return {Q: DiagonalGate.random(acquaintance_size)
             for Q in
             combinations(cirq.LineQubit.range(n_qubits), acquaintance_size)}


@pytest.mark.parametrize('n_qubits, acquaintance_size, gates',
    [(n_qubits, acquaintance_size,
      random_diagonal_gates(n_qubits, acquaintance_size))
      for acquaintance_size, n_qubits in
      ([(2, n) for n in range(2, 9)] +
       [(3, n) for n in range(3, 9)] +
       [(4, n) for n in (4, 7)] +
       [(5, n) for n in (5, 6)])
      for _ in range(2)
      ])
def test_executor_random(n_qubits: int,
                         acquaintance_size: int,
                         gates: Dict[Tuple[cirq.QubitId, ...], cirq.Gate]):
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = complete_acquaintance_strategy(qubits, acquaintance_size)

    logical_circuit = cirq.Circuit.from_ops([g(*Q) for Q, g in gates.items()])
    expected_unitary = logical_circuit.to_unitary_matrix()

    initial_mapping = {q: q for q in qubits}
    execution_strategy = GreedyExecutionStrategy(gates, initial_mapping)
    executor = StrategyExecutor(execution_strategy)
    final_mapping = executor(circuit)
    permutation = {q.x: qq.x for q, qq in final_mapping.items()}
    circuit.append(LinearPermutationGate(permutation)(*qubits))
    actual_unitary = circuit.to_unitary_matrix()

    np.testing.assert_allclose(
            actual=actual_unitary,
            desired=expected_unitary,
            verbose=True)
