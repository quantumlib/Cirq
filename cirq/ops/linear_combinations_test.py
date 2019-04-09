# Copyright 2019 The Cirq Developers
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

import collections
from typing import Union

import numpy as np
import pytest

import cirq


@pytest.mark.parametrize('terms', (
    {cirq.X: -2, cirq.H: 2},
    {cirq.XX: 1, cirq.YY: 1j, cirq.ZZ: -1},
    {cirq.TOFFOLI: 0.5j, cirq.FREDKIN: 0.5},
))
def test_accepts_consistent_gates(terms):
    combination_1 = cirq.LinearCombinationOfGates(terms)

    combination_2 = cirq.LinearCombinationOfGates({})
    combination_2.update(terms)

    combination_3 = cirq.LinearCombinationOfGates({})
    for gate, coefficient in terms.items():
        combination_3[gate] += coefficient

    assert combination_1 == combination_2 == combination_3


@pytest.mark.parametrize('terms', (
    {cirq.X: -2, cirq.CZ: 2},
    {cirq.X: 1, cirq.YY: 1j, cirq.ZZ: -1},
    {cirq.TOFFOLI: 0.5j, cirq.S: 0.5},
))
def test_rejects_inconsistent_gates(terms):
    with pytest.raises(ValueError):
        cirq.LinearCombinationOfGates(terms)

    combination = cirq.LinearCombinationOfGates({})
    with pytest.raises(ValueError):
        combination.update(terms)

    combination = cirq.LinearCombinationOfGates({})
    with pytest.raises(ValueError):
        for gate, coefficient in terms.items():
            combination[gate] += coefficient


@pytest.mark.parametrize('gate', (
    cirq.X, cirq.Y, cirq.XX, cirq.CZ, cirq.CSWAP, cirq.FREDKIN,
))
def test_empty_accepts_all_gates(gate):
    combination = cirq.LinearCombinationOfGates({})
    combination[gate] = -0.5j
    assert len(combination) == 1


@pytest.mark.parametrize('terms, expected_num_qubits', (
    ({cirq.X: 1}, 1),
    ({cirq.H: 10, cirq.S: -10j}, 1),
    ({cirq.XX: 1, cirq.YY: 2, cirq.ZZ: 3}, 2),
    ({cirq.CCZ: 0.1, cirq.CSWAP: 0.2}, 3),
))
def test_num_qubits(terms, expected_num_qubits):
    combination = cirq.LinearCombinationOfGates(terms)
    assert combination.num_qubits() == expected_num_qubits


def test_empty_has_no_matrix():
    empty = cirq.LinearCombinationOfGates({})
    assert empty.num_qubits() is None
    with pytest.raises(ValueError):
        empty.matrix()


@pytest.mark.parametrize('terms, expected_matrix', (
    ({cirq.I: 2, cirq.X: 3, cirq.Y: 4, cirq.Z: 5j},
        np.array([[2 + 5j, 3 - 4j],
                  [3 + 4j, 2 - 5j]])),
    ({cirq.XX: 0.5, cirq.YY: -0.5}, np.rot90(np.diag([1, 0, 0, 1]))),
    ({cirq.CCZ: 3j}, np.diag([3j, 3j, 3j, 3j, 3j, 3j, 3j, -3j])),
))
def test_matrix(terms, expected_matrix):
    combination = cirq.LinearCombinationOfGates(terms)
    assert np.all(combination.matrix() == expected_matrix)


@pytest.mark.parametrize('terms, expected_expansion', (
    ({cirq.X: 10, cirq.Y: -20}, {'X': 10, 'Y': -20}),
    ({cirq.Y: np.sqrt(0.5), cirq.H: 1},
     {'X': np.sqrt(0.5), 'Y': np.sqrt(0.5), 'Z': np.sqrt(0.5)}),
    ({cirq.X: 2, cirq.H: 1},
     {'X': 2 + np.sqrt(0.5), 'Z': np.sqrt(0.5)}),
    ({cirq.XX: -2, cirq.YY: 3j, cirq.ZZ: 4},
     {'XX': -2, 'YY': 3j, 'ZZ': 4}),
))
def test_pauli_expansion(terms, expected_expansion):
    combination = cirq.LinearCombinationOfGates(terms)
    actual_expansion = cirq.pauli_expansion(combination)
    assert set(actual_expansion.keys()) == set(expected_expansion.keys())
    for name in actual_expansion.keys():
        assert abs(actual_expansion[name] - expected_expansion[name]) < 1e-12


def get_matrix(operator: Union[cirq.LinearCombinationOfGates, cirq.Gate]
        ) -> np.ndarray:
    if isinstance(operator, cirq.LinearCombinationOfGates):
        return operator.matrix()
    return cirq.unitary(operator)


def assert_linear_combinations_of_gates_are_equal(
        actual: cirq.LinearCombinationOfGates,
        expected: cirq.LinearCombinationOfGates) -> None:
    actual_matrix = get_matrix(actual)
    expected_matrix = get_matrix(expected)
    assert np.allclose(actual_matrix, expected_matrix)

    actual_expansion = cirq.pauli_expansion(actual)
    expected_expansion = cirq.pauli_expansion(expected)
    assert set(actual_expansion.keys()) == set(expected_expansion.keys())
    for name in actual_expansion.keys():
        assert abs(actual_expansion[name] - expected_expansion[name]) < 1e-12


@pytest.mark.parametrize('expression, expected_result', (
    ((cirq.X + cirq.Z) / np.sqrt(2), cirq.H),
    (cirq.X - cirq.Y, -cirq.Y + cirq.X),
    (cirq.X + cirq.S - cirq.X, cirq.S),
    (cirq.Y - 2 * cirq.Y, -cirq.Y),
    (cirq.Rx(0.2), np.cos(0.1) * cirq.I - 1j * np.sin(0.1) * cirq.X),
    (1j * cirq.H * 1j, -cirq.H),
    (-1j * cirq.Y, cirq.Ry(np.pi)),
    (np.sqrt(-1j) * cirq.S, cirq.Rz(np.pi / 2)),
    (0.5 * (cirq.IdentityGate(2) + cirq.XX + cirq.YY + cirq.ZZ), cirq.SWAP),
    ((cirq.IdentityGate(2) + 1j * (cirq.XX + cirq.YY) + cirq.ZZ) / 2,
        cirq.ISWAP),
    (cirq.CNOT + 0 * cirq.SWAP, cirq.CNOT),
    (0.5 * cirq.FREDKIN, cirq.FREDKIN / 2),
    (cirq.FREDKIN * 0.5, cirq.FREDKIN / 2),
))
def test_expressions(expression, expected_result):
    assert_linear_combinations_of_gates_are_equal(expression, expected_result)


@pytest.mark.parametrize('gates', (
    (cirq.X, cirq.T, cirq.T, cirq.X, cirq.Z),
    (cirq.CZ, cirq.XX, cirq.YY, cirq.ZZ),
    (cirq.TOFFOLI, cirq.TOFFOLI, cirq.FREDKIN),
))
def test_in_place_manipulations(gates):
    a = cirq.LinearCombinationOfGates({})
    b = cirq.LinearCombinationOfGates({})

    for i, gate in enumerate(gates):
        a += gate
        b -= gate

        prefix = gates[:i + 1]
        expected_a = cirq.LinearCombinationOfGates(collections.Counter(prefix))
        expected_b = -expected_a

        assert_linear_combinations_of_gates_are_equal(a, expected_a)
        assert_linear_combinations_of_gates_are_equal(b, expected_b)
