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

_ = 0.0  # Make matrices readable by visually hiding off-diagonal elements.
q0, q1, q2, q3 = cirq.LineQubit.range(4)


@pytest.mark.parametrize('terms', (
    {cirq.X: -2, cirq.H: 2},
    {cirq.XX: 1, cirq.YY: 1j, cirq.ZZ: -1},
    {cirq.TOFFOLI: 0.5j, cirq.FREDKIN: 0.5},
))
def test_linear_combination_of_gates_accepts_consistent_gates(terms):
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
def test_linear_combination_of_gates_rejects_inconsistent_gates(terms):
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
def test_empty_linear_combination_of_gates_accepts_all_gates(gate):
    combination = cirq.LinearCombinationOfGates({})
    combination[gate] = -0.5j
    assert len(combination) == 1


@pytest.mark.parametrize('terms, expected_num_qubits', (
    ({cirq.X: 1}, 1),
    ({cirq.H: 10, cirq.S: -10j}, 1),
    ({cirq.XX: 1, cirq.YY: 2, cirq.ZZ: 3}, 2),
    ({cirq.CCZ: 0.1, cirq.CSWAP: 0.2}, 3),
))
def test_linear_combination_of_gates_has_correct_num_qubits(
        terms, expected_num_qubits):
    combination = cirq.LinearCombinationOfGates(terms)
    assert combination.num_qubits() == expected_num_qubits


def test_empty_linear_combination_of_gates_has_no_matrix():
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
def test_linear_combination_of_gates_has_correct_matrix(terms, expected_matrix):
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
def test_linear_combination_of_gates_has_correct_pauli_expansion(
        terms, expected_expansion):
    combination = cirq.LinearCombinationOfGates(terms)
    actual_expansion = cirq.pauli_expansion(combination)
    assert set(actual_expansion.keys()) == set(expected_expansion.keys())
    for name in actual_expansion.keys():
        assert abs(actual_expansion[name] - expected_expansion[name]) < 1e-12


def get_matrix(operator: Union[cirq.Gate, cirq.GateOperation, cirq.
                               LinearCombinationOfGates, cirq.
                               LinearCombinationOfOperations]) -> np.ndarray:
    if isinstance(
            operator,
        (cirq.LinearCombinationOfGates, cirq.LinearCombinationOfOperations)):
        return operator.matrix()
    return cirq.unitary(operator)


def assert_linear_combinations_are_equal(
        actual: Union[cirq.LinearCombinationOfGates, cirq.
                      LinearCombinationOfOperations],
        expected: Union[cirq.LinearCombinationOfGates, cirq.
                        LinearCombinationOfOperations]) -> None:
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
def test_gate_expressions(expression, expected_result):
    assert_linear_combinations_are_equal(expression, expected_result)


@pytest.mark.parametrize('gates', (
    (cirq.X, cirq.T, cirq.T, cirq.X, cirq.Z),
    (cirq.CZ, cirq.XX, cirq.YY, cirq.ZZ),
    (cirq.TOFFOLI, cirq.TOFFOLI, cirq.FREDKIN),
))
def test_in_place_manipulations_of_linear_combination_of_gates(gates):
    a = cirq.LinearCombinationOfGates({})
    b = cirq.LinearCombinationOfGates({})

    for i, gate in enumerate(gates):
        a += gate
        b -= gate

        prefix = gates[:i + 1]
        expected_a = cirq.LinearCombinationOfGates(collections.Counter(prefix))
        expected_b = -expected_a

        assert_linear_combinations_are_equal(a, expected_a)
        assert_linear_combinations_are_equal(b, expected_b)


@pytest.mark.parametrize('op', (
    cirq.X(q0),
    cirq.Y(q1),
    cirq.XX(q0, q1),
    cirq.CZ(q0, q1),
    cirq.FREDKIN(q0, q1, q2),
    cirq.ControlledOperation((q0, q1), cirq.H(q2)),
    cirq.ParallelGateOperation(cirq.X, (q0, q1, q2)),
    cirq.PauliString({
        q0: cirq.X,
        q1: cirq.Y,
        q2: cirq.Z
    }),
))
def test_empty_linear_combination_of_operations_accepts_all_operations(op):
    combination = cirq.LinearCombinationOfOperations({})
    combination[op] = -0.5j
    assert len(combination) == 1


@pytest.mark.parametrize('terms', (
    {
        cirq.X(q0): -2,
        cirq.H(q0): 2
    },
    {
        cirq.X(q0): -2,
        cirq.H(q1): 2j
    },
    {
        cirq.X(q0): 1,
        cirq.CZ(q0, q1): 3
    },
    {
        cirq.X(q0): 1 + 1j,
        cirq.CZ(q1, q2): 0.5
    },
))
def test_linear_combination_of_operations_is_consistent(terms):
    combination_1 = cirq.LinearCombinationOfOperations(terms)

    combination_2 = cirq.LinearCombinationOfOperations({})
    combination_2.update(terms)

    combination_3 = cirq.LinearCombinationOfOperations({})
    for gate, coefficient in terms.items():
        combination_3[gate] += coefficient

    assert combination_1 == combination_2 == combination_3


@pytest.mark.parametrize('terms, expected_qubits', (
    ({}, ()),
    ({
        cirq.I(q0): 1,
        cirq.H(q0): 1e-3j
    }, (q0,)),
    ({
        cirq.X(q0): 1j,
        cirq.H(q1): 2j
    }, (q0, q1)),
    ({
        cirq.Y(q0): -1,
        cirq.CZ(q0, q1): 3e3
    }, (q0, q1)),
    ({
        cirq.Z(q0): -1j,
        cirq.CNOT(q1, q2): 0.25
    }, (q0, q1, q2)),
))
def test_linear_combination_of_operations_has_correct_qubits(
        terms, expected_qubits):
    combination = cirq.LinearCombinationOfOperations(terms)
    assert combination.qubits == expected_qubits


@pytest.mark.parametrize('terms, expected_matrix', (
    ({}, np.array([0])),
    ({
        cirq.I(q0): 2,
        cirq.X(q0): 3,
        cirq.Y(q0): 4,
        cirq.Z(q0): 5j
    }, np.array([
        [2 + 5j, 3 - 4j],
        [3 + 4j, 2 - 5j],
    ])),
    ({
        cirq.X(q0): 2,
        cirq.Y(q1): 3
    }, np.array([
        [0, -3j, 2, 0],
        [3j, 0, 0, 2],
        [2, 0, 0, -3j],
        [0, 2, 3j, 0],
    ])),
    ({
        cirq.XX(q0, q1): 0.5,
        cirq.YY(q0, q1): -0.5
    }, np.rot90(np.diag([1, 0, 0, 1]))),
    ({
        cirq.CCZ(q0, q1, q2): 3j
    }, np.diag([3j, 3j, 3j, 3j, 3j, 3j, 3j, -3j])),
    ({
        cirq.I(q0): 0.1,
        cirq.CNOT(q1, q2): 1
    },
     np.array([
         [1.1, _, _, _, _, _, _, _],
         [_, 1.1, _, _, _, _, _, _],
         [_, _, 0.1, 1, _, _, _, _],
         [_, _, 1, 0.1, _, _, _, _],
         [_, _, _, _, 1.1, _, _, _],
         [_, _, _, _, _, 1.1, _, _],
         [_, _, _, _, _, _, 0.1, 1],
         [_, _, _, _, _, _, 1, 0.1],
     ])),
    ({
        cirq.I(q1): 0.1,
        cirq.CNOT(q0, q2): 1
    },
     np.array([
         [1.1, _, _, _, _, _, _, _],
         [_, 1.1, _, _, _, _, _, _],
         [_, _, 1.1, _, _, _, _, _],
         [_, _, _, 1.1, _, _, _, _],
         [_, _, _, _, 0.1, 1, _, _],
         [_, _, _, _, 1, 0.1, _, _],
         [_, _, _, _, _, _, 0.1, 1],
         [_, _, _, _, _, _, 1, 0.1],
     ])),
    ({
        cirq.I(q2): 0.1,
        cirq.CNOT(q0, q1): 1
    },
     np.array([
         [1.1, _, _, _, _, _, _, _],
         [_, 1.1, _, _, _, _, _, _],
         [_, _, 1.1, _, _, _, _, _],
         [_, _, _, 1.1, _, _, _, _],
         [_, _, _, _, 0.1, _, 1, _],
         [_, _, _, _, _, 0.1, _, 1],
         [_, _, _, _, 1, _, 0.1, _],
         [_, _, _, _, _, 1, _, 0.1],
     ])),
    ({
        cirq.I(q0): 0.1,
        cirq.ControlledGate(cirq.Y).on(q1, q2): 1
    },
     np.array([
         [1.1, _, _, _, _, _, _, _],
         [_, 1.1, _, _, _, _, _, _],
         [_, _, 0.1, -1j, _, _, _, _],
         [_, _, 1j, 0.1, _, _, _, _],
         [_, _, _, _, 1.1, _, _, _],
         [_, _, _, _, _, 1.1, _, _],
         [_, _, _, _, _, _, 0.1, -1j],
         [_, _, _, _, _, _, 1j, 0.1],
     ])),
    ({
        cirq.I(q1): 0.1,
        cirq.ControlledGate(cirq.Y).on(q0, q2): 1
    },
     np.array([
         [1.1, _, _, _, _, _, _, _],
         [_, 1.1, _, _, _, _, _, _],
         [_, _, 1.1, _, _, _, _, _],
         [_, _, _, 1.1, _, _, _, _],
         [_, _, _, _, 0.1, -1j, _, _],
         [_, _, _, _, 1j, 0.1, _, _],
         [_, _, _, _, _, _, 0.1, -1j],
         [_, _, _, _, _, _, 1j, 0.1],
     ])),
    ({
        cirq.I(q2): 0.1,
        cirq.ControlledGate(cirq.Y).on(q0, q1): 1
    },
     np.array([
         [1.1, _, _, _, _, _, _, _],
         [_, 1.1, _, _, _, _, _, _],
         [_, _, 1.1, _, _, _, _, _],
         [_, _, _, 1.1, _, _, _, _],
         [_, _, _, _, 0.1, _, -1j, _],
         [_, _, _, _, _, 0.1, _, -1j],
         [_, _, _, _, 1j, _, 0.1, _],
         [_, _, _, _, _, 1j, _, 0.1],
     ])),
    ({
        cirq.I(q0): 0.1,
        cirq.FREDKIN(q1, q2, q3): 1
    },
     np.array([
         [1.1, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
         [_, 1.1, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
         [_, _, 1.1, _, _, _, _, _, _, _, _, _, _, _, _, _],
         [_, _, _, 1.1, _, _, _, _, _, _, _, _, _, _, _, _],
         [_, _, _, _, 1.1, _, _, _, _, _, _, _, _, _, _, _],
         [_, _, _, _, _, 0.1, 1, _, _, _, _, _, _, _, _, _],
         [_, _, _, _, _, 1, 0.1, _, _, _, _, _, _, _, _, _],
         [_, _, _, _, _, _, _, 1.1, _, _, _, _, _, _, _, _],
         [_, _, _, _, _, _, _, _, 1.1, _, _, _, _, _, _, _],
         [_, _, _, _, _, _, _, _, _, 1.1, _, _, _, _, _, _],
         [_, _, _, _, _, _, _, _, _, _, 1.1, _, _, _, _, _],
         [_, _, _, _, _, _, _, _, _, _, _, 1.1, _, _, _, _],
         [_, _, _, _, _, _, _, _, _, _, _, _, 1.1, _, _, _],
         [_, _, _, _, _, _, _, _, _, _, _, _, _, 0.1, 1, _],
         [_, _, _, _, _, _, _, _, _, _, _, _, _, 1, 0.1, _],
         [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, 1.1],
     ])),
    ({
        cirq.I(q1): 0.1,
        cirq.FREDKIN(q0, q2, q3): 1
    },
     np.array([
         [1.1, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
         [_, 1.1, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
         [_, _, 1.1, _, _, _, _, _, _, _, _, _, _, _, _, _],
         [_, _, _, 1.1, _, _, _, _, _, _, _, _, _, _, _, _],
         [_, _, _, _, 1.1, _, _, _, _, _, _, _, _, _, _, _],
         [_, _, _, _, _, 1.1, _, _, _, _, _, _, _, _, _, _],
         [_, _, _, _, _, _, 1.1, _, _, _, _, _, _, _, _, _],
         [_, _, _, _, _, _, _, 1.1, _, _, _, _, _, _, _, _],
         [_, _, _, _, _, _, _, _, 1.1, _, _, _, _, _, _, _],
         [_, _, _, _, _, _, _, _, _, 0.1, 1, _, _, _, _, _],
         [_, _, _, _, _, _, _, _, _, 1, 0.1, _, _, _, _, _],
         [_, _, _, _, _, _, _, _, _, _, _, 1.1, _, _, _, _],
         [_, _, _, _, _, _, _, _, _, _, _, _, 1.1, _, _, _],
         [_, _, _, _, _, _, _, _, _, _, _, _, _, 0.1, 1, _],
         [_, _, _, _, _, _, _, _, _, _, _, _, _, 1, 0.1, _],
         [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, 1.1],
     ])),
    ({
        cirq.I(q2): 0.1,
        cirq.FREDKIN(q0, q1, q3): 1
    },
     np.array([
         [1.1, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
         [_, 1.1, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
         [_, _, 1.1, _, _, _, _, _, _, _, _, _, _, _, _, _],
         [_, _, _, 1.1, _, _, _, _, _, _, _, _, _, _, _, _],
         [_, _, _, _, 1.1, _, _, _, _, _, _, _, _, _, _, _],
         [_, _, _, _, _, 1.1, _, _, _, _, _, _, _, _, _, _],
         [_, _, _, _, _, _, 1.1, _, _, _, _, _, _, _, _, _],
         [_, _, _, _, _, _, _, 1.1, _, _, _, _, _, _, _, _],
         [_, _, _, _, _, _, _, _, 1.1, _, _, _, _, _, _, _],
         [_, _, _, _, _, _, _, _, _, 0.1, _, _, 1, _, _, _],
         [_, _, _, _, _, _, _, _, _, _, 1.1, _, _, _, _, _],
         [_, _, _, _, _, _, _, _, _, _, _, 0.1, _, _, 1, _],
         [_, _, _, _, _, _, _, _, _, 1, _, _, 0.1, _, _, _],
         [_, _, _, _, _, _, _, _, _, _, _, _, 0, 1.1, _, _],
         [_, _, _, _, _, _, _, _, _, _, _, 1, _, _, 0.1, _],
         [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, 1.1],
     ])),
    ({
        cirq.I(q3): 2j,
        cirq.FREDKIN(q0, q1, q2): 1j
    },
     np.array([
         [3j, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
         [_, 3j, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
         [_, _, 3j, _, _, _, _, _, _, _, _, _, _, _, _, _],
         [_, _, _, 3j, _, _, _, _, _, _, _, _, _, _, _, _],
         [_, _, _, _, 3j, _, _, _, _, _, _, _, _, _, _, _],
         [_, _, _, _, _, 3j, _, _, _, _, _, _, _, _, _, _],
         [_, _, _, _, _, _, 3j, _, _, _, _, _, _, _, _, _],
         [_, _, _, _, _, _, _, 3j, _, _, _, _, _, _, _, _],
         [_, _, _, _, _, _, _, _, 3j, _, _, _, _, _, _, _],
         [_, _, _, _, _, _, _, _, _, 3j, _, _, _, _, _, _],
         [_, _, _, _, _, _, _, _, _, _, 2j, _, 1j, _, _, _],
         [_, _, _, _, _, _, _, _, _, _, _, 2j, _, 1j, _, _],
         [_, _, _, _, _, _, _, _, _, _, 1j, _, 2j, _, _, _],
         [_, _, _, _, _, _, _, _, _, _, _, 1j, _, 2j, _, _],
         [_, _, _, _, _, _, _, _, _, _, _, _, _, _, 3j, _],
         [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, 3j],
     ])),
))
def test_linear_combination_of_operations_has_correct_matrix(
        terms, expected_matrix):
    combination = cirq.LinearCombinationOfOperations(terms)
    assert np.allclose(combination.matrix(), expected_matrix)


@pytest.mark.parametrize('terms, expected_expansion', (
    ({}, {}),
    ({
        cirq.X(q0): -10,
        cirq.Y(q0): 20
    }, {
        'X': -10,
        'Y': 20
    }),
    ({
        cirq.X(q0): -10,
        cirq.Y(q1): 20
    }, {
        'XI': -10,
        'IY': 20
    }),
    ({
        cirq.Y(q0): np.sqrt(0.5),
        cirq.H(q0): 1
    }, {
        'X': np.sqrt(0.5),
        'Y': np.sqrt(0.5),
        'Z': np.sqrt(0.5)
    }),
    ({
        cirq.Y(q0): np.sqrt(0.5),
        cirq.H(q2): 1
    }, {
        'IX': np.sqrt(0.5),
        'YI': np.sqrt(0.5),
        'IZ': np.sqrt(0.5)
    }),
    ({
        cirq.XX(q0, q1): -2,
        cirq.YY(q0, q1): 3j,
        cirq.ZZ(q0, q1): 4
    }, {
        'XX': -2,
        'YY': 3j,
        'ZZ': 4
    }),
    ({
        cirq.XX(q0, q1): -2,
        cirq.YY(q0, q2): 3j,
        cirq.ZZ(q1, q2): 4
    }, {
        'XXI': -2,
        'YIY': 3j,
        'IZZ': 4
    }),
    ({
        cirq.IdentityGate(2).on(q0, q3): -1,
        cirq.CZ(q1, q2): 2
    }, {
        'IIZI': 1,
        'IZII': 1,
        'IZZI': -1
    }),
    ({
        cirq.CNOT(q0, q1): 2,
        cirq.Z(q0): -1,
        cirq.X(q1): -1
    }, {
        'II': 1,
        'ZX': -1
    }),
))
def test_linear_combination_of_operations_has_correct_pauli_expansion(
        terms, expected_expansion):
    combination = cirq.LinearCombinationOfOperations(terms)
    actual_expansion = cirq.pauli_expansion(combination)
    assert set(actual_expansion.keys()) == set(expected_expansion.keys())
    for name in actual_expansion.keys():
        assert abs(actual_expansion[name] - expected_expansion[name]) < 1e-12


@pytest.mark.parametrize('expression, expected_result', (
    (cirq.LinearCombinationOfOperations({cirq.XX(q0, q1): 2}),
     cirq.LinearCombinationOfOperations(
         {cirq.ParallelGateOperation(cirq.X, (q0, q1)): 2})),
    (cirq.LinearCombinationOfOperations({cirq.CNOT(q0, q1): 2}),
     cirq.LinearCombinationOfOperations(
         {
             cirq.IdentityGate(2).on(q0, q1): 1,
             cirq.PauliString({q1: cirq.X}): 1,
             cirq.PauliString({q0: cirq.Z}): 1,
             cirq.PauliString({
                 q0: cirq.Z,
                 q1: cirq.X,
             }): -1
         })),
))
def test_operation_expressions(expression, expected_result):
    assert_linear_combinations_are_equal(expression, expected_result)


def test_pauli_sum_construction():
    q = cirq.LineQubit.range(2)
    pstr1 = cirq.X(q[0]) * cirq.X(q[1])
    pstr2 = cirq.Y(q[0]) * cirq.Y(q[1])
    psum = pstr1 + pstr2
    assert psum  # should be truthy
    assert list(psum) == [pstr1, pstr2]

    psum2 = cirq.PauliSum.from_pauli_strings([pstr1, pstr2])
    assert psum == psum2

    zero = cirq.PauliSum()
    assert len(zero) == 0


def test_pauli_sum_from_single_pauli():
    q = cirq.LineQubit.range(2)
    psum1 = cirq.X(q[0]) + cirq.Y(q[1])
    assert psum1 == cirq.PauliSum.from_pauli_strings(
        [cirq.X(q[0]) * 1, cirq.Y(q[1]) * 1])

    psum2 = cirq.X(q[0]) * cirq.X(q[1]) + cirq.Y(q[1])
    assert psum2 == cirq.PauliSum.from_pauli_strings(
        [cirq.X(q[0]) * cirq.X(q[1]),
         cirq.Y(q[1]) * 1])

    psum3 = cirq.Y(q[1]) + cirq.X(q[0]) * cirq.X(q[1])
    assert psum3 == psum2


def test_pauli_sub():
    q = cirq.LineQubit.range(2)
    pstr1 = cirq.X(q[0]) * cirq.X(q[1])
    pstr2 = cirq.Y(q[0]) * cirq.Y(q[1])
    psum = pstr1 - pstr2

    psum2 = cirq.PauliSum.from_pauli_strings([pstr1, -1 * pstr2])
    assert psum == psum2


def test_pauli_sub_simplify():
    q = cirq.LineQubit.range(2)
    pstr1 = cirq.X(q[0]) * cirq.X(q[1])
    pstr2 = cirq.X(q[0]) * cirq.X(q[1])
    psum = pstr1 - pstr2

    psum2 = cirq.PauliSum.from_pauli_strings([])
    assert psum == psum2


def test_pauli_sum_neg():
    q = cirq.LineQubit.range(2)
    pstr1 = cirq.X(q[0]) * cirq.X(q[1])
    pstr2 = cirq.Y(q[0]) * cirq.Y(q[1])
    psum1 = pstr1 + pstr2
    psum2 = -1 * pstr1 - pstr2

    assert -psum1 == psum2
    psum1 *= -1
    assert psum1 == psum2

    psum2 = psum1 * -1
    assert psum1 == -psum2


def test_paulisum_validation():
    q = cirq.LineQubit.range(2)
    pstr1 = cirq.X(q[0]) * cirq.X(q[1])
    pstr2 = cirq.Y(q[0]) * cirq.Y(q[1])
    with pytest.raises(ValueError) as e:
        cirq.PauliSum([pstr1, pstr2])
    assert e.match("Consider using")

    with pytest.raises(ValueError):
        ld = cirq.LinearDict({pstr1: 2.0})
        cirq.PauliSum(ld)

    with pytest.raises(ValueError):
        key = frozenset([('q0', cirq.X)])
        ld = cirq.LinearDict({key: 2.0})
        cirq.PauliSum(ld)

    with pytest.raises(ValueError):
        key = frozenset([(q[0], cirq.H)])
        ld = cirq.LinearDict({key: 2.0})
        cirq.PauliSum(ld)

    key = frozenset([(q[0], cirq.X)])
    ld = cirq.LinearDict({key: 2.0})
    assert (cirq.PauliSum(ld) == cirq.PauliSum.from_pauli_strings(
        [2 * cirq.X(q[0])]))


def test_add_number_paulisum():
    q = cirq.LineQubit.range(2)
    pstr1 = cirq.X(q[0]) * cirq.X(q[1])
    psum = cirq.PauliSum.from_pauli_strings([pstr1]) + 1.3
    assert psum == cirq.PauliSum.from_pauli_strings(
        [pstr1, cirq.PauliString({}, 1.3)])


def test_add_number_paulistring():
    a, b = cirq.LineQubit.range(2)
    pstr1 = cirq.X(a) * cirq.X(b)
    psum = pstr1 + 1.3
    assert psum == cirq.PauliSum.from_pauli_strings(
        [pstr1, cirq.PauliString({}, 1.3)])
    assert psum == 1.3 + pstr1

    psum = pstr1 - 1.3
    assert psum == psum + 0 == psum - 0 == 0 + psum == -(0 - psum)
    assert psum + 1 == 1 + psum
    assert psum - 1 == -(1 - psum)
    assert psum == cirq.PauliSum.from_pauli_strings(
        [pstr1, cirq.PauliString({}, -1.3)])
    assert psum == -1.3 + pstr1
    assert psum == -1.3 - -pstr1

    assert cirq.X(a) + 2 == 2 + cirq.X(a) == cirq.PauliSum.from_pauli_strings([
        cirq.PauliString() * 2,
        cirq.PauliString({a: cirq.X}),
    ])


def test_pauli_sum_formatting():
    q = cirq.LineQubit.range(2)
    pauli = cirq.X(q[0])
    assert str(pauli) == 'X(0)'
    paulistr = cirq.X(q[0]) * cirq.X(q[1])
    assert str(paulistr) == 'X(0)*X(1)'
    paulisum1 = cirq.X(q[0]) * cirq.X(q[1]) + 4
    assert str(paulisum1) == '1.000*X(0)*X(1)+4.000*I'
    paulisum2 = cirq.X(q[0]) * cirq.X(q[1]) + cirq.Z(q[0])
    assert str(paulisum2) == '1.000*X(0)*X(1)+1.000*Z(0)'
    paulisum3 = cirq.X(q[0]) * cirq.X(q[1]) + cirq.Z(q[0]) * cirq.Z(q[1])
    assert str(paulisum3) == '1.000*X(0)*X(1)+1.000*Z(0)*Z(1)'
    assert "{:.0f}".format(paulisum3) == '1*X(0)*X(1)+1*Z(0)*Z(1)'

    empty = cirq.PauliSum.from_pauli_strings([])
    assert str(empty) == "0.000"


def test_pauli_sum_repr():
    q = cirq.LineQubit.range(2)
    pstr1 = cirq.X(q[0]) * cirq.X(q[1])
    pstr2 = cirq.Y(q[0]) * cirq.Y(q[1])
    psum = pstr1 + 2 * pstr2 + 1
    cirq.testing.assert_equivalent_repr(psum)


def test_bad_arithmetic():
    q = cirq.LineQubit.range(2)
    pstr1 = cirq.X(q[0]) * cirq.X(q[1])
    pstr2 = cirq.Y(q[0]) * cirq.Y(q[1])
    psum = pstr1 + 2 * pstr2 + 1

    with pytest.raises(TypeError):
        psum += 'hi mom'

    with pytest.raises(TypeError):
        _ = psum + 'hi mom'

    with pytest.raises(TypeError):
        psum -= 'hi mom'

    with pytest.raises(TypeError):
        _ = psum - 'hi mom'

    with pytest.raises(TypeError):
        psum *= [1, 2, 3]

    with pytest.raises(TypeError):
        _ = psum * [1, 2, 3]

    with pytest.raises(TypeError):
        _ = [1, 2, 3] * psum

    with pytest.raises(TypeError):
        _ = psum / [1, 2, 3]
