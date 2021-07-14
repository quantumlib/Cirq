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
import sympy
import sympy.parsing.sympy_parser as sympy_parser

import cirq
import cirq.testing

_ = 0.0  # Make matrices readable by visually hiding off-diagonal elements.
q0, q1, q2, q3 = cirq.LineQubit.range(4)


@pytest.mark.parametrize(
    'terms',
    (
        {cirq.X: -2, cirq.H: 2},
        {cirq.XX: 1, cirq.YY: 1j, cirq.ZZ: -1},
        {cirq.TOFFOLI: 0.5j, cirq.FREDKIN: 0.5},
    ),
)
def test_linear_combination_of_gates_accepts_consistent_gates(terms):
    combination_1 = cirq.LinearCombinationOfGates(terms)

    combination_2 = cirq.LinearCombinationOfGates({})
    combination_2.update(terms)

    combination_3 = cirq.LinearCombinationOfGates({})
    for gate, coefficient in terms.items():
        combination_3[gate] += coefficient

    assert combination_1 == combination_2 == combination_3


@pytest.mark.parametrize(
    'terms',
    (
        {cirq.X: -2, cirq.CZ: 2},
        {cirq.X: 1, cirq.YY: 1j, cirq.ZZ: -1},
        {cirq.TOFFOLI: 0.5j, cirq.S: 0.5},
    ),
)
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


@pytest.mark.parametrize(
    'gate',
    (
        cirq.X,
        cirq.Y,
        cirq.XX,
        cirq.CZ,
        cirq.CSWAP,
        cirq.FREDKIN,
    ),
)
def test_empty_linear_combination_of_gates_accepts_all_gates(gate):
    combination = cirq.LinearCombinationOfGates({})
    combination[gate] = -0.5j
    assert len(combination) == 1


@pytest.mark.parametrize(
    'terms, expected_num_qubits',
    (
        ({cirq.X: 1}, 1),
        ({cirq.H: 10, cirq.S: -10j}, 1),
        ({cirq.XX: 1, cirq.YY: 2, cirq.ZZ: 3}, 2),
        ({cirq.CCZ: 0.1, cirq.CSWAP: 0.2}, 3),
    ),
)
def test_linear_combination_of_gates_has_correct_num_qubits(terms, expected_num_qubits):
    combination = cirq.LinearCombinationOfGates(terms)
    assert combination.num_qubits() == expected_num_qubits


def test_empty_linear_combination_of_gates_has_no_matrix():
    empty = cirq.LinearCombinationOfGates({})
    assert empty.num_qubits() is None
    with pytest.raises(ValueError):
        empty.matrix()


@pytest.mark.parametrize(
    'terms, expected_matrix',
    (
        (
            {cirq.I: 2, cirq.X: 3, cirq.Y: 4, cirq.Z: 5j},
            np.array([[2 + 5j, 3 - 4j], [3 + 4j, 2 - 5j]]),
        ),
        ({cirq.XX: 0.5, cirq.YY: -0.5}, np.rot90(np.diag([1, 0, 0, 1]))),
        ({cirq.CCZ: 3j}, np.diag([3j, 3j, 3j, 3j, 3j, 3j, 3j, -3j])),
    ),
)
def test_linear_combination_of_gates_has_correct_matrix(terms, expected_matrix):
    combination = cirq.LinearCombinationOfGates(terms)
    assert np.all(combination.matrix() == expected_matrix)


@pytest.mark.parametrize(
    'terms, expected_unitary',
    (
        (
            {cirq.X: np.sqrt(0.5), cirq.Y: np.sqrt(0.5)},
            np.array([[0, np.sqrt(-1j)], [np.sqrt(1j), 0]]),
        ),
        (
            {
                cirq.IdentityGate(2): np.sqrt(0.5),
                cirq.YY: -1j * np.sqrt(0.5),
            },
            np.sqrt(0.5) * np.array([[1, 0, 0, 1j], [0, 1, -1j, 0], [0, -1j, 1, 0], [1j, 0, 0, 1]]),
        ),
    ),
)
def test_unitary_linear_combination_of_gates_has_correct_unitary(terms, expected_unitary):
    combination = cirq.LinearCombinationOfGates(terms)
    assert cirq.has_unitary(combination)
    assert np.allclose(cirq.unitary(combination), expected_unitary)


@pytest.mark.parametrize(
    'terms', ({cirq.X: 2}, {cirq.Y ** sympy.Symbol('t'): 1}, {cirq.X: 1, cirq.S: 1})
)
def test_non_unitary_linear_combination_of_gates_has_no_unitary(terms):
    combination = cirq.LinearCombinationOfGates(terms)
    assert not cirq.has_unitary(combination)
    with pytest.raises((TypeError, ValueError)):
        _ = cirq.unitary(combination)


@pytest.mark.parametrize(
    'terms, expected_expansion',
    (
        ({cirq.X: 10, cirq.Y: -20}, {'X': 10, 'Y': -20}),
        (
            {cirq.Y: np.sqrt(0.5), cirq.H: 1},
            {'X': np.sqrt(0.5), 'Y': np.sqrt(0.5), 'Z': np.sqrt(0.5)},
        ),
        ({cirq.X: 2, cirq.H: 1}, {'X': 2 + np.sqrt(0.5), 'Z': np.sqrt(0.5)}),
        ({cirq.XX: -2, cirq.YY: 3j, cirq.ZZ: 4}, {'XX': -2, 'YY': 3j, 'ZZ': 4}),
    ),
)
def test_linear_combination_of_gates_has_correct_pauli_expansion(terms, expected_expansion):
    combination = cirq.LinearCombinationOfGates(terms)
    actual_expansion = cirq.pauli_expansion(combination)
    assert set(actual_expansion.keys()) == set(expected_expansion.keys())
    for name in actual_expansion.keys():
        assert abs(actual_expansion[name] - expected_expansion[name]) < 1e-12


@pytest.mark.parametrize(
    'terms, exponent, expected_terms',
    (
        (
            {
                cirq.X: 1,
            },
            2,
            {
                cirq.I: 1,
            },
        ),
        (
            {
                cirq.X: 1,
            },
            3,
            {
                cirq.X: 1,
            },
        ),
        (
            {
                cirq.Y: 0.5,
            },
            10,
            {
                cirq.I: 2 ** -10,
            },
        ),
        (
            {
                cirq.Y: 0.5,
            },
            11,
            {
                cirq.Y: 2 ** -11,
            },
        ),
        (
            {
                cirq.I: 1,
                cirq.X: 2,
                cirq.Y: 3,
                cirq.Z: 4,
            },
            2,
            {
                cirq.I: 30,
                cirq.X: 4,
                cirq.Y: 6,
                cirq.Z: 8,
            },
        ),
        (
            {
                cirq.X: 1,
                cirq.Y: 1j,
            },
            2,
            {},
        ),
        (
            {
                cirq.X: 0.4,
                cirq.Y: 0.4,
            },
            0,
            {
                cirq.I: 1,
            },
        ),
    ),
)
def test_linear_combinations_of_gates_valid_powers(terms, exponent, expected_terms):
    combination = cirq.LinearCombinationOfGates(terms)
    actual_result = combination ** exponent
    expected_result = cirq.LinearCombinationOfGates(expected_terms)
    assert cirq.approx_eq(actual_result, expected_result)
    assert len(actual_result) == len(expected_terms)


@pytest.mark.parametrize(
    'terms, exponent',
    (
        ({}, 2),
        (
            {
                cirq.H: 1,
            },
            2,
        ),
        (
            {
                cirq.CNOT: 2,
            },
            2,
        ),
        (
            {
                cirq.X: 1,
                cirq.S: -1,
            },
            2,
        ),
        (
            {
                cirq.X: 1,
            },
            -1,
        ),
        (
            {
                cirq.Y: 1,
            },
            sympy.Symbol('k'),
        ),
    ),
)
def test_linear_combinations_of_gates_invalid_powers(terms, exponent):
    combination = cirq.LinearCombinationOfGates(terms)
    with pytest.raises(TypeError):
        _ = combination ** exponent


@pytest.mark.parametrize(
    'terms, is_parameterized, parameter_names',
    [
        ({cirq.H: 1}, False, set()),
        ({cirq.X ** sympy.Symbol('t'): 1}, True, {'t'}),
    ],
)
@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_parameterized_linear_combination_of_gates(
    terms, is_parameterized, parameter_names, resolve_fn
):
    gate = cirq.LinearCombinationOfGates(terms)
    assert cirq.is_parameterized(gate) == is_parameterized
    assert cirq.parameter_names(gate) == parameter_names
    resolved = resolve_fn(gate, {p: 1 for p in parameter_names})
    assert not cirq.is_parameterized(resolved)


def get_matrix(
    operator: Union[
        cirq.Gate,
        cirq.GateOperation,
        cirq.LinearCombinationOfGates,
        cirq.LinearCombinationOfOperations,
    ]
) -> np.ndarray:
    if isinstance(operator, (cirq.LinearCombinationOfGates, cirq.LinearCombinationOfOperations)):
        return operator.matrix()
    return cirq.unitary(operator)


def assert_linear_combinations_are_equal(
    actual: Union[cirq.LinearCombinationOfGates, cirq.LinearCombinationOfOperations],
    expected: Union[cirq.LinearCombinationOfGates, cirq.LinearCombinationOfOperations],
) -> None:
    if not actual and not expected:
        assert len(actual) == 0
        assert len(expected) == 0
        return

    actual_matrix = get_matrix(actual)
    expected_matrix = get_matrix(expected)
    assert np.allclose(actual_matrix, expected_matrix)

    actual_expansion = cirq.pauli_expansion(actual)
    expected_expansion = cirq.pauli_expansion(expected)
    assert set(actual_expansion.keys()) == set(expected_expansion.keys())
    for name in actual_expansion.keys():
        assert abs(actual_expansion[name] - expected_expansion[name]) < 1e-12


@pytest.mark.parametrize(
    'expression, expected_result',
    (
        ((cirq.X + cirq.Z) / np.sqrt(2), cirq.H),
        (cirq.X - cirq.Y, -cirq.Y + cirq.X),
        (cirq.X + cirq.S - cirq.X, cirq.S),
        (cirq.Y - 2 * cirq.Y, -cirq.Y),
        (cirq.rx(0.2), np.cos(0.1) * cirq.I - 1j * np.sin(0.1) * cirq.X),
        (1j * cirq.H * 1j, -cirq.H),
        (-1j * cirq.Y, cirq.ry(np.pi)),
        (np.sqrt(-1j) * cirq.S, cirq.rz(np.pi / 2)),
        (0.5 * (cirq.IdentityGate(2) + cirq.XX + cirq.YY + cirq.ZZ), cirq.SWAP),
        ((cirq.IdentityGate(2) + 1j * (cirq.XX + cirq.YY) + cirq.ZZ) / 2, cirq.ISWAP),
        (cirq.CNOT + 0 * cirq.SWAP, cirq.CNOT),
        (0.5 * cirq.FREDKIN, cirq.FREDKIN / 2),
        (cirq.FREDKIN * 0.5, cirq.FREDKIN / 2),
        (((cirq.X + cirq.Y) / np.sqrt(2)) ** 2, cirq.I),
        ((cirq.X + cirq.Z) ** 3, 2 * (cirq.X + cirq.Z)),
        ((cirq.X + 1j * cirq.Y) ** 2, cirq.LinearCombinationOfGates({})),
        ((cirq.X - 1j * cirq.Y) ** 2, cirq.LinearCombinationOfGates({})),
        (((3 * cirq.X - 4 * cirq.Y + 12 * cirq.Z) / 13) ** 24, cirq.I),
        (
            ((3 * cirq.X - 4 * cirq.Y + 12 * cirq.Z) / 13) ** 25,
            (3 * cirq.X - 4 * cirq.Y + 12 * cirq.Z) / 13,
        ),
        ((cirq.X + cirq.Y + cirq.Z) ** 0, cirq.I),
        ((cirq.X - 1j * cirq.Y) ** 0, cirq.I),
    ),
)
def test_gate_expressions(expression, expected_result):
    assert_linear_combinations_are_equal(expression, expected_result)


@pytest.mark.parametrize(
    'gates',
    (
        (cirq.X, cirq.T, cirq.T, cirq.X, cirq.Z),
        (cirq.CZ, cirq.XX, cirq.YY, cirq.ZZ),
        (cirq.TOFFOLI, cirq.TOFFOLI, cirq.FREDKIN),
    ),
)
def test_in_place_manipulations_of_linear_combination_of_gates(gates):
    a = cirq.LinearCombinationOfGates({})
    b = cirq.LinearCombinationOfGates({})

    for i, gate in enumerate(gates):
        a += gate
        b -= gate

        prefix = gates[: i + 1]
        expected_a = cirq.LinearCombinationOfGates(collections.Counter(prefix))
        expected_b = -expected_a

        assert_linear_combinations_are_equal(a, expected_a)
        assert_linear_combinations_are_equal(b, expected_b)


@pytest.mark.parametrize(
    'op',
    (
        cirq.X(q0),
        cirq.Y(q1),
        cirq.XX(q0, q1),
        cirq.CZ(q0, q1),
        cirq.FREDKIN(q0, q1, q2),
        cirq.ControlledOperation((q0, q1), cirq.H(q2)),
        cirq.ParallelGateOperation(cirq.X, (q0, q1, q2)),
        cirq.PauliString({q0: cirq.X, q1: cirq.Y, q2: cirq.Z}),
    ),
)
def test_empty_linear_combination_of_operations_accepts_all_operations(op):
    combination = cirq.LinearCombinationOfOperations({})
    combination[op] = -0.5j
    assert len(combination) == 1


@pytest.mark.parametrize(
    'terms',
    (
        {cirq.X(q0): -2, cirq.H(q0): 2},
        {cirq.X(q0): -2, cirq.H(q1): 2j},
        {cirq.X(q0): 1, cirq.CZ(q0, q1): 3},
        {cirq.X(q0): 1 + 1j, cirq.CZ(q1, q2): 0.5},
    ),
)
def test_linear_combination_of_operations_is_consistent(terms):
    combination_1 = cirq.LinearCombinationOfOperations(terms)

    combination_2 = cirq.LinearCombinationOfOperations({})
    combination_2.update(terms)

    combination_3 = cirq.LinearCombinationOfOperations({})
    for gate, coefficient in terms.items():
        combination_3[gate] += coefficient

    assert combination_1 == combination_2 == combination_3


@pytest.mark.parametrize(
    'terms, expected_qubits',
    (
        ({}, ()),
        ({cirq.I(q0): 1, cirq.H(q0): 1e-3j}, (q0,)),
        ({cirq.X(q0): 1j, cirq.H(q1): 2j}, (q0, q1)),
        ({cirq.Y(q0): -1, cirq.CZ(q0, q1): 3e3}, (q0, q1)),
        ({cirq.Z(q0): -1j, cirq.CNOT(q1, q2): 0.25}, (q0, q1, q2)),
    ),
)
def test_linear_combination_of_operations_has_correct_qubits(terms, expected_qubits):
    combination = cirq.LinearCombinationOfOperations(terms)
    assert combination.qubits == expected_qubits


@pytest.mark.parametrize(
    'terms, expected_matrix',
    (
        ({}, np.array([0])),
        (
            {cirq.I(q0): 2, cirq.X(q0): 3, cirq.Y(q0): 4, cirq.Z(q0): 5j},
            np.array(
                [
                    [2 + 5j, 3 - 4j],
                    [3 + 4j, 2 - 5j],
                ]
            ),
        ),
        (
            {cirq.X(q0): 2, cirq.Y(q1): 3},
            np.array(
                [
                    [0, -3j, 2, 0],
                    [3j, 0, 0, 2],
                    [2, 0, 0, -3j],
                    [0, 2, 3j, 0],
                ]
            ),
        ),
        ({cirq.XX(q0, q1): 0.5, cirq.YY(q0, q1): -0.5}, np.rot90(np.diag([1, 0, 0, 1]))),
        ({cirq.CCZ(q0, q1, q2): 3j}, np.diag([3j, 3j, 3j, 3j, 3j, 3j, 3j, -3j])),
        (
            {cirq.I(q0): 0.1, cirq.CNOT(q1, q2): 1},
            np.array(
                [
                    [1.1, _, _, _, _, _, _, _],
                    [_, 1.1, _, _, _, _, _, _],
                    [_, _, 0.1, 1, _, _, _, _],
                    [_, _, 1, 0.1, _, _, _, _],
                    [_, _, _, _, 1.1, _, _, _],
                    [_, _, _, _, _, 1.1, _, _],
                    [_, _, _, _, _, _, 0.1, 1],
                    [_, _, _, _, _, _, 1, 0.1],
                ]
            ),
        ),
        (
            {cirq.I(q1): 0.1, cirq.CNOT(q0, q2): 1},
            np.array(
                [
                    [1.1, _, _, _, _, _, _, _],
                    [_, 1.1, _, _, _, _, _, _],
                    [_, _, 1.1, _, _, _, _, _],
                    [_, _, _, 1.1, _, _, _, _],
                    [_, _, _, _, 0.1, 1, _, _],
                    [_, _, _, _, 1, 0.1, _, _],
                    [_, _, _, _, _, _, 0.1, 1],
                    [_, _, _, _, _, _, 1, 0.1],
                ]
            ),
        ),
        (
            {cirq.I(q2): 0.1, cirq.CNOT(q0, q1): 1},
            np.array(
                [
                    [1.1, _, _, _, _, _, _, _],
                    [_, 1.1, _, _, _, _, _, _],
                    [_, _, 1.1, _, _, _, _, _],
                    [_, _, _, 1.1, _, _, _, _],
                    [_, _, _, _, 0.1, _, 1, _],
                    [_, _, _, _, _, 0.1, _, 1],
                    [_, _, _, _, 1, _, 0.1, _],
                    [_, _, _, _, _, 1, _, 0.1],
                ]
            ),
        ),
        (
            {cirq.I(q0): 0.1, cirq.ControlledGate(cirq.Y).on(q1, q2): 1},
            np.array(
                [
                    [1.1, _, _, _, _, _, _, _],
                    [_, 1.1, _, _, _, _, _, _],
                    [_, _, 0.1, -1j, _, _, _, _],
                    [_, _, 1j, 0.1, _, _, _, _],
                    [_, _, _, _, 1.1, _, _, _],
                    [_, _, _, _, _, 1.1, _, _],
                    [_, _, _, _, _, _, 0.1, -1j],
                    [_, _, _, _, _, _, 1j, 0.1],
                ]
            ),
        ),
        (
            {cirq.I(q1): 0.1, cirq.ControlledGate(cirq.Y).on(q0, q2): 1},
            np.array(
                [
                    [1.1, _, _, _, _, _, _, _],
                    [_, 1.1, _, _, _, _, _, _],
                    [_, _, 1.1, _, _, _, _, _],
                    [_, _, _, 1.1, _, _, _, _],
                    [_, _, _, _, 0.1, -1j, _, _],
                    [_, _, _, _, 1j, 0.1, _, _],
                    [_, _, _, _, _, _, 0.1, -1j],
                    [_, _, _, _, _, _, 1j, 0.1],
                ]
            ),
        ),
        (
            {cirq.I(q2): 0.1, cirq.ControlledGate(cirq.Y).on(q0, q1): 1},
            np.array(
                [
                    [1.1, _, _, _, _, _, _, _],
                    [_, 1.1, _, _, _, _, _, _],
                    [_, _, 1.1, _, _, _, _, _],
                    [_, _, _, 1.1, _, _, _, _],
                    [_, _, _, _, 0.1, _, -1j, _],
                    [_, _, _, _, _, 0.1, _, -1j],
                    [_, _, _, _, 1j, _, 0.1, _],
                    [_, _, _, _, _, 1j, _, 0.1],
                ]
            ),
        ),
        (
            {cirq.I(q0): 0.1, cirq.FREDKIN(q1, q2, q3): 1},
            np.array(
                [
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
                ]
            ),
        ),
        (
            {cirq.I(q1): 0.1, cirq.FREDKIN(q0, q2, q3): 1},
            np.array(
                [
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
                ]
            ),
        ),
        (
            {cirq.I(q2): 0.1, cirq.FREDKIN(q0, q1, q3): 1},
            np.array(
                [
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
                ]
            ),
        ),
        (
            {cirq.I(q3): 2j, cirq.FREDKIN(q0, q1, q2): 1j},
            np.array(
                [
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
                ]
            ),
        ),
    ),
)
def test_linear_combination_of_operations_has_correct_matrix(terms, expected_matrix):
    combination = cirq.LinearCombinationOfOperations(terms)
    assert np.allclose(combination.matrix(), expected_matrix)


@pytest.mark.parametrize(
    'terms, expected_unitary',
    (
        (
            {cirq.X(q0): np.sqrt(0.5), cirq.Z(q0): np.sqrt(0.5)},
            np.sqrt(0.5) * np.array([[1, 1], [1, -1]]),
        ),
        (
            {
                cirq.IdentityGate(3).on(q0, q1, q2): np.sqrt(0.5),
                cirq.CCZ(q0, q1, q2): 1j * np.sqrt(0.5),
            },
            np.diag(
                [
                    np.sqrt(1j),
                    np.sqrt(1j),
                    np.sqrt(1j),
                    np.sqrt(1j),
                    np.sqrt(1j),
                    np.sqrt(1j),
                    np.sqrt(1j),
                    np.sqrt(-1j),
                ]
            ),
        ),
    ),
)
def test_unitary_linear_combination_of_operations_has_correct_unitary(terms, expected_unitary):
    combination = cirq.LinearCombinationOfOperations(terms)
    assert cirq.has_unitary(combination)
    assert np.allclose(cirq.unitary(combination), expected_unitary)


@pytest.mark.parametrize(
    'terms',
    (
        {cirq.CNOT(q0, q1): 1.1},
        {cirq.CZ(q0, q1) ** sympy.Symbol('t'): 1},
        {cirq.X(q0): 1, cirq.S(q0): 1},
    ),
)
def test_non_unitary_linear_combination_of_operations_has_no_unitary(terms):
    combination = cirq.LinearCombinationOfOperations(terms)
    assert not cirq.has_unitary(combination)
    with pytest.raises((TypeError, ValueError)):
        _ = cirq.unitary(combination)


@pytest.mark.parametrize(
    'terms, expected_expansion',
    (
        ({}, {}),
        ({cirq.X(q0): -10, cirq.Y(q0): 20}, {'X': -10, 'Y': 20}),
        ({cirq.X(q0): -10, cirq.Y(q1): 20}, {'XI': -10, 'IY': 20}),
        (
            {cirq.Y(q0): np.sqrt(0.5), cirq.H(q0): 1},
            {'X': np.sqrt(0.5), 'Y': np.sqrt(0.5), 'Z': np.sqrt(0.5)},
        ),
        (
            {cirq.Y(q0): np.sqrt(0.5), cirq.H(q2): 1},
            {'IX': np.sqrt(0.5), 'YI': np.sqrt(0.5), 'IZ': np.sqrt(0.5)},
        ),
        (
            {cirq.XX(q0, q1): -2, cirq.YY(q0, q1): 3j, cirq.ZZ(q0, q1): 4},
            {'XX': -2, 'YY': 3j, 'ZZ': 4},
        ),
        (
            {cirq.XX(q0, q1): -2, cirq.YY(q0, q2): 3j, cirq.ZZ(q1, q2): 4},
            {'XXI': -2, 'YIY': 3j, 'IZZ': 4},
        ),
        (
            {cirq.IdentityGate(2).on(q0, q3): -1, cirq.CZ(q1, q2): 2},
            {'IIZI': 1, 'IZII': 1, 'IZZI': -1},
        ),
        ({cirq.CNOT(q0, q1): 2, cirq.Z(q0): -1, cirq.X(q1): -1}, {'II': 1, 'ZX': -1}),
    ),
)
def test_linear_combination_of_operations_has_correct_pauli_expansion(terms, expected_expansion):
    combination = cirq.LinearCombinationOfOperations(terms)
    actual_expansion = cirq.pauli_expansion(combination)
    assert set(actual_expansion.keys()) == set(expected_expansion.keys())
    for name in actual_expansion.keys():
        assert abs(actual_expansion[name] - expected_expansion[name]) < 1e-12


@pytest.mark.parametrize(
    'terms, exponent, expected_terms',
    (
        (
            {
                cirq.X(q0): 1,
            },
            2,
            {
                cirq.I(q0): 1,
            },
        ),
        (
            {
                cirq.X(q0): 1,
            },
            3,
            {
                cirq.X(q0): 1,
            },
        ),
        (
            {
                cirq.Y(q0): 0.5,
            },
            10,
            {
                cirq.I(q0): 2 ** -10,
            },
        ),
        (
            {
                cirq.Y(q0): 0.5,
            },
            11,
            {
                cirq.Y(q0): 2 ** -11,
            },
        ),
        (
            {
                cirq.I(q0): 1,
                cirq.X(q0): 2,
                cirq.Y(q0): 3,
                cirq.Z(q0): 4,
            },
            2,
            {
                cirq.I(q0): 30,
                cirq.X(q0): 4,
                cirq.Y(q0): 6,
                cirq.Z(q0): 8,
            },
        ),
        (
            {
                cirq.X(q0): 1,
                cirq.Y(q0): 1j,
            },
            2,
            {},
        ),
        (
            {
                cirq.Y(q1): 2,
                cirq.Z(q1): 3,
            },
            0,
            {
                cirq.I(q1): 1,
            },
        ),
    ),
)
def test_linear_combinations_of_operations_valid_powers(terms, exponent, expected_terms):
    combination = cirq.LinearCombinationOfOperations(terms)
    actual_result = combination ** exponent
    expected_result = cirq.LinearCombinationOfOperations(expected_terms)
    assert cirq.approx_eq(actual_result, expected_result)
    assert len(actual_result) == len(expected_terms)


@pytest.mark.parametrize(
    'terms, exponent',
    (
        ({}, 2),
        (
            {
                cirq.H(q0): 1,
            },
            2,
        ),
        (
            {
                cirq.CNOT(q0, q1): 2,
            },
            2,
        ),
        (
            {
                cirq.X(q0): 1,
                cirq.S(q0): -1,
            },
            2,
        ),
        (
            {
                cirq.X(q0): 1,
                cirq.Y(q1): 1,
            },
            2,
        ),
        (
            {
                cirq.Z(q0): 1,
            },
            -1,
        ),
        (
            {
                cirq.X(q0): 1,
            },
            sympy.Symbol('k'),
        ),
    ),
)
def test_linear_combinations_of_operations_invalid_powers(terms, exponent):
    combination = cirq.LinearCombinationOfOperations(terms)
    with pytest.raises(TypeError):
        _ = combination ** exponent


@pytest.mark.parametrize(
    'terms, is_parameterized, parameter_names',
    [
        ({cirq.H(cirq.LineQubit(0)): 1}, False, set()),
        ({cirq.X(cirq.LineQubit(0)) ** sympy.Symbol('t'): 1}, True, {'t'}),
    ],
)
@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_parameterized_linear_combination_of_ops(
    terms, is_parameterized, parameter_names, resolve_fn
):
    op = cirq.LinearCombinationOfOperations(terms)
    assert cirq.is_parameterized(op) == is_parameterized
    assert cirq.parameter_names(op) == parameter_names
    resolved = resolve_fn(op, {p: 1 for p in parameter_names})
    assert not cirq.is_parameterized(resolved)


@pytest.mark.parametrize(
    'expression, expected_result',
    (
        (
            cirq.LinearCombinationOfOperations({cirq.XX(q0, q1): 2}),
            cirq.LinearCombinationOfOperations({cirq.ParallelGateOperation(cirq.X, (q0, q1)): 2}),
        ),
        (
            cirq.LinearCombinationOfOperations({cirq.CNOT(q0, q1): 2}),
            cirq.LinearCombinationOfOperations(
                {
                    cirq.IdentityGate(2).on(q0, q1): 1,
                    cirq.PauliString({q1: cirq.X}): 1,
                    cirq.PauliString({q0: cirq.Z}): 1,
                    cirq.PauliString(
                        {
                            q0: cirq.Z,
                            q1: cirq.X,
                        }
                    ): -1,
                }
            ),
        ),
        (
            cirq.LinearCombinationOfOperations({cirq.X(q0): 1}) ** 2,
            cirq.LinearCombinationOfOperations({cirq.I(q0): 1}),
        ),
        (
            cirq.LinearCombinationOfOperations({cirq.X(q0): np.sqrt(0.5), cirq.Y(q0): np.sqrt(0.5)})
            ** 2,
            cirq.LinearCombinationOfOperations({cirq.I(q0): 1}),
        ),
        (
            cirq.LinearCombinationOfOperations({cirq.X(q0): 1, cirq.Z(q0): 1}) ** 3,
            cirq.LinearCombinationOfOperations({cirq.X(q0): 2, cirq.Z(q0): 2}),
        ),
        (
            cirq.LinearCombinationOfOperations({cirq.X(q0): 1j, cirq.Y(q0): 1}) ** 2,
            cirq.LinearCombinationOfOperations({}),
        ),
        (
            cirq.LinearCombinationOfOperations({cirq.X(q0): -1j, cirq.Y(q0): 1}) ** 2,
            cirq.LinearCombinationOfOperations({}),
        ),
        (
            cirq.LinearCombinationOfOperations(
                {cirq.X(q0): 3 / 13, cirq.Y(q0): -4 / 13, cirq.Z(q0): 12 / 13}
            )
            ** 24,
            cirq.LinearCombinationOfOperations({cirq.I(q0): 1}),
        ),
        (
            cirq.LinearCombinationOfOperations(
                {cirq.X(q0): 3 / 13, cirq.Y(q0): -4 / 13, cirq.Z(q0): 12 / 13}
            )
            ** 25,
            cirq.LinearCombinationOfOperations(
                {cirq.X(q0): 3 / 13, cirq.Y(q0): -4 / 13, cirq.Z(q0): 12 / 13}
            ),
        ),
        (
            cirq.LinearCombinationOfOperations({cirq.X(q1): 2, cirq.Z(q1): 3}) ** 0,
            cirq.LinearCombinationOfOperations({cirq.I(q1): 1}),
        ),
    ),
)
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


@pytest.mark.parametrize(
    'psum, expected_unitary',
    (
        (np.sqrt(0.5) * (cirq.X(q0) + cirq.Z(q0)), np.sqrt(0.5) * np.array([[1, 1], [1, -1]])),
        (
            np.sqrt(0.5) * (cirq.X(q0) * cirq.X(q1) + cirq.Z(q1)),
            np.sqrt(0.5) * np.array([[1, 0, 0, 1], [0, -1, 1, 0], [0, 1, 1, 0], [1, 0, 0, -1]]),
        ),
    ),
)
def test_unitary_pauli_sum_has_correct_unitary(psum, expected_unitary):
    assert cirq.has_unitary(psum)
    assert np.allclose(cirq.unitary(psum), expected_unitary)


@pytest.mark.parametrize(
    'psum',
    (
        cirq.X(q0) + cirq.Z(q0),
        2 * cirq.Z(q0) * cirq.X(q1) + cirq.Y(q2),
        cirq.X(q0) * cirq.Z(q1) - cirq.Z(q1) * cirq.X(q0),
    ),
)
def test_non_pauli_sum_has_no_unitary(psum):
    assert isinstance(psum, cirq.PauliSum)
    assert not cirq.has_unitary(psum)
    with pytest.raises(ValueError):
        _ = cirq.unitary(psum)


@pytest.mark.parametrize(
    'psum, expected_qubits',
    (
        (cirq.Z(q1), (q1,)),
        (cirq.X(q0) + cirq.Y(q0), (q0,)),
        (cirq.X(q0) + cirq.Y(q2), (q0, q2)),
        (cirq.X(q2) + cirq.Y(q0), (q0, q2)),
        (cirq.X(q0) * cirq.Y(q1) + cirq.Y(q1) * cirq.Z(q3), (q0, q1, q3)),
    ),
)
def test_pauli_sum_qubits(psum, expected_qubits):
    assert psum.qubits == expected_qubits


@pytest.mark.parametrize(
    'psum, expected_psum',
    (
        (cirq.Z(q0) + cirq.Y(q0), cirq.Z(q1) + cirq.Y(q0)),
        (2 * cirq.X(q0) + 3 * cirq.Y(q2), 2 * cirq.X(q1) + 3 * cirq.Y(q3)),
        (
            cirq.X(q0) * cirq.Y(q1) + cirq.Y(q1) * cirq.Z(q3),
            cirq.X(q1) * cirq.Y(q2) + cirq.Y(q2) * cirq.Z(q3),
        ),
    ),
)
def test_pauli_sum_with_qubits(psum, expected_psum):
    if len(expected_psum.qubits) == len(psum.qubits):
        assert psum.with_qubits(*expected_psum.qubits) == expected_psum
    else:
        with pytest.raises(ValueError, match='number'):
            psum.with_qubits(*expected_psum.qubits)


def test_pauli_sum_from_single_pauli():
    q = cirq.LineQubit.range(2)
    psum1 = cirq.X(q[0]) + cirq.Y(q[1])
    assert psum1 == cirq.PauliSum.from_pauli_strings([cirq.X(q[0]) * 1, cirq.Y(q[1]) * 1])

    psum2 = cirq.X(q[0]) * cirq.X(q[1]) + cirq.Y(q[1])
    assert psum2 == cirq.PauliSum.from_pauli_strings(
        [cirq.X(q[0]) * cirq.X(q[1]), cirq.Y(q[1]) * 1]
    )

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
    assert cirq.PauliSum(ld) == cirq.PauliSum.from_pauli_strings([2 * cirq.X(q[0])])

    ps = cirq.PauliSum()
    ps += cirq.I(cirq.LineQubit(0))
    assert ps == cirq.PauliSum(cirq.LinearDict({frozenset(): complex(1)}))


def test_add_number_paulisum():
    q = cirq.LineQubit.range(2)
    pstr1 = cirq.X(q[0]) * cirq.X(q[1])
    psum = cirq.PauliSum.from_pauli_strings([pstr1]) + 1.3
    assert psum == cirq.PauliSum.from_pauli_strings([pstr1, cirq.PauliString({}, 1.3)])


def test_add_number_paulistring():
    a, b = cirq.LineQubit.range(2)
    pstr1 = cirq.X(a) * cirq.X(b)
    psum = pstr1 + 1.3
    assert psum == cirq.PauliSum.from_pauli_strings([pstr1, cirq.PauliString({}, 1.3)])
    assert psum == 1.3 + pstr1

    psum = pstr1 - 1.3
    assert psum == psum + 0 == psum - 0 == 0 + psum == -(0 - psum)
    assert psum + 1 == 1 + psum
    assert psum - 1 == -(1 - psum)
    assert psum == cirq.PauliSum.from_pauli_strings([pstr1, cirq.PauliString({}, -1.3)])
    assert psum == -1.3 + pstr1
    assert psum == -1.3 - -pstr1

    assert (
        cirq.X(a) + 2
        == 2 + cirq.X(a)
        == cirq.PauliSum.from_pauli_strings(
            [
                cirq.PauliString() * 2,
                cirq.PauliString({a: cirq.X}),
            ]
        )
    )


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
    assert f"{paulisum3:.0f}" == '1*X(0)*X(1)+1*Z(0)*Z(1)'

    empty = cirq.PauliSum.from_pauli_strings([])
    assert str(empty) == "0.000"


def test_pauli_sum_matrix():
    q = cirq.LineQubit.range(3)
    paulisum = cirq.X(q[0]) * cirq.X(q[1]) + cirq.Z(q[0])
    H1 = np.array(
        [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0], [0.0, 1.0, -1.0, 0.0], [1.0, 0.0, 0.0, -1.0]]
    )
    assert np.allclose(H1, paulisum.matrix())
    assert np.allclose(H1, paulisum.matrix([q[0], q[1]]))
    # Expects a different matrix when change qubits order.
    H2 = np.array(
        [[1.0, 0.0, 0.0, 1.0], [0.0, -1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, -1.0]]
    )
    assert np.allclose(H2, paulisum.matrix([q[1], q[0]]))
    # Expects matrix with a different size when add a new qubit.
    H3 = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        ]
    )
    assert np.allclose(H3, paulisum.matrix([q[1], q[2], q[0]]))


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

    with pytest.raises(TypeError):
        _ = psum ** 1.2

    with pytest.raises(TypeError):
        _ = psum ** -2

    with pytest.raises(TypeError):
        _ = psum ** "string"


def test_paulisum_mul_paulistring():
    q0, q1 = cirq.LineQubit.range(2)

    psum1 = cirq.X(q0) + 2 * cirq.Y(q0) + 3 * cirq.Z(q0)
    x0 = cirq.PauliString(cirq.X(q0))
    y1 = cirq.PauliString(cirq.Y(q1))
    assert x0 * psum1 == cirq.PauliString(cirq.I(q0)) + 2j * cirq.PauliString(
        cirq.Z(q0)
    ) - 3j * cirq.PauliString(cirq.Y(q0))
    assert y1 * psum1 == cirq.PauliString(cirq.X(q0) * cirq.Y(q1)) + 2 * cirq.PauliString(
        cirq.Y(q0) * cirq.Y(q1)
    ) + 3 * cirq.PauliString(cirq.Z(q0) * cirq.Y(q1))
    assert cirq.PauliString(cirq.I(q0)) * psum1 == psum1
    assert psum1 * x0 == cirq.PauliString(cirq.I(q0)) - 2j * cirq.PauliString(
        cirq.Z(q0)
    ) + 3j * cirq.PauliString(cirq.Y(q0))
    assert psum1 * y1 == y1 * psum1

    psum1 *= cirq.Z(q0)
    assert psum1 == -1j * cirq.Y(q0) + 2j * cirq.X(q0) + 3


def test_paulisum_mul_paulisum():
    q0, q1, q2 = cirq.LineQubit.range(3)

    psum1 = cirq.X(q0) + 2 * cirq.Y(q0) * cirq.Y(q1)
    psum2 = cirq.X(q0) * cirq.Y(q1) + 3 * cirq.Z(q2)
    assert psum1 * psum2 == cirq.Y(q1) + 3 * cirq.X(q0) * cirq.Z(q2) - 2j * cirq.Z(q0) + 6 * cirq.Y(
        q0
    ) * cirq.Y(q1) * cirq.Z(q2)
    assert psum2 * psum1 == cirq.Y(q1) + 3 * cirq.X(q0) * cirq.Z(q2) + 2j * cirq.Z(q0) + 6 * cirq.Y(
        q0
    ) * cirq.Y(q1) * cirq.Z(q2)
    psum3 = cirq.X(q1) + cirq.X(q2)
    psum1 *= psum3
    assert psum1 == cirq.X(q0) * cirq.X(q1) - 2j * cirq.Y(q0) * cirq.Z(q1) + cirq.X(q0) * cirq.X(
        q2
    ) + 2 * cirq.Y(q0) * cirq.Y(q1) * cirq.X(q2)

    psum4 = cirq.X(q0) + cirq.Y(q0) + cirq.Z(q1)
    psum5 = cirq.Z(q0) + cirq.Y(q0) + cirq.PauliString(coefficient=1.2)
    assert (
        psum4 * psum5
        == -1j * cirq.Y(q0)
        + 1j * (cirq.X(q0) + cirq.Z(q0))
        + (cirq.Z(q0) + cirq.Y(q0)) * cirq.Z(q1)
        + 1
        + 1.2 * psum4
    )
    assert (
        psum5 * psum4
        == 1j * cirq.Y(q0)
        + -1j * (cirq.X(q0) + cirq.Z(q0))
        + (cirq.Z(q0) + cirq.Y(q0)) * cirq.Z(q1)
        + 1
        + 1.2 * psum4
    )


def test_pauli_sum_pow():
    identity = cirq.PauliSum.from_pauli_strings([cirq.PauliString(coefficient=1)])
    psum1 = cirq.X(q0) + cirq.Y(q0)
    assert psum1 ** 2 == psum1 * psum1
    assert psum1 ** 2 == 2 * identity

    psum2 = cirq.X(q0) + cirq.Y(q1)
    assert psum2 ** 2 == cirq.PauliString(cirq.I(q0)) + 2 * cirq.X(q0) * cirq.Y(
        q1
    ) + cirq.PauliString(cirq.I(q1))

    psum3 = cirq.X(q0) * cirq.Z(q1) + 1.3 * cirq.Z(q0)
    sqd = cirq.PauliSum.from_pauli_strings([2.69 * cirq.PauliString(cirq.I(q0))])
    assert cirq.approx_eq(psum3 ** 2, sqd, atol=1e-8)

    psum4 = cirq.X(q0) * cirq.Z(q1) + 1.3 * cirq.Z(q1)
    sqd2 = cirq.PauliSum.from_pauli_strings([2.69 * cirq.PauliString(cirq.I(q0)), 2.6 * cirq.X(q0)])
    assert cirq.approx_eq(psum4 ** 2, sqd2, atol=1e-8)

    for psum in [psum1, psum2, psum3, psum4]:
        assert cirq.approx_eq(psum ** 0, identity)


# Using the entries of table 1 of https://arxiv.org/abs/1804.09130 as golden values.
@pytest.mark.parametrize(
    'boolean_expr,expected_pauli_sum',
    [
        ('x', ['(-0.5+0j)*Z(x)', '(0.5+0j)*I']),
        ('~x', ['(0.5+0j)*I', '(0.5+0j)*Z(x)']),
        ('x0 ^ x1', ['(-0.5+0j)*Z(x0)*Z(x1)', '(0.5+0j)*I']),
        (
            'x0 & x1',
            ['(-0.25+0j)*Z(x0)', '(-0.25+0j)*Z(x1)', '(0.25+0j)*I', '(0.25+0j)*Z(x0)*Z(x1)'],
        ),
        (
            'x0 | x1',
            ['(-0.25+0j)*Z(x0)', '(-0.25+0j)*Z(x0)*Z(x1)', '(-0.25+0j)*Z(x1)', '(0.75+0j)*I'],
        ),
        ('x0 ^ x1 ^ x2', ['(-0.5+0j)*Z(x0)*Z(x1)*Z(x2)', '(0.5+0j)*I']),
    ],
)
def test_from_boolean_expression(boolean_expr, expected_pauli_sum):
    boolean = sympy_parser.parse_expr(boolean_expr)
    qubit_map = {name: cirq.NamedQubit(name) for name in sorted(cirq.parameter_names(boolean))}
    actual = cirq.PauliSum.from_boolean_expression(boolean, qubit_map)
    # Instead of calling str() directly, first make sure that the items are sorted. This is to make
    # the unit test more robut in case Sympy would result in a different parsing order. By sorting
    # the individual items, we would have a canonical representation.
    actual_items = list(sorted(str(pauli_string) for pauli_string in actual))
    assert expected_pauli_sum == actual_items


def test_unsupported_op():
    not_a_boolean = sympy_parser.parse_expr('x * x')
    qubit_map = {name: cirq.NamedQubit(name) for name in cirq.parameter_names(not_a_boolean)}
    with pytest.raises(ValueError, match='Unsupported type'):
        cirq.PauliSum.from_boolean_expression(not_a_boolean, qubit_map)


def test_imul_aliasing():
    q0, q1, q2 = cirq.LineQubit.range(3)
    psum1 = cirq.X(q0) + cirq.Y(q1)
    psum2 = psum1
    psum2 *= cirq.X(q0) * cirq.Y(q2)
    assert psum1 is psum2
    assert psum1 == psum2


def test_expectation_from_state_vector_invalid_input():
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    psum = cirq.X(q0) + 2 * cirq.Y(q1) + 3 * cirq.Z(q3)
    q_map = {q0: 0, q1: 1, q3: 2}
    wf = np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.complex64)

    im_psum = (1j + 1) * psum
    with pytest.raises(NotImplementedError, match='non-Hermitian'):
        im_psum.expectation_from_state_vector(wf, q_map)

    with pytest.raises(TypeError, match='dtype'):
        psum.expectation_from_state_vector(np.array([1, 0], dtype=int), q_map)

    with pytest.raises(TypeError, match='mapping'):
        psum.expectation_from_state_vector(wf, "bad type")
    with pytest.raises(TypeError, match='mapping'):
        psum.expectation_from_state_vector(wf, {"bad key": 1})
    with pytest.raises(TypeError, match='mapping'):
        psum.expectation_from_state_vector(wf, {q0: "bad value"})
    with pytest.raises(ValueError, match='complete'):
        psum.expectation_from_state_vector(wf, {q0: 0})
    with pytest.raises(ValueError, match='complete'):
        psum.expectation_from_state_vector(wf, {q0: 0, q2: 2})
    with pytest.raises(ValueError, match='indices'):
        psum.expectation_from_state_vector(wf, {q0: -1, q1: 1, q3: 2})
    with pytest.raises(ValueError, match='indices'):
        psum.expectation_from_state_vector(wf, {q0: 0, q1: 3, q3: 2})
    with pytest.raises(ValueError, match='indices'):
        psum.expectation_from_state_vector(wf, {q0: 0, q1: 0, q3: 2})

    with pytest.raises(ValueError, match='9'):
        psum.expectation_from_state_vector(np.arange(9, dtype=np.complex64), q_map)
    q_map_2 = {q0: 0, q1: 1, q2: 2, q3: 3}
    with pytest.raises(ValueError, match='normalized'):
        psum.expectation_from_state_vector(np.arange(16, dtype=np.complex64), q_map_2)

    wf = np.arange(16, dtype=np.complex64) / np.linalg.norm(np.arange(16))
    with pytest.raises(ValueError, match='shape'):
        psum.expectation_from_state_vector(wf.reshape((16, 1)), q_map_2)
    with pytest.raises(ValueError, match='shape'):
        psum.expectation_from_state_vector(wf.reshape((4, 4, 1)), q_map_2)


def test_expectation_from_state_vector_check_preconditions():
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    psum = cirq.X(q0) + 2 * cirq.Y(q1) + 3 * cirq.Z(q3)
    q_map = {q0: 0, q1: 1, q2: 2, q3: 3}

    with pytest.raises(ValueError, match='normalized'):
        psum.expectation_from_state_vector(np.arange(16, dtype=np.complex64), q_map)

    _ = psum.expectation_from_state_vector(
        np.arange(16, dtype=np.complex64), q_map, check_preconditions=False
    )


def test_expectation_from_state_vector_basis_states():
    q = cirq.LineQubit.range(2)
    psum = cirq.X(q[0]) + 2 * cirq.Y(q[0]) + 3 * cirq.Z(q[0])
    q_map = {x: i for i, x in enumerate(q)}

    np.testing.assert_allclose(
        psum.expectation_from_state_vector(
            np.array([1, 1], dtype=complex) / np.sqrt(2), qubit_map=q_map
        ),
        1,
    )
    np.testing.assert_allclose(
        psum.expectation_from_state_vector(
            np.array([1, -1], dtype=complex) / np.sqrt(2), qubit_map=q_map
        ),
        -1,
    )
    np.testing.assert_allclose(
        psum.expectation_from_state_vector(
            np.array([1, 1j], dtype=complex) / np.sqrt(2), qubit_map=q_map
        ),
        2,
    )
    np.testing.assert_allclose(
        psum.expectation_from_state_vector(
            np.array([1, -1j], dtype=complex) / np.sqrt(2), qubit_map=q_map
        ),
        -2,
    )
    np.testing.assert_allclose(
        psum.expectation_from_state_vector(np.array([1, 0], dtype=complex), qubit_map=q_map), 3
    )
    np.testing.assert_allclose(
        psum.expectation_from_state_vector(np.array([0, 1], dtype=complex), qubit_map=q_map), -3
    )


def test_expectation_from_state_vector_two_qubit_states():
    q = cirq.LineQubit.range(2)
    q_map = {x: i for i, x in enumerate(q)}

    psum1 = cirq.Z(q[0]) + 3.2 * cirq.Z(q[1])
    psum2 = -1 * cirq.X(q[0]) + 2 * cirq.X(q[1])
    wf1 = np.array([0, 1, 0, 0], dtype=complex)
    for state in [wf1, wf1.reshape(2, 2)]:
        np.testing.assert_allclose(
            psum1.expectation_from_state_vector(state, qubit_map=q_map), -2.2, atol=1e-7
        )
        np.testing.assert_allclose(
            psum2.expectation_from_state_vector(state, qubit_map=q_map), 0, atol=1e-7
        )

    wf2 = np.array([1, 1, 1, 1], dtype=complex) / 2
    for state in [wf2, wf2.reshape(2, 2)]:
        np.testing.assert_allclose(
            psum1.expectation_from_state_vector(state, qubit_map=q_map), 0, atol=1e-7
        )
        np.testing.assert_allclose(
            psum2.expectation_from_state_vector(state, qubit_map=q_map), 1, atol=1e-7
        )

    psum3 = cirq.Z(q[0]) + cirq.X(q[1])
    wf3 = np.array([1, 1, 0, 0], dtype=complex) / np.sqrt(2)
    q_map_2 = {q0: 1, q1: 0}
    for state in [wf3, wf3.reshape(2, 2)]:
        np.testing.assert_allclose(
            psum3.expectation_from_state_vector(state, qubit_map=q_map), 2, atol=1e-7
        )
        np.testing.assert_allclose(
            psum3.expectation_from_state_vector(state, qubit_map=q_map_2), 0, atol=1e-7
        )


def test_expectation_from_density_matrix_invalid_input():
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    psum = cirq.X(q0) + 2 * cirq.Y(q1) + 3 * cirq.Z(q3)
    q_map = {q0: 0, q1: 1, q3: 2}
    wf = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.complex64)
    rho = np.kron(wf.conjugate().T, wf).reshape(8, 8)

    im_psum = (1j + 1) * psum
    with pytest.raises(NotImplementedError, match='non-Hermitian'):
        im_psum.expectation_from_density_matrix(rho, q_map)

    with pytest.raises(TypeError, match='dtype'):
        psum.expectation_from_density_matrix(0.5 * np.eye(2, dtype=int), q_map)

    with pytest.raises(TypeError, match='mapping'):
        psum.expectation_from_density_matrix(rho, "bad type")
    with pytest.raises(TypeError, match='mapping'):
        psum.expectation_from_density_matrix(rho, {"bad key": 1})
    with pytest.raises(TypeError, match='mapping'):
        psum.expectation_from_density_matrix(rho, {q0: "bad value"})
    with pytest.raises(ValueError, match='complete'):
        psum.expectation_from_density_matrix(rho, {q0: 0})
    with pytest.raises(ValueError, match='complete'):
        psum.expectation_from_density_matrix(rho, {q0: 0, q2: 2})
    with pytest.raises(ValueError, match='indices'):
        psum.expectation_from_density_matrix(rho, {q0: -1, q1: 1, q3: 2})
    with pytest.raises(ValueError, match='indices'):
        psum.expectation_from_density_matrix(rho, {q0: 0, q1: 3, q3: 2})
    with pytest.raises(ValueError, match='indices'):
        psum.expectation_from_density_matrix(rho, {q0: 0, q1: 0, q3: 2})

    with pytest.raises(ValueError, match='hermitian'):
        psum.expectation_from_density_matrix(1j * np.eye(8), q_map)
    with pytest.raises(ValueError, match='trace'):
        psum.expectation_from_density_matrix(np.eye(8, dtype=np.complex64), q_map)

    not_psd = np.zeros((8, 8), dtype=np.complex64)
    not_psd[0, 0] = 1.1
    not_psd[1, 1] = -0.1
    with pytest.raises(ValueError, match='semidefinite'):
        psum.expectation_from_density_matrix(not_psd, q_map)

    not_square = np.ones((8, 9), dtype=np.complex64)
    with pytest.raises(ValueError, match='shape'):
        psum.expectation_from_density_matrix(not_square, q_map)
    bad_wf = np.zeros(128, dtype=np.complex64)
    bad_wf[0] = 1
    with pytest.raises(ValueError, match='shape'):
        psum.expectation_from_density_matrix(bad_wf, q_map)

    with pytest.raises(ValueError, match='shape'):
        psum.expectation_from_density_matrix(rho.reshape((8, 8, 1)), q_map)
    with pytest.raises(ValueError, match='shape'):
        psum.expectation_from_density_matrix(rho.reshape((-1)), q_map)


def test_expectation_from_density_matrix_check_preconditions():
    q0, q1, _, q3 = cirq.LineQubit.range(4)
    psum = cirq.X(q0) + 2 * cirq.Y(q1) + 3 * cirq.Z(q3)
    q_map = {q0: 0, q1: 1, q3: 2}
    not_psd = np.zeros((8, 8), dtype=np.complex64)
    not_psd[0, 0] = 1.1
    not_psd[1, 1] = -0.1

    with pytest.raises(ValueError, match='semidefinite'):
        psum.expectation_from_density_matrix(not_psd, q_map)

    _ = psum.expectation_from_density_matrix(not_psd, q_map, check_preconditions=False)


def test_expectation_from_density_matrix_basis_states():
    q = cirq.LineQubit.range(2)
    psum = cirq.X(q[0]) + 2 * cirq.Y(q[0]) + 3 * cirq.Z(q[0])
    q_map = {x: i for i, x in enumerate(q)}

    np.testing.assert_allclose(
        psum.expectation_from_density_matrix(np.array([[1, 1], [1, 1]], dtype=complex) / 2, q_map),
        1,
    )
    np.testing.assert_allclose(
        psum.expectation_from_density_matrix(
            np.array([[1, -1], [-1, 1]], dtype=complex) / 2, q_map
        ),
        -1,
    )
    np.testing.assert_allclose(
        psum.expectation_from_density_matrix(
            np.array([[1, -1j], [1j, 1]], dtype=complex) / 2, qubit_map=q_map
        ),
        2,
    )
    np.testing.assert_allclose(
        psum.expectation_from_density_matrix(
            np.array([[1, 1j], [-1j, 1]], dtype=complex) / 2, qubit_map=q_map
        ),
        -2,
    )
    np.testing.assert_allclose(
        psum.expectation_from_density_matrix(np.array([[1, 0], [0, 0]], dtype=complex), q_map), 3
    )
    np.testing.assert_allclose(
        psum.expectation_from_density_matrix(np.array([[0, 0], [0, 1]], dtype=complex), q_map),
        -3,
    )


def test_expectation_from_density_matrix_two_qubit_states():
    q = cirq.LineQubit.range(2)
    q_map = {x: i for i, x in enumerate(q)}

    psum1 = cirq.Z(q[0]) + 3.2 * cirq.Z(q[1])
    psum2 = -1 * cirq.X(q[0]) + 2 * cirq.X(q[1])
    wf1 = np.array([0, 1, 0, 0], dtype=complex)
    rho1 = np.kron(wf1, wf1).reshape(4, 4)
    for state in [rho1, rho1.reshape(2, 2, 2, 2)]:
        np.testing.assert_allclose(
            psum1.expectation_from_density_matrix(state, qubit_map=q_map), -2.2
        )
        np.testing.assert_allclose(psum2.expectation_from_density_matrix(state, qubit_map=q_map), 0)

    wf2 = np.array([1, 1, 1, 1], dtype=complex) / 2
    rho2 = np.kron(wf2, wf2).reshape(4, 4)
    for state in [rho2, rho2.reshape(2, 2, 2, 2)]:
        np.testing.assert_allclose(psum1.expectation_from_density_matrix(state, qubit_map=q_map), 0)
        np.testing.assert_allclose(psum2.expectation_from_density_matrix(state, qubit_map=q_map), 1)

    psum3 = cirq.Z(q[0]) + cirq.X(q[1])
    wf3 = np.array([1, 1, 0, 0], dtype=complex) / np.sqrt(2)
    rho3 = np.kron(wf3, wf3).reshape(4, 4)
    q_map_2 = {q0: 1, q1: 0}
    for state in [rho3, rho3.reshape(2, 2, 2, 2)]:
        np.testing.assert_allclose(psum3.expectation_from_density_matrix(state, qubit_map=q_map), 2)
        np.testing.assert_allclose(
            psum3.expectation_from_density_matrix(state, qubit_map=q_map_2), 0
        )
