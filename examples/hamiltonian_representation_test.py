import functools
import math
import random

import numpy as np
import pytest
import sympy.parsing.sympy_parser as sympy_parser

import cirq
import examples.hamiltonian_representation as hr

# These are some of the entries of table 1 of https://arxiv.org/pdf/1804.09130.pdf.
@pytest.mark.parametrize(
    'boolean_expr,expected_hamiltonian_polynomial',
    [
        ('x', '0.50.I; -0.50.Z_0'),
        ('~x', '0.50.I; 0.50.Z_0'),
        ('x0 ^ x1', '0.50.I; -0.50.Z_0.Z_1'),
        ('x0 & x1', '0.25.I; -0.25.Z_0; 0.25.Z_0.Z_1; -0.25.Z_1'),
        ('x0 | x1', '0.75.I; -0.25.Z_0; -0.25.Z_0.Z_1; -0.25.Z_1'),
        ('x0 ^ x1 ^ x2', '0.50.I; -0.50.Z_0.Z_1.Z_2'),
    ],
)
def test_build_hamiltonian_from_boolean(boolean_expr, expected_hamiltonian_polynomial):
    boolean = sympy_parser.parse_expr(boolean_expr)
    name_to_id = hr.get_name_to_id([boolean])
    actual = hr.build_hamiltonian_from_boolean(boolean, name_to_id)
    assert expected_hamiltonian_polynomial == str(actual)


def test_unsupported_op():
    not_a_boolean = sympy_parser.parse_expr('x * x')
    name_to_id = hr.get_name_to_id([not_a_boolean])
    with pytest.raises(ValueError, match='Unsupported type'):
        hr.build_hamiltonian_from_boolean(not_a_boolean, name_to_id)


@pytest.mark.parametrize(
    'boolean_expr, expected',
    [
        ('x', [False, True]),
        ('~x', [True, False]),
        ('x0 ^ x1', [False, True, True, False]),
        ('x0 & x1', [False, False, False, True]),
        ('x0 | x1', [False, True, True, True]),
        ('x0 & x1 & x2', [False, False, False, False, False, False, False, True]),
        ('x0 & x1 & ~x2', [False, False, False, False, False, False, True, False]),
        ('x0 & ~x1 & x2', [False, False, False, False, False, True, False, False]),
        ('x0 & ~x1 & ~x2', [False, False, False, False, True, False, False, False]),
        ('~x0 & x1 & x2', [False, False, False, True, False, False, False, False]),
        ('~x0 & x1 & ~x2', [False, False, True, False, False, False, False, False]),
        ('~x0 & ~x1 & x2', [False, True, False, False, False, False, False, False]),
        ('~x0 & ~x1 & ~x2', [True, False, False, False, False, False, False, False]),
        ('x0 ^ x1 ^ x2', [False, True, True, False, True, False, False, True]),
        ('x0 | (x1 & x2)', [False, False, False, True, True, True, True, True]),
        ('x0 & (x1 | x2)', [False, False, False, False, False, True, True, True]),
        (
            '(x0 ^ x1 ^ x2) | (x2 ^ x3 ^ x4)',
            [
                False,
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                True,
                False,
                True,
                True,
                False,
                True,
                True,
                True,
                True,
            ],
        ),
        (
            '(x0 ^ x2 ^ x4) | (x1 ^ x2 ^ x3)',
            [
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                True,
                True,
                False,
                True,
                True,
                False,
                True,
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                False,
                False,
                True,
                True,
                True,
            ],
        ),
    ],
)
def test_circuit(boolean_expr, expected):
    circuit_hamiltonians, qubits = hr.build_circuit_from_boolean_expressions(
        [boolean_expr], 0.1 * math.pi
    )

    circuit_hadamard = cirq.Circuit()
    circuit_hadamard.append(cirq.H.on_each(*qubits))

    circuit = circuit_hadamard + circuit_hamiltonians

    phi = cirq.Simulator().simulate(circuit, qubit_order=qubits, initial_state=0).state_vector()
    actual = np.arctan2(phi.real, phi.imag) - math.pi / 2.0 > 0.0

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    'n_bits, expected_hs',
    [
        (1, [(), (0,)]),
        (2, [(), (0,), (0, 1), (1,)]),
        (3, [(), (0,), (0, 1), (1,), (1, 2), (0, 1, 2), (0, 2), (2,)]),
    ],
)
def test_gray_code_sorting(n_bits, expected_hs):
    hs = []
    for x in range(2 ** n_bits):
        h = []
        for i in range(n_bits):
            if x % 2 == 1:
                h.append(i)
                x -= 1
            x //= 2
        hs.append(tuple(sorted(h)))
    random.shuffle(hs)

    sorted_hs = sorted(list(hs), key=functools.cmp_to_key(hr._gray_code_comparator))

    np.testing.assert_array_equal(sorted_hs, expected_hs)


@pytest.mark.parametrize(
    'seq_a, seq_b, expected',
    [
        ((), (), 0),
        ((), (0,), -1),
        ((0,), (), 1),
        ((0,), (0,), 0),
    ],
)
def test_gray_code_comparison(seq_a, seq_b, expected):
    assert hr._gray_code_comparator(seq_a, seq_b) == expected
