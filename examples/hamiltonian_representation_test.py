import math

import numpy as np
import pytest
from sympy.parsing.sympy_parser import parse_expr

import cirq
import examples.hamiltonian_representation as hr

# These are some of the entries of table 1.
@pytest.mark.parametrize(
    'boolean_expr,hamiltonian',
    [
        ('x', '0.50.I; -0.50.Z_0'),
        ('~x', '0.50.I; 0.50.Z_0'),
        ('x0 ^ x1', '0.50.I; -0.50.Z_0.Z_1'),
        ('x0 & x1', '0.25.I; -0.25.Z_0; 0.25.Z_0.Z_1; -0.25.Z_1'),
        ('x0 | x1', '0.75.I; -0.25.Z_0; -0.25.Z_0.Z_1; -0.25.Z_1'),
        ('x0 ^ x1 ^ x2', '0.50.I; -0.50.Z_0.Z_1.Z_2'),
    ],
)
def test_build_hamiltonian_from_boolean(boolean_expr, hamiltonian):
    boolean = parse_expr(boolean_expr)
    name_to_id = hr.get_name_to_id(boolean)
    actual = hr.build_hamiltonian_from_boolean(boolean, name_to_id)
    assert hamiltonian == str(actual)


def test_unsupported_op():
    not_a_boolean = parse_expr('x * x')
    name_to_id = hr.get_name_to_id(not_a_boolean)
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
        ('x0 & ~x1 & x2', [False, False, False, False, False, True, False, False]),
    ],
)
def test_circuit(boolean_expr, expected):
    boolean = parse_expr(boolean_expr)
    name_to_id = hr.get_name_to_id(boolean)
    hamiltonian = hr.build_hamiltonian_from_boolean(boolean, name_to_id)

    theta = 0.1 * math.pi
    circuit, qubits = hr.build_circuit_from_hamiltonian(hamiltonian, name_to_id, theta)

    phi = cirq.Simulator().simulate(circuit, qubit_order=qubits, initial_state=0).state_vector()
    actual = np.arctan2(phi.real, phi.imag) - math.pi / 2.0 > 0.0

    np.testing.assert_array_equal(actual, expected)
