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

import cmath
import random

import numpy as np
import pytest

import cirq
from cirq import value
from cirq.optimizers.two_qubit_decompositions import (
    _parity_interaction, _is_trivial_angle
)


@pytest.mark.parametrize('rad,expected', (lambda err, largeErr: [
    (np.pi/4, True),
    (np.pi/4 + err, True),
    (np.pi/4 + largeErr, False),
    (np.pi/4 - err, True),
    (np.pi/4 - largeErr, False),
    (-np.pi/4, True),
    (-np.pi/4 + err, True),
    (-np.pi/4 + largeErr, False),
    (-np.pi/4 - err, True),
    (-np.pi/4 - largeErr, False),
    (0, True),
    (err, True),
    (largeErr, False),
    (-err, True),
    (-largeErr, False),
    (np.pi/8, False),
    (-np.pi/8, False),
])(1e-8*2/3, 1e-8*4/3))
def test_is_trivial_angle(rad, expected):
    tolerance = 1e-8
    out = _is_trivial_angle(rad, tolerance)
    assert out == expected, 'rad = {}'.format(rad)


def _operations_to_matrix(operations, qubits):
    return cirq.Circuit.from_ops(operations).to_unitary_matrix(
        qubit_order=cirq.QubitOrder.explicit(qubits),
        qubits_that_should_be_present=qubits)


def _random_single_partial_cz_effect():
    return cirq.dot(
        cirq.kron(cirq.testing.random_unitary(2),
                  cirq.testing.random_unitary(2)),
        np.diag([1, 1, 1, cmath.exp(2j * random.random() * np.pi)]),
        cirq.kron(cirq.testing.random_unitary(2),
                  cirq.testing.random_unitary(2)))


def _random_double_partial_cz_effect():
    return cirq.dot(
        cirq.kron(cirq.testing.random_unitary(2),
                  cirq.testing.random_unitary(2)),
        np.diag([1, 1, 1, cmath.exp(2j * random.random() * np.pi)]),
        cirq.kron(cirq.testing.random_unitary(2),
                  cirq.testing.random_unitary(2)),
        np.diag([1, 1, 1, cmath.exp(2j * random.random() * np.pi)]),
        cirq.kron(cirq.testing.random_unitary(2),
                  cirq.testing.random_unitary(2)))


def _random_double_full_cz_effect():
    return cirq.dot(
        cirq.kron(cirq.testing.random_unitary(2),
                  cirq.testing.random_unitary(2)),
        cirq.unitary(cirq.CZ),
        cirq.kron(cirq.testing.random_unitary(2),
                  cirq.testing.random_unitary(2)),
        cirq.unitary(cirq.CZ),
        cirq.kron(cirq.testing.random_unitary(2),
                  cirq.testing.random_unitary(2)))


def assert_cz_depth_below(operations, threshold, must_be_full):
    total_cz = 0

    for op in operations:
        assert len(op.qubits) <= 2
        if len(op.qubits) == 2:
            assert isinstance(op, cirq.GateOperation)
            assert isinstance(op.gate, cirq.CZPowGate)
            e = value.canonicalize_half_turns(op.gate.exponent)
            if must_be_full:
                assert e == 1
            total_cz += abs(e)

    assert total_cz <= threshold


def assert_ops_implement_unitary(q0, q1, operations, intended_effect,
                                 atol=0.01):
    actual_effect = _operations_to_matrix(operations, (q0, q1))
    assert cirq.allclose_up_to_global_phase(actual_effect, intended_effect,
                                              atol=atol)


@pytest.mark.parametrize('max_partial_cz_depth,max_full_cz_depth,effect', [
    (0, 0, np.eye(4)),
    (0, 0, np.array([
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0j],
    ])),
    (0, 0, cirq.unitary(cirq.CZ**0.00000001)),

    (0.5, 2, cirq.unitary(cirq.CZ**0.5)),

    (1, 1, cirq.unitary(cirq.CZ)),
    (1, 1, cirq.unitary(cirq.CNOT)),
    (1, 1, np.array([
        [1, 0, 0, 1j],
        [0, 1, 1j, 0],
        [0, 1j, 1, 0],
        [1j, 0, 0, 1],
    ]) * np.sqrt(0.5)),
    (1, 1, np.array([
        [1, 0, 0, -1j],
        [0, 1, -1j, 0],
        [0, -1j, 1, 0],
        [-1j, 0, 0, 1],
    ]) * np.sqrt(0.5)),
    (1, 1, np.array([
        [1, 0, 0, 1j],
        [0, 1, -1j, 0],
        [0, -1j, 1, 0],
        [1j, 0, 0, 1],
    ]) * np.sqrt(0.5)),

    (1.5, 3, cirq.map_eigenvalues(cirq.unitary(cirq.SWAP),
                                  lambda e: e**0.5)),

    (2, 2, cirq.unitary(cirq.SWAP).dot(cirq.unitary(cirq.CZ))),

    (3, 3, cirq.unitary(cirq.SWAP)),
    (3, 3, np.array([
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0j],
    ])),
] + [
    (1, 2, _random_single_partial_cz_effect()) for _ in range(10)
] + [
    (2, 2, _random_double_full_cz_effect()) for _ in range(10)
] + [
    (2, 3, _random_double_partial_cz_effect()) for _ in range(10)
] + [
    (3, 3, cirq.testing.random_unitary(4)) for _ in range(10)
])
def test_two_to_ops_equivalent_and_bounded_for_known_and_random(
        max_partial_cz_depth,
        max_full_cz_depth,
        effect):
    q0 = cirq.NamedQubit('q0')
    q1 = cirq.NamedQubit('q1')

    operations_with_partial = cirq.two_qubit_matrix_to_operations(
        q0, q1, effect, True)
    operations_with_full = cirq.two_qubit_matrix_to_operations(
        q0, q1, effect, False)

    assert_ops_implement_unitary(q0, q1, operations_with_partial, effect)
    assert_ops_implement_unitary(q0, q1, operations_with_full, effect)

    assert_cz_depth_below(operations_with_partial, max_partial_cz_depth, False)
    assert_cz_depth_below(operations_with_full, max_full_cz_depth, True)


def test_trivial_parity_interaction_corner_case():
    q0 = cirq.NamedQubit('q0')
    q1 = cirq.NamedQubit('q1')
    nearPi4 = np.pi/4 * 0.99
    tolerance = 1e-2
    circuit = cirq.Circuit.from_ops(
        _parity_interaction(q0, q1, -nearPi4, tolerance))
    assert len(circuit) == 2


def test_kak_decomposition_depth_full_cz():
    a, b = cirq.LineQubit.range(2)

    # Random.
    u = cirq.testing.random_unitary(4)
    operations_with_full = cirq.two_qubit_matrix_to_operations(a, b, u, False)
    c = cirq.Circuit.from_ops(operations_with_full)
    # 3 CZ, 3+1 PhasedX, 1 Z
    assert len(c) <= 8

    # Double-axis interaction.
    u = cirq.unitary(cirq.Circuit.from_ops(cirq.CNOT(a, b),
                                           cirq.CNOT(b, a)))
    operations_with_part = cirq.two_qubit_matrix_to_operations(a, b, u, False)
    c = cirq.Circuit.from_ops(operations_with_part)
    # 2 CZ, 2+1 PhasedX, 1 Z
    assert len(c) <= 6

    # Partial single-axis interaction.
    u = cirq.unitary(cirq.CNOT**0.1)
    operations_with_part = cirq.two_qubit_matrix_to_operations(a, b, u, False)
    c = cirq.Circuit.from_ops(operations_with_part)
    # 2 CZ, 2+1 PhasedX, 1 Z
    assert len(c) <= 6

    # Full single-axis interaction.
    u = cirq.unitary(cirq.ControlledGate(cirq.Y))
    operations_with_part = cirq.two_qubit_matrix_to_operations(a, b, u, False)
    c = cirq.Circuit.from_ops(operations_with_part)
    # 1 CZ, 1+1 PhasedX, 1 Z
    assert len(c) <= 4


def test_kak_decomposition_depth_partial_cz():
    a, b = cirq.LineQubit.range(2)

    # Random.
    u = cirq.testing.random_unitary(4)
    operations_with_full = cirq.two_qubit_matrix_to_operations(a, b, u, True)
    c = cirq.Circuit.from_ops(operations_with_full)
    # 3 CP, 3+1 PhasedX, 1 Z
    assert len(c) <= 8

    # Double-axis interaction.
    u = cirq.unitary(cirq.Circuit.from_ops(cirq.CNOT(a, b),
                                           cirq.CNOT(b, a)))
    operations_with_part = cirq.two_qubit_matrix_to_operations(a, b, u, True)
    c = cirq.Circuit.from_ops(operations_with_part)
    # 2 CP, 2+1 PhasedX, 1 Z
    assert len(c) <= 6

    # Partial single-axis interaction.
    u = cirq.unitary(cirq.CNOT**0.1)
    operations_with_part = cirq.two_qubit_matrix_to_operations(a, b, u, True)
    c = cirq.Circuit.from_ops(operations_with_part)
    # 1 CP, 1+1 PhasedX, 1 Z
    assert len(c) <= 4

    # Full single-axis interaction.
    u = cirq.unitary(cirq.ControlledGate(cirq.Y))
    operations_with_part = cirq.two_qubit_matrix_to_operations(a, b, u, True)
    c = cirq.Circuit.from_ops(operations_with_part)
    # 1 CP, 1+1 PhasedX, 1 Z
    assert len(c) <= 4
