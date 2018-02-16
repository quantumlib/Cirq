# Copyright 2018 Google LLC
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
import math
import random

import numpy as np

import pytest

from cirq import circuits
from cirq import linalg
from cirq import ops
from cirq import testing


def cis(x: float):
    return cmath.exp(1j * x)


def inv_unit(x: complex) -> complex:
    return cis(-cmath.phase(x))


def _operation_to_matrix(operation, qubits):
    # All qubits operation.
    if operation.qubits == tuple(qubits):
        return operation.gate.matrix()
    if operation.qubits == tuple(qubits[::-1]) and isinstance(
            operation.gate, ops.Rot11Gate):
        return operation.gate.matrix()

    # Single qubit operation.
    for i in range(len(qubits)):
        if operation.qubits == (qubits[i],):
            m = operation.gate.matrix()
            below = np.eye(1 << i)
            above = np.eye(1 << (len(qubits) - i - 1))
            return linalg.kron(above, m, below)

    raise NotImplementedError(str(operation))


def _operations_to_matrix(operations, qubits):
    m = np.eye(1 << len(qubits), dtype=np.complex128)
    for op in operations:
        m = _operation_to_matrix(op, qubits).dot(m)
    return m


def _gates_to_matrix(gates):
    m = gates[0].matrix()
    for gate in gates[1:]:
        m = gate.matrix().dot(m)
    return m


def assert_gates_implement_unitary(gates, intended_effect):
    actual_effect = _gates_to_matrix(gates)
    assert linalg.allclose_up_to_global_phase(actual_effect, intended_effect)


def test_single_qubit_matrix_to_native_gates_known_x():
    actual = circuits.single_qubit_matrix_to_native_gates(
        np.array([[0, 1], [1, 0]]), tolerance=0.01)

    assert actual == [ops.X]


def test_single_qubit_matrix_to_native_gates_known_y():
    actual = circuits.single_qubit_matrix_to_native_gates(
        np.array([[0, -1j], [1j, 0]]), tolerance=0.01)

    assert actual == [ops.Y]


def test_single_qubit_matrix_to_native_gates_known_z():
    actual = circuits.single_qubit_matrix_to_native_gates(
        np.array([[1, 0], [0, -1]]), tolerance=0.01)

    assert actual == [ops.Z]


def test_single_qubit_matrix_to_native_gates_known_s():
    actual = circuits.single_qubit_matrix_to_native_gates(
        np.array([[1, 0], [0, 1j]]), tolerance=0.01)

    assert actual == [ops.Z**0.5]


def test_known_s_dag():
    actual = circuits.single_qubit_matrix_to_native_gates(
        np.array([[1, 0], [0, -1j]]), tolerance=0.01)

    assert actual == [ops.Z**-0.5]


def test_known_h():
    actual = circuits.single_qubit_matrix_to_native_gates(
        np.array([[1, 1], [1, -1]]) * np.sqrt(0.5), tolerance=0.001)

    assert actual == [ops.Y**-0.5, ops.Z]


@pytest.mark.parametrize('intended_effect', [
    np.array([[0, 1j], [1, 0]]),
] + [
    testing.random_unitary(2) for _ in range(10)
])
def test_single_qubit_matrix_to_native_gates_cases(intended_effect):
    gates = circuits.single_qubit_matrix_to_native_gates(
        intended_effect, tolerance=0.0001)
    assert len(gates) <= 2
    assert_gates_implement_unitary(gates, intended_effect)


@pytest.mark.parametrize('pre_turns,post_turns',
                         [(random.random(), random.random())
                          for _ in range(10)])
def test_single_qubit_matrix_to_native_gates_fuzz_half_turns_always_one_gate(
        pre_turns, post_turns):
    intended_effect = linalg.dot(
        ops.RotZGate(half_turns=2 * pre_turns).matrix(),
        ops.X.matrix(),
        ops.RotZGate(half_turns=2 * post_turns).matrix())

    gates = circuits.single_qubit_matrix_to_native_gates(
        intended_effect, tolerance=0.0001)

    assert len(gates) == 1
    assert_gates_implement_unitary(gates, intended_effect)


def test_single_qubit_matrix_to_native_gates_tolerance_z():
    z = np.diag([1, np.exp(1j * 0.01)])

    optimized_away = circuits.single_qubit_matrix_to_native_gates(
        z, tolerance=0.1)
    assert len(optimized_away) == 0

    kept = circuits.single_qubit_matrix_to_native_gates(z,
                                                        tolerance=0.0001)
    assert len(kept) == 1


def test_single_qubit_matrix_to_native_gates_tolerance_xy():
    c, s = np.cos(0.01), np.sin(0.01)
    xy = np.array([[c, -s], [s, c]])

    optimized_away = circuits.single_qubit_matrix_to_native_gates(
        xy, tolerance=0.1)
    assert len(optimized_away) == 0

    kept = circuits.single_qubit_matrix_to_native_gates(xy,
                                                        tolerance=0.0001)
    assert len(kept) == 1


def test_single_qubit_matrix_to_native_gates_tolerance_half_turn_phasing():
    a = np.pi / 2 + 0.01
    c, s = np.cos(a), np.sin(a)
    nearly_x = np.array([[c, -s], [s, c]])
    z1 = np.diag([1, np.exp(1j * 1.2)])
    z2 = np.diag([1, np.exp(1j * 1.6)])
    phased_nearly_x = z1.dot(nearly_x).dot(z2)

    optimized_away = circuits.single_qubit_matrix_to_native_gates(
        phased_nearly_x, tolerance=0.1)
    assert len(optimized_away) == 1

    kept = circuits.single_qubit_matrix_to_native_gates(
        phased_nearly_x, tolerance=0.0001)
    assert len(kept) == 2


def test_single_qubit_op_to_framed_phase_form_output_on_example_case():
    u, t, g = circuits.single_qubit_op_to_framed_phase_form(
        (ops.Y**0.25).matrix())
    assert linalg.allclose_up_to_global_phase(u, (ops.X**0.5).matrix())
    assert abs(t - (1 + 1j) * math.sqrt(0.5)) < 0.00001
    assert abs(g - 1) < 0.00001


@pytest.mark.parametrize('mat', [
    np.eye(2),
    ops.H.matrix(),
    ops.X.matrix(),
    (ops.X**0.5).matrix(),
    ops.Y.matrix(),
    ops.Z.matrix(),
    (ops.Z**0.5).matrix(),
] + [testing.random_unitary(2)
     for _ in range(10)])
def test_single_qubit_op_to_framed_phase_form_equivalent_on_known_and_random(
        mat):
    u, t, g = circuits.single_qubit_op_to_framed_phase_form(mat)
    z = np.diag([g, g * t])
    assert np.allclose(mat, np.conj(u.T).dot(z).dot(u))


def test_controlled_op_to_gates_concrete_case():
    qc = ops.QubitId()
    qt = ops.QubitId()
    operations = circuits.controlled_op_to_native_gates(
        control=qc,
        target=qt,
        operation=np.array([[1, 1j], [1j, 1]]) * np.sqrt(0.5),
        tolerance=0.0001)

    assert operations == [ops.Y(qt)**-0.5, ops.CZ(qc, qt)**1.5,
                          ops.Z(qc)**0.25, ops.Y(qt)**0.5]


def test_controlled_op_to_gates_omits_negligible_global_phase():
    qc = ops.QubitId()
    qt = ops.QubitId()
    operations = circuits.controlled_op_to_native_gates(
        control=qc, target=qt, operation=ops.H.matrix(), tolerance=0.0001)

    assert operations == [ops.Y(qt)**-0.25, ops.CZ(qc, qt), ops.Y(qt)**0.25]


@pytest.mark.parametrize('mat', [
    np.eye(2),
    ops.H.matrix(),
    ops.X.matrix(),
    (ops.X**0.5).matrix(),
    ops.Y.matrix(),
    ops.Z.matrix(),
    (ops.Z**0.5).matrix(),
] + [
    testing.random_unitary(2) for _ in range(10)
])
def test_controlled_op_to_gates_equivalent_on_known_and_random(mat):
    qc = ops.QubitId()
    qt = ops.QubitId()
    operations = circuits.controlled_op_to_native_gates(
        control=qc, target=qt, operation=mat)
    actual_effect = _operations_to_matrix(operations, (qc, qt))
    intended_effect = linalg.kron_with_controls(mat, linalg.CONTROL_TAG)
    assert linalg.allclose_up_to_global_phase(actual_effect, intended_effect)


def _random_single_partial_cz_effect():
    return linalg.dot(
        linalg.kron(testing.random_unitary(2), testing.random_unitary(2)),
        np.diag([1, 1, 1, cmath.exp(2j * random.random() * np.pi)]),
        linalg.kron(testing.random_unitary(2), testing.random_unitary(2)))


def _random_double_partial_cz_effect():
    return linalg.dot(
        linalg.kron(testing.random_unitary(2), testing.random_unitary(2)),
        np.diag([1, 1, 1, cmath.exp(2j * random.random() * np.pi)]),
        linalg.kron(testing.random_unitary(2), testing.random_unitary(2)),
        np.diag([1, 1, 1, cmath.exp(2j * random.random() * np.pi)]),
        linalg.kron(testing.random_unitary(2), testing.random_unitary(2)))


def _random_double_full_cz_effect():
    return linalg.dot(
        linalg.kron(testing.random_unitary(2), testing.random_unitary(2)),
        ops.CZ.matrix(),
        linalg.kron(testing.random_unitary(2), testing.random_unitary(2)),
        ops.CZ.matrix(),
        linalg.kron(testing.random_unitary(2), testing.random_unitary(2)))


def assert_cz_depth_below(operations, threshold, must_be_full):
    total_cz = 0

    for op in operations:
        assert len(op.qubits) <= 2
        if len(op.qubits) == 2:
            assert isinstance(op.gate, ops.Rot11Gate)
            if must_be_full:
                assert op.gate.half_turns == 1
            total_cz += abs(op.gate.half_turns)

    assert total_cz <= threshold


def assert_ops_implement_unitary(q0, q1, operations, intended_effect,
                                 atol=0.01):
    actual_effect = _operations_to_matrix(operations, (q0, q1))
    assert linalg.allclose_up_to_global_phase(actual_effect, intended_effect,
                                              atol=atol)


@pytest.mark.parametrize('max_partial_cz_depth,max_full_cz_depth,effect', [
    [0, 0, np.eye(4)],
    [0, 0, np.array([
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0j],
    ])],
    [0, 0, (ops.CZ**0.00000001).matrix()],

    [0.5, 2, (ops.CZ**0.5).matrix()],

    [1, 1, ops.CZ.matrix()],
    [1, 1, ops.CNOT.matrix()],
    [1, 1, np.array([
        [1, 0, 0, 1j],
        [0, 1, 1j, 0],
        [0, 1j, 1, 0],
        [1j, 0, 0, 1],
    ]) * np.sqrt(0.5)],
    [1, 1, np.array([
        [1, 0, 0, -1j],
        [0, 1, -1j, 0],
        [0, -1j, 1, 0],
        [-1j, 0, 0, 1],
    ]) * np.sqrt(0.5)],
    [1, 1, np.array([
        [1, 0, 0, 1j],
        [0, 1, -1j, 0],
        [0, -1j, 1, 0],
        [1j, 0, 0, 1],
    ]) * np.sqrt(0.5)],

    [1.5, 3, linalg.map_eigenvalues(ops.SWAP.matrix(),
                                    lambda e: complex(e)**0.5)],

    [2, 2, ops.SWAP.matrix().dot(ops.CZ.matrix())],

    [3, 3, ops.SWAP.matrix()],
    [3, 3, np.array([
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0j],
    ])],
] + [
    (1, 2, _random_single_partial_cz_effect()) for _ in range(10)
] + [
    (2, 2, _random_double_full_cz_effect()) for _ in range(10)
] + [
    (2, 3, _random_double_partial_cz_effect()) for _ in range(10)
] + [
    (3, 3, testing.random_unitary(4)) for _ in range(10)
])
def test_two_to_native_equivalent_and_bounded_for_known_and_random(
        max_partial_cz_depth,
        max_full_cz_depth,
        effect):
    q0 = ops.QubitId()
    q1 = ops.QubitId()

    operations_with_partial = circuits.two_qubit_matrix_to_native_gates(
        q0, q1, effect, True)
    operations_with_full = circuits.two_qubit_matrix_to_native_gates(
        q0, q1, effect, False)

    assert_ops_implement_unitary(q0, q1, operations_with_partial, effect)
    assert_ops_implement_unitary(q0, q1, operations_with_full, effect)

    assert_cz_depth_below(operations_with_partial, max_partial_cz_depth, False)
    assert_cz_depth_below(operations_with_full, max_full_cz_depth, True)
