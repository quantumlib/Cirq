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
from typing import Iterable, List

import random

import numpy as np
import pytest

import cirq
from cirq.google import decompositions


def _operations_to_matrix(operations: Iterable[cirq.Operation],
                          qubits: Iterable[cirq.QubitId]):
    return cirq.Circuit.from_ops(operations).to_unitary_matrix(
        qubit_order=cirq.QubitOrder.explicit(qubits),
        qubits_that_should_be_present=qubits)


def assert_gates_implement_unitary(gates: List[cirq.SingleQubitGate],
                                   intended_effect: np.ndarray):
    actual_effect = cirq.dot(*(cirq.unitary_effect(g) for g in gates))
    assert cirq.allclose_up_to_global_phase(actual_effect, intended_effect)


def test_single_qubit_matrix_to_native_gates_known_x():
    actual = decompositions.single_qubit_matrix_to_native_gates(
        np.array([[0, 1], [1, 0]]), tolerance=0.01)

    assert actual == [cirq.X]


def test_single_qubit_matrix_to_native_gates_known_y():
    actual = decompositions.single_qubit_matrix_to_native_gates(
        np.array([[0, -1j], [1j, 0]]), tolerance=0.01)

    assert actual == [cirq.Y]


def test_single_qubit_matrix_to_native_gates_known_z():
    actual = decompositions.single_qubit_matrix_to_native_gates(
        np.array([[1, 0], [0, -1]]), tolerance=0.01)

    assert actual == [cirq.Z]


def test_single_qubit_matrix_to_native_gates_known_s():
    actual = decompositions.single_qubit_matrix_to_native_gates(
        np.array([[1, 0], [0, 1j]]), tolerance=0.01)

    assert actual == [cirq.Z**0.5]


def test_known_s_dag():
    actual = decompositions.single_qubit_matrix_to_native_gates(
        np.array([[1, 0], [0, -1j]]), tolerance=0.01)

    assert actual == [cirq.Z**-0.5]


def test_known_h():
    actual = decompositions.single_qubit_matrix_to_native_gates(
        np.array([[1, 1], [1, -1]]) * np.sqrt(0.5), tolerance=0.001)

    assert actual == [cirq.Y**-0.5, cirq.Z]


@pytest.mark.parametrize('intended_effect', [
    np.array([[0, 1j], [1, 0]]),
] + [
    cirq.testing.random_unitary(2) for _ in range(10)
])
def test_single_qubit_matrix_to_native_gates_cases(intended_effect):
    gates = decompositions.single_qubit_matrix_to_native_gates(
        intended_effect, tolerance=0.0001)
    assert len(gates) <= 2
    assert_gates_implement_unitary(gates, intended_effect)


@pytest.mark.parametrize('pre_turns,post_turns',
                         [(random.random(), random.random())
                          for _ in range(10)])
def test_single_qubit_matrix_to_native_gates_fuzz_half_turns_always_one_gate(
        pre_turns, post_turns):
    intended_effect = cirq.dot(
        cirq.unitary_effect(cirq.RotZGate(half_turns=2 * pre_turns)),
        cirq.unitary_effect(cirq.X),
        cirq.unitary_effect(cirq.RotZGate(half_turns=2 * post_turns)))

    gates = decompositions.single_qubit_matrix_to_native_gates(
        intended_effect, tolerance=0.0001)

    assert len(gates) == 1
    assert_gates_implement_unitary(gates, intended_effect)


def test_single_qubit_matrix_to_native_gates_tolerance_z():
    z = np.diag([1, np.exp(1j * 0.01)])

    optimized_away = decompositions.single_qubit_matrix_to_native_gates(
        z, tolerance=0.1)
    assert len(optimized_away) == 0

    kept = decompositions.single_qubit_matrix_to_native_gates(z,
                                                              tolerance=0.0001)
    assert len(kept) == 1


def test_single_qubit_matrix_to_native_gates_tolerance_xy():
    c, s = np.cos(0.01), np.sin(0.01)
    xy = np.array([[c, -s], [s, c]])

    optimized_away = decompositions.single_qubit_matrix_to_native_gates(
        xy, tolerance=0.1)
    assert len(optimized_away) == 0

    kept = decompositions.single_qubit_matrix_to_native_gates(xy,
                                                              tolerance=0.0001)
    assert len(kept) == 1


def test_single_qubit_matrix_to_native_gates_tolerance_half_turn_phasing():
    a = np.pi / 2 + 0.01
    c, s = np.cos(a), np.sin(a)
    nearly_x = np.array([[c, -s], [s, c]])
    z1 = np.diag([1, np.exp(1j * 1.2)])
    z2 = np.diag([1, np.exp(1j * 1.6)])
    phased_nearly_x = z1.dot(nearly_x).dot(z2)

    optimized_away = decompositions.single_qubit_matrix_to_native_gates(
        phased_nearly_x, tolerance=0.1)
    assert len(optimized_away) == 1

    kept = decompositions.single_qubit_matrix_to_native_gates(
        phased_nearly_x, tolerance=0.0001)
    assert len(kept) == 2


def test_controlled_op_to_gates_concrete_case():
    c = cirq.NamedQubit('c')
    t = cirq.NamedQubit('t')
    operations = decompositions.controlled_op_to_native_gates(
        control=c,
        target=t,
        operation=np.array([[1, 1j], [1j, 1]]) * np.sqrt(0.5),
        tolerance=0.0001)

    assert operations == [cirq.Y(t)**-0.5, cirq.CZ(c, t)**1.5,
                          cirq.Z(c)**0.25, cirq.Y(t)**0.5]


def test_controlled_op_to_gates_omits_negligible_global_phase():
    qc = cirq.QubitId()
    qt = cirq.QubitId()
    operations = decompositions.controlled_op_to_native_gates(
        control=qc,
        target=qt,
        operation=cirq.unitary_effect(cirq.H),
        tolerance=0.0001)

    assert operations == [cirq.Y(qt)**-0.25, cirq.CZ(qc, qt), cirq.Y(qt)**0.25]


@pytest.mark.parametrize('mat', [
    np.eye(2),
    cirq.unitary_effect(cirq.H),
    cirq.unitary_effect(cirq.X),
    cirq.unitary_effect(cirq.X**0.5),
    cirq.unitary_effect(cirq.Y),
    cirq.unitary_effect(cirq.Z),
    cirq.unitary_effect(cirq.Z**0.5),
] + [
    cirq.testing.random_unitary(2) for _ in range(10)
])
def test_controlled_op_to_gates_equivalent_on_known_and_random(mat):
    qc = cirq.QubitId()
    qt = cirq.QubitId()
    operations = decompositions.controlled_op_to_native_gates(
        control=qc, target=qt, operation=mat)
    actual_effect = _operations_to_matrix(operations, (qc, qt))
    intended_effect = cirq.kron_with_controls(cirq.CONTROL_TAG, mat)
    assert cirq.allclose_up_to_global_phase(actual_effect, intended_effect)
