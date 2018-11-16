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

import math
import random
from typing import Sequence

import numpy as np
import pytest

import cirq


def assert_gates_implement_unitary(gates: Sequence[cirq.SingleQubitGate],
                                   intended_effect: np.ndarray,
                                   atol: float):
    actual_effect = cirq.dot(*[cirq.unitary(g) for g in reversed(gates)])
    cirq.testing.assert_allclose_up_to_global_phase(actual_effect,
                                                    intended_effect,
                                                    atol=atol)


def test_single_qubit_matrix_to_gates_known_x():
    actual = cirq.single_qubit_matrix_to_gates(
        np.array([[0, 1], [1, 0]]), tolerance=0.01)

    assert actual == [cirq.X]


def test_single_qubit_matrix_to_gates_known_y():
    actual = cirq.single_qubit_matrix_to_gates(
        np.array([[0, -1j], [1j, 0]]), tolerance=0.01)

    assert actual == [cirq.Y]


def test_single_qubit_matrix_to_gates_known_z():
    actual = cirq.single_qubit_matrix_to_gates(
        np.array([[1, 0], [0, -1]]), tolerance=0.01)

    assert actual == [cirq.Z]


def test_single_qubit_matrix_to_gates_known_s():
    actual = cirq.single_qubit_matrix_to_gates(
        np.array([[1, 0], [0, 1j]]), tolerance=0.01)

    assert actual == [cirq.Z**0.5]


def test_known_s_dag():
    actual = cirq.single_qubit_matrix_to_gates(
        np.array([[1, 0], [0, -1j]]), tolerance=0.01)

    assert actual == [cirq.Z**-0.5]


def test_known_h():
    actual = cirq.single_qubit_matrix_to_gates(
        np.array([[1, 1], [1, -1]]) * np.sqrt(0.5), tolerance=0.001)

    assert actual == [cirq.Y**-0.5, cirq.Z]


@pytest.mark.parametrize('intended_effect', [
    np.array([[0, 1j], [1, 0]]),
    # Historical failure:
    np.array([[-0.10313355-0.62283483j,  0.76512225-0.1266025j],
              [-0.72184177+0.28352196j,  0.23073193+0.5876415j]]),
] + [
    cirq.testing.random_unitary(2) for _ in range(10)
])
def test_single_qubit_matrix_to_gates_cases(intended_effect):
    for atol in [1e-1, 1e-8]:
        gates = cirq.single_qubit_matrix_to_gates(
            intended_effect, tolerance=atol / 10)
        assert len(gates) <= 3
        assert sum(1 for g in gates if not isinstance(g, cirq.ZPowGate)) <= 1
        assert_gates_implement_unitary(gates, intended_effect, atol=atol)


@pytest.mark.parametrize('pre_turns,post_turns',
                         [(random.random(), random.random())
                          for _ in range(10)])
def test_single_qubit_matrix_to_gates_fuzz_half_turns_merge_z_gates(
        pre_turns, post_turns):
    intended_effect = cirq.dot(
        cirq.unitary(cirq.Z**(2 * pre_turns)),
        cirq.unitary(cirq.X),
        cirq.unitary(cirq.Z**(2 * post_turns)))

    gates = cirq.single_qubit_matrix_to_gates(
        intended_effect, tolerance=1e-7)

    assert len(gates) <= 2
    assert_gates_implement_unitary(gates, intended_effect, atol=1e-6)


def test_single_qubit_matrix_to_gates_tolerance_z():
    z = np.diag([1, np.exp(1j * 0.01)])

    optimized_away = cirq.single_qubit_matrix_to_gates(
        z, tolerance=0.1)
    assert len(optimized_away) == 0

    kept = cirq.single_qubit_matrix_to_gates(z, tolerance=0.0001)
    assert len(kept) == 1


def test_single_qubit_matrix_to_gates_tolerance_xy():
    c, s = np.cos(0.01), np.sin(0.01)
    xy = np.array([[c, -s], [s, c]])

    optimized_away = cirq.single_qubit_matrix_to_gates(
        xy, tolerance=0.1)
    assert len(optimized_away) == 0

    kept = cirq.single_qubit_matrix_to_gates(xy, tolerance=0.0001)
    assert len(kept) == 1


def test_single_qubit_matrix_to_gates_tolerance_half_turn_phasing():
    a = np.pi / 2 + 0.01
    c, s = np.cos(a), np.sin(a)
    nearly_x = np.array([[c, -s], [s, c]])
    z1 = np.diag([1, np.exp(1j * 1.2)])
    z2 = np.diag([1, np.exp(1j * 1.6)])
    phased_nearly_x = z1.dot(nearly_x).dot(z2)

    optimized_away = cirq.single_qubit_matrix_to_gates(
        phased_nearly_x, tolerance=0.1)
    assert len(optimized_away) == 2

    kept = cirq.single_qubit_matrix_to_gates(
        phased_nearly_x, tolerance=0.0001)
    assert len(kept) == 3


def test_single_qubit_op_to_framed_phase_form_output_on_example_case():
    u, t, g = cirq.single_qubit_op_to_framed_phase_form(
        cirq.unitary(cirq.Y**0.25))
    assert cirq.allclose_up_to_global_phase(u, cirq.unitary(cirq.X**0.5))
    assert abs(t - (1 + 1j) * math.sqrt(0.5)) < 0.00001
    assert abs(g - 1) < 0.00001


@pytest.mark.parametrize('mat', [
    np.eye(2),
    cirq.unitary(cirq.H),
    cirq.unitary(cirq.X),
    cirq.unitary(cirq.X**0.5),
    cirq.unitary(cirq.Y),
    cirq.unitary(cirq.Z),
    cirq.unitary(cirq.Z**0.5),
] + [cirq.testing.random_unitary(2)
     for _ in range(10)])
def test_single_qubit_op_to_framed_phase_form_equivalent_on_known_and_random(
        mat):
    u, t, g = cirq.single_qubit_op_to_framed_phase_form(mat)
    z = np.diag([g, g * t])
    assert np.allclose(mat, np.conj(u.T).dot(z).dot(u))


def test_single_qubit_matrix_to_native_gates_known():
    actual = cirq.single_qubit_matrix_to_phased_x_z(
        np.array([[0, 1], [1, 0]]), atol=0.01)
    assert actual == [cirq.X]

    actual = cirq.single_qubit_matrix_to_phased_x_z(
        np.array([[0, -1j], [1j, 0]]), atol=0.01)
    assert actual == [cirq.Y]

    actual = cirq.single_qubit_matrix_to_phased_x_z(
        np.array([[1, 0], [0, -1]]), atol=0.01)
    assert actual == [cirq.Z]

    actual = cirq.single_qubit_matrix_to_phased_x_z(
        np.array([[1, 0], [0, 1j]]), atol=0.01)
    assert actual == [cirq.Z**0.5]

    actual = cirq.single_qubit_matrix_to_phased_x_z(
        np.array([[1, 0], [0, -1j]]), atol=0.01)
    assert actual == [cirq.Z**-0.5]

    actual = cirq.single_qubit_matrix_to_phased_x_z(
        np.array([[1, 1], [1, -1]]) * np.sqrt(0.5), atol=0.001)
    assert actual == [cirq.Y**-0.5, cirq.Z]


@pytest.mark.parametrize('intended_effect', [
    np.array([[0, 1j], [1, 0]]),
] + [
    cirq.testing.random_unitary(2) for _ in range(10)
])
def test_single_qubit_matrix_to_native_gates_cases(intended_effect):
    gates = cirq.single_qubit_matrix_to_phased_x_z(
        intended_effect, atol=0.0001)
    assert len(gates) <= 2
    assert_gates_implement_unitary(gates, intended_effect, atol=1e-8)


@pytest.mark.parametrize('pre_turns,post_turns',
                         [(random.random(), random.random())
                          for _ in range(10)])
def test_single_qubit_matrix_to_native_gates_fuzz_half_turns_always_one_gate(
        pre_turns, post_turns):
    intended_effect = cirq.dot(
        cirq.unitary(cirq.Z**(2 * pre_turns)),
        cirq.unitary(cirq.X),
        cirq.unitary(cirq.Z**(2 * post_turns)))

    gates = cirq.single_qubit_matrix_to_phased_x_z(
        intended_effect, atol=0.0001)

    assert len(gates) == 1
    assert_gates_implement_unitary(gates, intended_effect, atol=1e-8)


def test_single_qubit_matrix_to_native_gates_tolerance_z():
    z = np.diag([1, np.exp(1j * 0.01)])

    optimized_away = cirq.single_qubit_matrix_to_phased_x_z(
        z, atol=0.1)
    assert len(optimized_away) == 0

    kept = cirq.single_qubit_matrix_to_phased_x_z(z, atol=0.0001)
    assert len(kept) == 1


def test_single_qubit_matrix_to_native_gates_tolerance_xy():
    c, s = np.cos(0.01), np.sin(0.01)
    xy = np.array([[c, -s], [s, c]])

    optimized_away = cirq.single_qubit_matrix_to_phased_x_z(
        xy, atol=0.1)
    assert len(optimized_away) == 0

    kept = cirq.single_qubit_matrix_to_phased_x_z(xy, atol=0.0001)
    assert len(kept) == 1


def test_single_qubit_matrix_to_native_gates_tolerance_half_turn_phasing():
    a = np.pi / 2 + 0.01
    c, s = np.cos(a), np.sin(a)
    nearly_x = np.array([[c, -s], [s, c]])
    z1 = np.diag([1, np.exp(1j * 1.2)])
    z2 = np.diag([1, np.exp(1j * 1.6)])
    phased_nearly_x = z1.dot(nearly_x).dot(z2)

    optimized_away = cirq.single_qubit_matrix_to_phased_x_z(
        phased_nearly_x, atol=0.1)
    assert len(optimized_away) == 1

    kept = cirq.single_qubit_matrix_to_phased_x_z(
        phased_nearly_x, atol=0.0001)
    assert len(kept) == 2
