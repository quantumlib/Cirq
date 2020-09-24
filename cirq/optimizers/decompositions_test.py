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

import random
from typing import Sequence

import numpy as np
import pytest
import sympy

import cirq


def assert_gates_implement_unitary(gates: Sequence[cirq.SingleQubitGate],
                                   intended_effect: np.ndarray,
                                   atol: float):
    actual_effect = cirq.dot(*[cirq.unitary(g) for g in reversed(gates)])
    cirq.testing.assert_allclose_up_to_global_phase(actual_effect,
                                                    intended_effect,
                                                    atol=atol)


def test_is_negligible_turn():
    assert cirq.is_negligible_turn(0, 1e-5)
    assert cirq.is_negligible_turn(1e-6, 1e-5)
    assert cirq.is_negligible_turn(1, 1e-5)
    assert cirq.is_negligible_turn(1 + 1e-6, 1e-5)
    assert cirq.is_negligible_turn(1 - 1e-6, 1e-5)
    assert cirq.is_negligible_turn(-1, 1e-5)
    assert cirq.is_negligible_turn(-1 + 1e-6, 1e-5)
    assert cirq.is_negligible_turn(-1 - 1e-6, 1e-5)
    assert cirq.is_negligible_turn(3, 1e-5)
    assert cirq.is_negligible_turn(3 + 1e-6, 1e-5)
    assert not cirq.is_negligible_turn(1e-4, 1e-5)
    assert not cirq.is_negligible_turn(-1e-4, 1e-5)
    assert not cirq.is_negligible_turn(0.5, 1e-5)
    assert not cirq.is_negligible_turn(-0.5, 1e-5)
    assert not cirq.is_negligible_turn(0.5, 1e-5)
    assert not cirq.is_negligible_turn(4.5, 1e-5)
    # Variable sympy expression
    assert not cirq.is_negligible_turn(sympy.Symbol('a'), 1e-5)
    assert not cirq.is_negligible_turn(sympy.Symbol('a') + 1, 1e-5)
    assert not cirq.is_negligible_turn(sympy.Symbol('a') * 1e-10, 1e-5)
    # Constant sympy expression
    assert cirq.is_negligible_turn(sympy.Symbol('a') * 0 + 3 + 1e-6, 1e-5)
    assert not cirq.is_negligible_turn(sympy.Symbol('a') * 0 + 1.5 - 1e-6, 1e-5)


def test_single_qubit_matrix_to_gates_known_x():
    actual = cirq.single_qubit_matrix_to_gates(
        np.array([[0, 1], [1, 0]]), tolerance=0.01)

    assert cirq.approx_eq(actual, [cirq.X], atol=1e-9)


def test_single_qubit_matrix_to_gates_known_y():
    actual = cirq.single_qubit_matrix_to_gates(
        np.array([[0, -1j], [1j, 0]]), tolerance=0.01)

    assert cirq.approx_eq(actual, [cirq.Y], atol=1e-9)


def test_single_qubit_matrix_to_gates_known_z():
    actual = cirq.single_qubit_matrix_to_gates(
        np.array([[1, 0], [0, -1]]), tolerance=0.01)

    assert cirq.approx_eq(actual, [cirq.Z], atol=1e-9)


def test_single_qubit_matrix_to_gates_known_s():
    actual = cirq.single_qubit_matrix_to_gates(
        np.array([[1, 0], [0, 1j]]), tolerance=0.01)

    assert cirq.approx_eq(actual, [cirq.Z**0.5], atol=1e-9)


def test_known_s_dag():
    actual = cirq.single_qubit_matrix_to_gates(
        np.array([[1, 0], [0, -1j]]), tolerance=0.01)

    assert cirq.approx_eq(actual, [cirq.Z**-0.5], atol=1e-9)


def test_known_h():
    actual = cirq.single_qubit_matrix_to_gates(
        np.array([[1, 1], [1, -1]]) * np.sqrt(0.5), tolerance=0.001)

    assert cirq.approx_eq(actual, [cirq.Y**-0.5, cirq.Z], atol=1e-9)


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


def _random_unitary_with_close_eigenvalues():
    U = cirq.testing.random_unitary(2)
    d = np.diag(np.exp([-0.2312j, -0.2312j]))
    return U @ d @ U.conj().T


@pytest.mark.parametrize('mat', [
    np.eye(2),
    cirq.unitary(cirq.H),
    cirq.unitary(cirq.X),
    cirq.unitary(cirq.X**0.5),
    cirq.unitary(cirq.Y),
    cirq.unitary(cirq.Z),
    cirq.unitary(cirq.Z**0.5),
    _random_unitary_with_close_eigenvalues(),
] + [cirq.testing.random_unitary(2) for _ in range(10)])
def test_single_qubit_op_to_framed_phase_form_equivalent_on_known_and_random(
        mat):
    u, t, g = cirq.single_qubit_op_to_framed_phase_form(mat)
    z = np.diag([g, g * t])
    assert np.allclose(mat, np.conj(u.T).dot(z).dot(u))

def test_single_qubit_matrix_to_phased_x_z_known():
    actual = cirq.single_qubit_matrix_to_phased_x_z(
        np.array([[0, 1], [1, 0]]), atol=0.01)
    assert cirq.approx_eq(actual, [cirq.PhasedXPowGate(phase_exponent=1.0)],
                          atol=1e-9)

    actual = cirq.single_qubit_matrix_to_phased_x_z(
        np.array([[0, -1j], [1j, 0]]), atol=0.01)
    print(repr(actual))
    assert cirq.approx_eq(
        actual, [cirq.PhasedXPowGate(phase_exponent=0.5, exponent=-1)],
        atol=1e-9)

    actual = cirq.single_qubit_matrix_to_phased_x_z(
        np.array([[1, 0], [0, -1]]), atol=0.01)
    assert cirq.approx_eq(actual, [cirq.Z], atol=1e-9)

    actual = cirq.single_qubit_matrix_to_phased_x_z(
        np.array([[1, 0], [0, 1j]]), atol=0.01)
    assert cirq.approx_eq(actual, [cirq.Z**0.5], atol=1e-9)

    actual = cirq.single_qubit_matrix_to_phased_x_z(
        np.array([[1, 0], [0, -1j]]), atol=0.01)
    assert cirq.approx_eq(actual, [cirq.Z**-0.5], atol=1e-9)

    actual = cirq.single_qubit_matrix_to_phased_x_z(
        np.array([[1, 1], [1, -1]]) * np.sqrt(0.5), atol=0.001)
    assert cirq.approx_eq(
        actual,
        [cirq.PhasedXPowGate(phase_exponent=-0.5, exponent=0.5), cirq.Z**-1],
        atol=1e-9)


@pytest.mark.parametrize('intended_effect', [
    np.array([[0, 1j], [1, 0]]),
] + [
    cirq.testing.random_unitary(2) for _ in range(10)
])
def test_single_qubit_matrix_to_phased_x_z_cases(intended_effect):
    gates = cirq.single_qubit_matrix_to_phased_x_z(intended_effect, atol=1e-6)
    assert len(gates) <= 2
    assert_gates_implement_unitary(gates, intended_effect, atol=1e-5)


@pytest.mark.parametrize('pre_turns,post_turns',
                         [(random.random(), random.random())
                          for _ in range(10)])
def test_single_qubit_matrix_to_phased_x_z_fuzz_half_turns_always_one_gate(
        pre_turns, post_turns):
    atol = 1e-6
    aggr_atol = atol * 10.0

    intended_effect = cirq.dot(
        cirq.unitary(cirq.Z**(2 * pre_turns)),
        cirq.unitary(cirq.X),
        cirq.unitary(cirq.Z**(2 * post_turns)))

    gates = cirq.single_qubit_matrix_to_phased_x_z(
        intended_effect, atol=atol)

    assert len(gates) == 1
    assert_gates_implement_unitary(gates, intended_effect, atol=aggr_atol)


def test_single_qubit_matrix_to_phased_x_z_tolerance_z():
    z = np.diag([1, np.exp(1j * 0.01)])

    optimized_away = cirq.single_qubit_matrix_to_phased_x_z(
        z, atol=0.1)
    assert len(optimized_away) == 0

    kept = cirq.single_qubit_matrix_to_phased_x_z(z, atol=0.0001)
    assert len(kept) == 1


def test_single_qubit_matrix_to_phased_x_z_tolerance_xy():
    c, s = np.cos(0.01), np.sin(0.01)
    xy = np.array([[c, -s], [s, c]])

    optimized_away = cirq.single_qubit_matrix_to_phased_x_z(
        xy, atol=0.1)
    assert len(optimized_away) == 0

    kept = cirq.single_qubit_matrix_to_phased_x_z(xy, atol=0.0001)
    assert len(kept) == 1


def test_single_qubit_matrix_to_phased_x_z_tolerance_half_turn_phasing():
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


@pytest.mark.parametrize('intended_effect', [
    np.array([[0, 1], [1, 0]]),
    np.array([[0, -1j], [1j, 0]]),
    np.array([[1, 0], [0, -1]]),
    np.array([[1, 0], [0, 1j]]),
    np.array([[1, 0], [0, -1j]]),
    np.array([[1, 1], [1, -1]]) * np.sqrt(0.5),
    np.array([[0, 1j], [1, 0]]),
    *[cirq.testing.random_unitary(2) for _ in range(10)],
])
def test_single_qubit_matrix_to_phxz_cases(intended_effect):
    gate = cirq.single_qubit_matrix_to_phxz(intended_effect, atol=1e-6)
    assert_gates_implement_unitary([gate], intended_effect, atol=1e-5)


@pytest.mark.parametrize('pre_turns,post_turns',
                         [(random.random(), random.random()) for _ in range(10)]
                        )
def test_single_qubit_matrix_to_phxz_fuzz_half_turns_always_one_gate(
        pre_turns, post_turns):
    atol = 1e-6
    aggr_atol = atol * 10.0

    intended_effect = cirq.dot(cirq.unitary(cirq.Z**(2 * pre_turns)),
                               cirq.unitary(cirq.X),
                               cirq.unitary(cirq.Z**(2 * post_turns)))

    gate = cirq.single_qubit_matrix_to_phxz(intended_effect, atol=atol)

    assert gate.z_exponent == 0
    assert_gates_implement_unitary([gate], intended_effect, atol=aggr_atol)


def test_single_qubit_matrix_to_phxz_tolerance_z():
    z = np.diag([1, np.exp(1j * 0.01)])

    optimized_away = cirq.single_qubit_matrix_to_phxz(z, atol=0.1)
    assert optimized_away is None

    kept = cirq.single_qubit_matrix_to_phxz(z, atol=0.0001)
    assert kept is not None


def test_single_qubit_matrix_to_phxz_tolerance_xy():
    c, s = np.cos(0.01), np.sin(0.01)
    xy = np.array([[c, -s], [s, c]])

    optimized_away = cirq.single_qubit_matrix_to_phxz(xy, atol=0.1)
    assert optimized_away is None

    kept = cirq.single_qubit_matrix_to_phxz(xy, atol=0.0001)
    assert kept is not None


def test_single_qubit_matrix_to_phxz_tolerance_half_turn_phasing():
    a = np.pi / 2 + 0.01
    c, s = np.cos(a), np.sin(a)
    nearly_x = np.array([[c, -s], [s, c]])
    z1 = np.diag([1, np.exp(1j * 1.2)])
    z2 = np.diag([1, np.exp(1j * 1.6)])
    phased_nearly_x = z1.dot(nearly_x).dot(z2)

    optimized_away = cirq.single_qubit_matrix_to_phxz(phased_nearly_x, atol=0.1)
    assert optimized_away.z_exponent == 0

    kept = cirq.single_qubit_matrix_to_phxz(phased_nearly_x, atol=0.0001)
    assert kept.z_exponent != 0
