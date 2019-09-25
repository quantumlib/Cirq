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

import itertools

import numpy as np
import pytest
import scipy
import sympy

import cirq

np.set_printoptions(linewidth=300)


def test_phased_iswap_init():
    p = -0.25
    t = 0.75
    g = 0.1
    gate = cirq.PhasedISwapPowGate(phase_exponent=p, exponent=t, global_shift=g)
    assert gate.phase_exponent == p
    assert gate.exponent == t
    assert gate.global_shift == g


def test_phased_iswap_equality():
    assert (cirq.PhasedISwapPowGate(phase_exponent=0,
                                    exponent=0.4) == cirq.ISWAP**0.4)
    assert (cirq.PhasedISwapPowGate(phase_exponent=0,
                                    exponent=0.4,
                                    global_shift=0.1) == cirq.ISwapPowGate(
                                        exponent=0.4, global_shift=0.1))


def test_phased_iswap_unitary():
    p = 0.3
    t = 0.4
    actual = cirq.unitary(cirq.PhasedISwapPowGate(phase_exponent=p, exponent=t))
    c = np.cos(np.pi * t / 2)
    s = np.sin(np.pi * t / 2) * 1j
    f = np.exp(2j * np.pi * p)
    # yapf: disable
    expected = np.array([[1, 0, 0, 0],
                         [0, c, s * f, 0],
                         [0, s * f.conjugate(), c, 0],
                         [0, 0, 0, 1]])
    # yapf: enable
    assert np.allclose(actual, expected)


def test_phased_iswap_equivalent_circuit():
    p = 0.7
    t = -0.4
    gate = cirq.PhasedISwapPowGate(phase_exponent=p, exponent=t)
    q0, q1 = cirq.LineQubit.range(2)
    equivalent_circuit = cirq.Circuit.from_ops([
        cirq.Z(q0)**p,
        cirq.Z(q1)**-p,
        cirq.ISWAP(q0, q1)**t,
        cirq.Z(q0)**-p,
        cirq.Z(q1)**p,
    ])
    assert np.allclose(cirq.unitary(gate), cirq.unitary(equivalent_circuit))


def test_phased_iswap_str():
    assert str(cirq.PhasedISwapPowGate(exponent=1)) == 'PhasedISWAP'
    assert str(cirq.PhasedISwapPowGate(exponent=0.5)) == 'PhasedISWAP**0.5'


def test_phased_iswap_pow():
    gate1 = cirq.PhasedISwapPowGate(phase_exponent=0.1, exponent=0.25)
    gate2 = cirq.PhasedISwapPowGate(phase_exponent=0.1, exponent=0.5)
    assert gate1**2 == gate2

    u1 = cirq.unitary(gate1)
    u2 = cirq.unitary(gate2)
    assert np.allclose(u1 @ u1, u2)


def test_decompose_invalid_qubits():
    qs = cirq.LineQubit.range(3)
    with pytest.raises(ValueError):
        cirq.protocols.decompose_once_with_qubits(cirq.PhasedISwapPowGate(), qs)


@pytest.mark.parametrize('phase_exponent, exponent',
                         itertools.product(
                             (-0.3, 0, 0.1, 0.5, 1, 2, sympy.Symbol('p')),
                             (-0.1, 0, 0.1, 1, sympy.Symbol('t')),
                         ))
def test_phased_iswap_has_consistent_protocols(phase_exponent, exponent):
    cirq.testing.assert_implements_consistent_protocols(
        cirq.PhasedISwapPowGate(phase_exponent=phase_exponent,
                                exponent=exponent),
        ignoring_global_phase=False)


@pytest.mark.parametrize('phase_exponent, exponent, global_shift',
                         itertools.product(
                             (0.2, 1),
                             (0.2, 1),
                             (-0.1, 0, 0.1),
                         ))
def test_phased_iswap_has_consistent_protocols_up_to_global_phase(
        phase_exponent, exponent, global_shift):
    cirq.testing.assert_implements_consistent_protocols(
        cirq.PhasedISwapPowGate(phase_exponent=phase_exponent,
                                exponent=exponent,
                                global_shift=global_shift),
        ignoring_global_phase=True)


@pytest.mark.parametrize('exponent', (-0.5, 0.1, 1.2))
def test_givens_rotation_unitary(exponent):
    actual = cirq.unitary(cirq.GivensRotation**exponent)
    c = np.cos(np.pi * exponent / 2)
    s = np.sin(np.pi * exponent / 2)
    # yapf: disable
    expected = np.array([[1, 0, 0, 0],
                         [0, c, -s, 0],
                         [0, s, c, 0],
                         [0, 0, 0, 1]])
    # yapf: enable
    assert np.allclose(actual, expected)


@pytest.mark.parametrize('exponent', (-1, 0.2, 1))
def test_givens_rotation_hamiltonian(exponent):
    actual = cirq.unitary(cirq.GivensRotation**exponent)
    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    yx = np.kron(y, x)
    xy = np.kron(x, y)
    expected = scipy.linalg.expm(-0.25j * np.pi * exponent * (yx - xy))
    assert np.allclose(actual, expected)


def test_givens_rotation_equivalent_circuit():
    t = 0.123
    gate = cirq.GivensRotation**t
    q0, q1 = cirq.LineQubit.range(2)
    equivalent_circuit = cirq.Circuit.from_ops([
        cirq.T(q0),
        cirq.T(q1)**-1,
        cirq.ISWAP(q0, q1)**t,
        cirq.T(q0)**-1,
        cirq.T(q1),
    ])
    assert np.allclose(cirq.unitary(gate), cirq.unitary(equivalent_circuit))


@pytest.mark.parametrize('exponent', (-0.3, 0.7))
def test_givens_rotation_has_consistent_protocols(exponent):
    cirq.testing.assert_implements_consistent_protocols(
        cirq.GivensRotation**exponent, ignoring_global_phase=False)
