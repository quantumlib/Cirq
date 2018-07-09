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

import numpy as np

from cirq import linalg
from cirq.ops.matrix_gates import SingleQubitMatrixGate, TwoQubitMatrixGate
from cirq.testing import EqualsTester

H = np.array([[1, 1], [1, -1]]) * np.sqrt(0.5)
HH = linalg.kron(H, H)
QFT2 = np.array([[1, 1, 1, 1],
                 [1, 1j, -1, -1j],
                 [1, -1, 1, -1],
                 [1, -1j, -1, 1j]]) * 0.5


def test_single_qubit_init():
    m = np.array([[1, 1j], [1j, 1]]) * np.sqrt(0.5)
    x2 = SingleQubitMatrixGate(m)
    assert np.alltrue(x2.matrix() == m)


def test_single_qubit_eq():
    eq = EqualsTester()
    eq.make_equality_group(lambda: SingleQubitMatrixGate(np.eye(2)))
    eq.make_equality_group(
        lambda: SingleQubitMatrixGate(np.array([[0, 1], [1, 0]])))
    x2 = np.array([[1, 1j], [1j, 1]]) * np.sqrt(0.5)
    eq.make_equality_group(lambda: SingleQubitMatrixGate(x2))


def test_single_qubit_phase_by():
    x = SingleQubitMatrixGate(np.array([[0, 1], [1, 0]]))
    y = SingleQubitMatrixGate(np.array([[0, -1j], [1j, 0]]))
    z = SingleQubitMatrixGate(np.array([[1, 0], [0, -1]]))
    assert x.phase_by(0.25, 0).approx_eq(y)
    assert y.phase_by(-0.25, 0).approx_eq(x)
    assert z.phase_by(0.25, 0).approx_eq(z)


def test_single_qubit_trace_distance_bound():
    x = SingleQubitMatrixGate(np.array([[0, 1], [1, 0]]))
    x2 = SingleQubitMatrixGate(
        np.array([[1, 1j], [1j, 1]]) * np.sqrt(0.5))
    assert x.trace_distance_bound() >= 1
    assert x2.trace_distance_bound() >= 0.5


def test_single_qubit_approx_eq():
    x = SingleQubitMatrixGate(np.array([[0, 1], [1, 0]]))
    i = SingleQubitMatrixGate(np.array([[1, 0], [0, 1]]))
    i_ish = SingleQubitMatrixGate(
        np.array([[1, 0.000000000000001], [0, 1]]))
    assert i.approx_eq(i_ish)
    assert i.approx_eq(i)
    assert not i.approx_eq(x)
    assert i.approx_eq('') is NotImplemented


def test_single_qubit_extrapolate():
    i = SingleQubitMatrixGate(np.eye(2))
    x = SingleQubitMatrixGate(np.array([[0, 1], [1, 0]]))
    x2 = SingleQubitMatrixGate(
        np.array([[1, 1j], [1j, 1]]) * (1 - 1j) / 2)
    x2i = SingleQubitMatrixGate(np.conj(x2.matrix().T))

    assert x.extrapolate_effect(0).approx_eq(i)
    assert x2.extrapolate_effect(0).approx_eq(i)
    assert x2.extrapolate_effect(2).approx_eq(x)
    assert x2.extrapolate_effect(-1).approx_eq(x2i)
    assert x2.extrapolate_effect(3).approx_eq(x2i)
    assert x.extrapolate_effect(-1).approx_eq(x)

    z2 = SingleQubitMatrixGate(np.array([[1, 0], [0, 1j]]))
    z4 = SingleQubitMatrixGate(
        np.array([[1, 0], [0, (1 + 1j) * np.sqrt(0.5)]]))
    assert z2.extrapolate_effect(0.5).approx_eq(z4)


def test_two_qubit_init():
    x2 = TwoQubitMatrixGate(QFT2)
    assert np.alltrue(x2.matrix() == QFT2)


def test_two_qubit_eq():
    eq = EqualsTester()
    eq.make_equality_group(lambda: TwoQubitMatrixGate(np.eye(4)))
    eq.make_equality_group(lambda: TwoQubitMatrixGate(QFT2))
    eq.make_equality_group(lambda: TwoQubitMatrixGate(HH))


def test_two_qubit_phase_by():
    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    z = np.array([[1, 0], [0, -1]])

    xx = TwoQubitMatrixGate(np.kron(x, x))
    yx = TwoQubitMatrixGate(np.kron(x, y))
    xy = TwoQubitMatrixGate(np.kron(y, x))
    yy = TwoQubitMatrixGate(np.kron(y, y))
    assert xx.phase_by(0.25, 0).approx_eq(yx)
    assert xx.phase_by(0.25, 1).approx_eq(xy)
    assert xy.phase_by(0.25, 0).approx_eq(yy)
    assert xy.phase_by(-0.25, 1).approx_eq(xx)

    zz = TwoQubitMatrixGate(np.kron(z, z))
    assert zz.phase_by(0.25, 0).approx_eq(zz)
    assert zz.phase_by(0.25, 1).approx_eq(zz)


def test_two_qubit_approx_eq():
    f = TwoQubitMatrixGate(QFT2)
    perturb = np.zeros(shape=QFT2.shape, dtype=np.float64)
    perturb[1, 2] = 0.00000001
    assert f.approx_eq(TwoQubitMatrixGate(QFT2))
    assert f.approx_eq(TwoQubitMatrixGate(QFT2 + perturb))
    assert not f.approx_eq(TwoQubitMatrixGate(HH))


def test_two_qubit_extrapolate():
    cz2 = TwoQubitMatrixGate(np.diag([1, 1, 1, 1j]))
    cz4 = TwoQubitMatrixGate(np.diag([1, 1, 1, (1 + 1j) * np.sqrt(0.5)]))
    i = TwoQubitMatrixGate(np.eye(4))

    assert cz2.extrapolate_effect(0).approx_eq(i)
    assert cz4.extrapolate_effect(0).approx_eq(i)
    assert cz2.extrapolate_effect(0.5).approx_eq(cz4)
