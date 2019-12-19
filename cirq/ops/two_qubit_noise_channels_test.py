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
import pytest
import sympy

import cirq

from cirq.ops.two_qubit_noise_channels import *


@pytest.mark.parametrize('eigen_gate_type', [
    XXGate,
    XYGate,
    XZGate,
    YXGate,
    YYGate,
    YZGate,
    ZXGate,
    ZYGate,
    ZZGate,
    IXGate,
    IYGate,
    IZGate,
])
def test_phase_insensitive_eigen_gates_consistent_protocols(eigen_gate_type):
    cirq.testing.assert_eigengate_implements_consistent_protocols(
        eigen_gate_type)


def test_xi_init():
    assert XIGate(exponent=0.5).exponent == 0.5
    assert XIGate(exponent=5).exponent == 5
    assert (XIGate()**0.5).exponent == 0.5


def test_xi_str():
    assert str(XIGate()) == 'XI'
    assert str(XIGate()**0.5) == 'XI**0.5'
    assert str(XIGate()**-0.25) == 'XI**-0.25'


def test_xi_repr():
    assert repr(XIGate()) == 'cirq.XI'
    assert repr(XIGate()**0.5) == '(cirq.XI**0.5)'
    assert repr(XIGate()**-0.25) == '(cirq.XI**-0.25)'


def test_xi_unitary():
    assert np.allclose(
        cirq.unitary(XIGate()),
        np.array([[0., 0., 1., 0.], [0., 0., 0., 1.], [1., 0., 0., 0.],
                  [0., 1., 0., 0.]]))

    assert np.allclose(
        cirq.unitary(XIGate()**0.5),
        np.array([[0.5 + 0.5j, 0, 0.5 - 0.5j,
                   0], [0, 0.5 + 0.5j, 0, 0.5 - 0.5j],
                  [0.5 - 0.5j, 0, 0.5 + 0.5j, 0],
                  [0, 0.5 - 0.5j, 0, 0.5 + 0.5j]]))

    assert np.allclose(
        cirq.unitary(XIGate()**0),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    assert np.allclose(
        cirq.unitary(XIGate()**-0.5),
        np.array([[0.5 - 0.5j, 0, 0.5 + 0.5j,
                   0], [0, 0.5 - 0.5j, 0, 0.5 + 0.5j],
                  [0.5 + 0.5j, 0, 0.5 - 0.5j, 0],
                  [0, 0.5 + 0.5j, 0, 0.5 - 0.5j]]))


def test_xx_init():
    assert XXGate(exponent=0.5).exponent == 0.5
    assert XXGate(exponent=5).exponent == 5
    assert (XXGate()**0.5).exponent == 0.5


def test_xx_str():
    assert str(XXGate()) == 'XX'
    assert str(XXGate()**0.5) == 'XX**0.5'
    assert str(XXGate()**-0.25) == 'XX**-0.25'


def test_xx_repr():
    assert repr(XXGate()) == 'cirq.XX'
    assert repr(XXGate()**0.5) == '(cirq.XX**0.5)'
    assert repr(XXGate()**-0.25) == '(cirq.XX**-0.25)'


def test_xx_unitary():
    assert np.allclose(
        cirq.unitary(XXGate()),
        np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]))

    assert np.allclose(
        cirq.unitary(XXGate()**0.5),
        np.array([[0.5 + 1j * 0.5, 0, 0, 0.5 + 1j * -0.5],
                  [0, 0.5 + 1j * 0.5, 0.5 + 1j * -0.5, 0],
                  [0, 0.5 + 1j * -0.5, 0.5 + 1j * 0.5, 0],
                  [0.5 + 1j * -0.5, 0, 0, 0.5 + 1j * 0.5]]))

    assert np.allclose(
        cirq.unitary(XXGate()**0),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    assert np.allclose(
        cirq.unitary(XXGate()**-0.5),
        np.array([[0.5 + 1j * -0.5, 0, 0, 0.5 + 1j * 0.5],
                  [0, 0.5 + 1j * -0.5, 0.5 + 1j * 0.5, 0],
                  [0, 0.5 + 1j * 0.5, 0.5 + 1j * -0.5, 0],
                  [0.5 + 1j * 0.5, 0, 0, 0.5 + 1j * -0.5]]))


def test_xy_init():
    assert XYGate(exponent=0.5).exponent == 0.5
    assert XYGate(exponent=5).exponent == 5
    assert (XYGate()**0.5).exponent == 0.5


def test_xy_str():
    assert str(XYGate()) == 'XY'
    assert str(XYGate()**0.5) == 'XY**0.5'
    assert str(XYGate()**-0.25) == 'XY**-0.25'


def test_xy_repr():
    assert repr(XYGate()) == 'cirq.XY'
    assert repr(XYGate()**0.5) == '(cirq.XY**0.5)'
    assert repr(XYGate()**-0.25) == '(cirq.XY**-0.25)'


def test_xy_unitary():
    assert np.allclose(
        cirq.unitary(XYGate()),
        np.array([[0. + 0.j, 0. + 0.j, 0. + 0.j, 0. - 1.j],
                  [0. + 0.j, 0. + 0.j, 0. + 1.j, 0. + 0.j],
                  [0. + 0.j, 0. - 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 1.j, 0. + 0.j, 0. + 0.j, 0. + 0.j]]))

    assert np.allclose(
        cirq.unitary(XYGate()**0.5),
        np.array([[0.5 + 0.5j, 0. + 0.j, 0. + 0.j, -0.5 - 0.5j],
                  [0. + 0.j, 0.5 + 0.5j, 0.5 + 0.5j, 0. + 0.j],
                  [0. + 0.j, -0.5 - 0.5j, 0.5 + 0.5j, 0. + 0.j],
                  [0.5 + 0.5j, 0. + 0.j, 0. + 0.j, 0.5 + 0.5j]]))

    assert np.allclose(
        cirq.unitary(XYGate()**0),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    assert np.allclose(
        cirq.unitary(XYGate()**-0.5),
        np.array([[0.5 - 0.5j, 0. + 0.j, 0. + 0.j, 0.5 - 0.5j],
                  [0. + 0.j, 0.5 - 0.5j, -0.5 + 0.5j, 0. + 0.j],
                  [0. + 0.j, 0.5 - 0.5j, 0.5 - 0.5j, 0. + 0.j],
                  [-0.5 + 0.5j, 0. + 0.j, 0. + 0.j, 0.5 - 0.5j]]))


def test_xz_init():
    assert XZGate(exponent=0.5).exponent == 0.5
    assert XZGate(exponent=5).exponent == 5
    assert (XZGate()**0.5).exponent == 0.5


def test_xz_str():
    assert str(XZGate()) == 'XZ'
    assert str(XZGate()**0.5) == 'XZ**0.5'
    assert str(XZGate()**-0.25) == 'XZ**-0.25'


def test_xz_repr():
    assert repr(XZGate()) == 'cirq.XZ'
    assert repr(XZGate()**0.5) == '(cirq.XZ**0.5)'
    assert repr(XZGate()**-0.25) == '(cirq.XZ**-0.25)'


def test_xz_unitary():
    assert np.allclose(
        cirq.unitary(XZGate()),
        np.array([[0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, -1. + 0.j],
                  [1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, -1. + 0.j, 0. + 0.j, 0. + 0.j]]))

    assert np.allclose(
        cirq.unitary(XZGate()**0.5),
        np.array([[0.5 + 0.5j, 0. + 0.j, 0.5 - 0.5j, 0. + 0.j],
                  [0. + 0.j, 0.5 + 0.5j, 0. + 0.j, -0.5 + 0.5j],
                  [0.5 - 0.5j, 0. + 0.j, 0.5 + 0.5j, 0. + 0.j],
                  [0. + 0.j, -0.5 + 0.5j, 0. + 0.j, 0.5 + 0.5j]]))

    assert np.allclose(
        cirq.unitary(XZGate()**0),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    assert np.allclose(
        cirq.unitary(XZGate()**-0.5),
        np.array([[0.5 - 0.5j, 0. + 0.j, 0.5 + 0.5j, 0. + 0.j],
                  [0. + 0.j, 0.5 - 0.5j, 0. + 0.j, -0.5 - 0.5j],
                  [0.5 + 0.5j, 0. + 0.j, 0.5 - 0.5j, 0. + 0.j],
                  [0. + 0.j, -0.5 - 0.5j, 0. + 0.j, 0.5 - 0.5j]]))


def test_yi_init():
    assert YIGate(exponent=0.5).exponent == 0.5
    assert YIGate(exponent=5).exponent == 5
    assert (YIGate()**0.5).exponent == 0.5


def test_yi_str():
    assert str(YIGate()) == 'YI'
    assert str(YIGate()**0.5) == 'YI**0.5'
    assert str(YIGate()**-0.25) == 'YI**-0.25'


def test_yi_repr():
    assert repr(YIGate()) == 'cirq.YI'
    assert repr(YIGate()**0.5) == '(cirq.YI**0.5)'
    assert repr(YIGate()**-0.25) == '(cirq.YI**-0.25)'


def test_yi_unitary():
    assert np.allclose(
        cirq.unitary(YIGate()),
        np.array([[0. + 0.j, 0. + 0.j, 0. - 1.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. - 1.j],
                  [0. + 1.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 1.j, 0. + 0.j, 0. + 0.j]]))

    assert np.allclose(
        cirq.unitary(YIGate()**0.5),
        np.array([[0.5 + 0.5j, 0. + 0.j, -0.5 - 0.5j, 0. + 0.j],
                  [0. + 0.j, 0.5 + 0.5j, 0. + 0.j, -0.5 - 0.5j],
                  [0.5 + 0.5j, 0. + 0.j, 0.5 + 0.5j, 0. + 0.j],
                  [0. + 0.j, 0.5 + 0.5j, 0. + 0.j, 0.5 + 0.5j]]))

    assert np.allclose(
        cirq.unitary(YIGate()**0),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    assert np.allclose(
        cirq.unitary(YIGate()**-0.5),
        np.array([[0.5 - 0.5j, 0. + 0.j, 0.5 - 0.5j, 0. + 0.j],
                  [0. + 0.j, 0.5 - 0.5j, 0. + 0.j, 0.5 - 0.5j],
                  [-0.5 + 0.5j, 0. + 0.j, 0.5 - 0.5j, 0. + 0.j],
                  [0. + 0.j, -0.5 + 0.5j, 0. + 0.j, 0.5 - 0.5j]]))


def test_yx_init():
    assert YXGate(exponent=0.5).exponent == 0.5
    assert YXGate(exponent=5).exponent == 5
    assert (YXGate()**0.5).exponent == 0.5


def test_yx_str():
    assert str(YXGate()) == 'YX'
    assert str(YXGate()**0.5) == 'YX**0.5'
    assert str(YXGate()**-0.25) == 'YX**-0.25'


def test_yx_repr():
    assert repr(YXGate()) == 'cirq.YX'
    assert repr(YXGate()**0.5) == '(cirq.YX**0.5)'
    assert repr(YXGate()**-0.25) == '(cirq.YX**-0.25)'


def test_yx_unitary():
    assert np.allclose(
        cirq.unitary(YXGate()),
        np.array([[0. + 0.j, 0. + 0.j, 0. + 0.j, 0. - 1.j],
                  [0. + 0.j, 0. + 0.j, 0. - 1.j, 0. + 0.j],
                  [0. + 0.j, 0. + 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 1.j, 0. + 0.j, 0. + 0.j, 0. + 0.j]]))

    assert np.allclose(
        cirq.unitary(YXGate()**0.5),
        np.array([[0.5 + 0.5j, 0. + 0.j, 0. + 0.j, -0.5 - 0.5j],
                  [0. + 0.j, 0.5 + 0.5j, -0.5 - 0.5j, 0. + 0.j],
                  [0. + 0.j, 0.5 + 0.5j, 0.5 + 0.5j, 0. + 0.j],
                  [0.5 + 0.5j, 0. + 0.j, 0. + 0.j, 0.5 + 0.5j]]))

    assert np.allclose(
        cirq.unitary(YXGate()**0),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    assert np.allclose(
        cirq.unitary(YXGate()**-0.5),
        np.array([[0.5 - 0.5j, 0. + 0.j, 0. + 0.j, 0.5 - 0.5j],
                  [0. + 0.j, 0.5 - 0.5j, 0.5 - 0.5j, 0. + 0.j],
                  [0. + 0.j, -0.5 + 0.5j, 0.5 - 0.5j, 0. + 0.j],
                  [-0.5 + 0.5j, 0. + 0.j, 0. + 0.j, 0.5 - 0.5j]]))


def test_yy_init():
    assert YYGate(exponent=0.5).exponent == 0.5
    assert YYGate(exponent=5).exponent == 5
    assert (YYGate()**0.5).exponent == 0.5


def test_yy_str():
    assert str(YYGate()) == 'YY'
    assert str(YYGate()**0.5) == 'YY**0.5'
    assert str(YYGate()**-0.25) == 'YY**-0.25'


def test_yy_repr():
    assert repr(YYGate()) == 'cirq.YY'
    assert repr(YYGate()**0.5) == '(cirq.YY**0.5)'
    assert repr(YYGate()**-0.25) == '(cirq.YY**-0.25)'


def test_yy_unitary():
    assert np.allclose(
        cirq.unitary(YYGate()),
        np.array([[0. + 0.j, 0. + 0.j, 0. + 0.j, -1. + 0.j],
                  [0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j],
                  [0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j],
                  [-1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j]]))

    assert np.allclose(
        cirq.unitary(YYGate()**0.5),
        np.array([[0.5 + 0.5j, 0. + 0.j, 0. + 0.j, -0.5 + 0.5j],
                  [0. + 0.j, 0.5 + 0.5j, 0.5 - 0.5j, 0. + 0.j],
                  [0. + 0.j, 0.5 - 0.5j, 0.5 + 0.5j, 0. + 0.j],
                  [-0.5 + 0.5j, 0. + 0.j, 0. + 0.j, 0.5 + 0.5j]]))

    assert np.allclose(
        cirq.unitary(YYGate()**0),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    assert np.allclose(
        cirq.unitary(YYGate()**-0.5),
        np.array([[0.5 - 0.5j, 0. + 0.j, 0. + 0.j, -0.5 - 0.5j],
                  [0. + 0.j, 0.5 - 0.5j, 0.5 + 0.5j, 0. + 0.j],
                  [0. + 0.j, 0.5 + 0.5j, 0.5 - 0.5j, 0. + 0.j],
                  [-0.5 - 0.5j, 0. + 0.j, 0. + 0.j, 0.5 - 0.5j]]))


def test_yz_init():
    assert YZGate(exponent=0.5).exponent == 0.5
    assert YZGate(exponent=5).exponent == 5
    assert (YZGate()**0.5).exponent == 0.5


def test_yz_str():
    assert str(YZGate()) == 'YZ'
    assert str(YZGate()**0.5) == 'YZ**0.5'
    assert str(YZGate()**-0.25) == 'YZ**-0.25'


def test_yz_repr():
    assert repr(YZGate()) == 'cirq.YZ'
    assert repr(YZGate()**0.5) == '(cirq.YZ**0.5)'
    assert repr(YZGate()**-0.25) == '(cirq.YZ**-0.25)'


def test_yz_unitary():
    assert np.allclose(
        cirq.unitary(YZGate()),
        np.array([[0. + 0.j, 0. + 0.j, 0. - 1.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 1.j],
                  [0. + 1.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. - 1.j, 0. + 0.j, 0. + 0.j]]))

    assert np.allclose(
        cirq.unitary(YZGate()**0.5),
        np.array([[0.5 + 0.5j, 0. + 0.j, -0.5 - 0.5j, 0. + 0.j],
                  [0. + 0.j, 0.5 + 0.5j, 0. + 0.j, 0.5 + 0.5j],
                  [0.5 + 0.5j, 0. + 0.j, 0.5 + 0.5j, 0. + 0.j],
                  [0. + 0.j, -0.5 - 0.5j, 0. + 0.j, 0.5 + 0.5j]]))

    assert np.allclose(
        cirq.unitary(YZGate()**0),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    assert np.allclose(
        cirq.unitary(YZGate()**-0.5),
        np.array([[0.5 - 0.5j, 0. + 0.j, 0.5 - 0.5j, 0. + 0.j],
                  [0. + 0.j, 0.5 - 0.5j, 0. + 0.j, -0.5 + 0.5j],
                  [-0.5 + 0.5j, 0. + 0.j, 0.5 - 0.5j, 0. + 0.j],
                  [0. + 0.j, 0.5 - 0.5j, 0. + 0.j, 0.5 - 0.5j]]))


def test_zi_init():
    assert ZIGate(exponent=0.5).exponent == 0.5
    assert ZIGate(exponent=5).exponent == 5
    assert (ZIGate()**0.5).exponent == 0.5


def test_zi_str():
    assert str(ZIGate()) == 'ZI'
    assert str(ZIGate()**0.5) == 'ZI**0.5'
    assert str(ZIGate()**-0.25) == 'ZI**-0.25'


def test_zi_repr():
    assert repr(ZIGate()) == 'cirq.ZI'
    assert repr(ZIGate()**0.5) == '(cirq.ZI**0.5)'
    assert repr(ZIGate()**-0.25) == '(cirq.ZI**-0.25)'


def test_zi_unitary():
    assert np.allclose(
        cirq.unitary(ZIGate()),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, -1. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, -1. + 0.j]]))

    assert np.allclose(
        cirq.unitary(ZIGate()**0.5),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 1.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 1.j]]))

    assert np.allclose(
        cirq.unitary(ZIGate()**0),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    assert np.allclose(
        cirq.unitary(ZIGate()**-0.5),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. - 1.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. - 1.j]]))


def test_zx_init():
    assert ZXGate(exponent=0.5).exponent == 0.5
    assert ZXGate(exponent=5).exponent == 5
    assert (ZXGate()**0.5).exponent == 0.5


def test_zx_str():
    assert str(ZXGate()) == 'ZX'
    assert str(ZXGate()**0.5) == 'ZX**0.5'
    assert str(ZXGate()**-0.25) == 'ZX**-0.25'


def test_zx_repr():
    assert repr(ZXGate()) == 'cirq.ZX'
    assert repr(ZXGate()**0.5) == '(cirq.ZX**0.5)'
    assert repr(ZXGate()**-0.25) == '(cirq.ZX**-0.25)'


def test_zx_unitary():
    assert np.allclose(
        cirq.unitary(ZXGate()),
        np.array([[0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j],
                  [1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, -1. + 0.j],
                  [0. + 0.j, 0. + 0.j, -1. + 0.j, 0. + 0.j]]))

    assert np.allclose(
        cirq.unitary(ZXGate()**0.5),
        np.array([[0.5 + 0.5j, 0.5 - 0.5j, 0. + 0.j, 0. + 0.j],
                  [0.5 - 0.5j, 0.5 + 0.5j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0.5 + 0.5j, -0.5 + 0.5j],
                  [0. + 0.j, 0. + 0.j, -0.5 + 0.5j, 0.5 + 0.5j]]))

    assert np.allclose(
        cirq.unitary(ZXGate()**0),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    assert np.allclose(
        cirq.unitary(ZXGate()**-0.5),
        np.array([[0.5 - 0.5j, 0.5 + 0.5j, 0. + 0.j, 0. + 0.j],
                  [0.5 + 0.5j, 0.5 - 0.5j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0.5 - 0.5j, -0.5 - 0.5j],
                  [0. + 0.j, 0. + 0.j, -0.5 - 0.5j, 0.5 - 0.5j]]))


def test_zy_init():
    assert ZYGate(exponent=0.5).exponent == 0.5
    assert ZYGate(exponent=5).exponent == 5
    assert (ZYGate()**0.5).exponent == 0.5


def test_zy_str():
    assert str(ZYGate()) == 'ZY'
    assert str(ZYGate()**0.5) == 'ZY**0.5'
    assert str(ZYGate()**-0.25) == 'ZY**-0.25'


def test_zy_repr():
    assert repr(ZYGate()) == 'cirq.ZY'
    assert repr(ZYGate()**0.5) == '(cirq.ZY**0.5)'
    assert repr(ZYGate()**-0.25) == '(cirq.ZY**-0.25)'


def test_zy_unitary():
    assert np.allclose(
        cirq.unitary(ZYGate()),
        np.array([[0. + 0.j, 0. - 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 1.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 1.j],
                  [0. + 0.j, 0. + 0.j, 0. - 1.j, 0. + 0.j]]))

    assert np.allclose(
        cirq.unitary(ZYGate()**0.5),
        np.array([[0.5 + 0.5j, -0.5 - 0.5j, 0. + 0.j, 0. + 0.j],
                  [0.5 + 0.5j, 0.5 + 0.5j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0.5 + 0.5j, 0.5 + 0.5j],
                  [0. + 0.j, 0. + 0.j, -0.5 - 0.5j, 0.5 + 0.5j]]))

    assert np.allclose(
        cirq.unitary(ZYGate()**0),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    assert np.allclose(
        cirq.unitary(ZYGate()**-0.5),
        np.array([[0.5 - 0.5j, 0.5 - 0.5j, 0. + 0.j, 0. + 0.j],
                  [-0.5 + 0.5j, 0.5 - 0.5j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0.5 - 0.5j, -0.5 + 0.5j],
                  [0. + 0.j, 0. + 0.j, 0.5 - 0.5j, 0.5 - 0.5j]]))


def test_zz_init():
    assert ZZGate(exponent=0.5).exponent == 0.5
    assert ZZGate(exponent=5).exponent == 5
    assert (ZZGate()**0.5).exponent == 0.5


def test_zz_str():
    assert str(ZZGate()) == 'ZZ'
    assert str(ZZGate()**0.5) == 'ZZ**0.5'
    assert str(ZZGate()**-0.25) == 'ZZ**-0.25'


def test_zz_repr():
    assert repr(ZZGate()) == 'cirq.ZZ'
    assert repr(ZZGate()**0.5) == '(cirq.ZZ**0.5)'
    assert repr(ZZGate()**-0.25) == '(cirq.ZZ**-0.25)'


def test_zz_unitary():
    assert np.allclose(
        cirq.unitary(ZZGate()),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, -1. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, -1. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j]]))

    assert np.allclose(
        cirq.unitary(ZZGate()**0.5),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 1.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j]]))

    assert np.allclose(
        cirq.unitary(ZZGate()**0),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    assert np.allclose(
        cirq.unitary(ZZGate()**-0.5),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. - 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. - 1.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j]]))


def test_ix_init():
    assert IXGate(exponent=0.5).exponent == 0.5
    assert IXGate(exponent=5).exponent == 5
    assert (IXGate()**0.5).exponent == 0.5


def test_ix_str():
    assert str(IXGate()) == 'IX'
    assert str(IXGate()**0.5) == 'IX**0.5'
    assert str(IXGate()**-0.25) == 'IX**-0.25'


def test_ix_repr():
    assert repr(IXGate()) == 'cirq.IX'
    assert repr(IXGate()**0.5) == '(cirq.IX**0.5)'
    assert repr(IXGate()**-0.25) == '(cirq.IX**-0.25)'


def test_ix_unitary():
    assert np.allclose(
        cirq.unitary(IXGate()),
        np.array([[0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j],
                  [1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j],
                  [0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j]]))

    assert np.allclose(
        cirq.unitary(IXGate()**0.5),
        np.array([[0.5 + 0.5j, 0.5 - 0.5j, 0. + 0.j, 0. + 0.j],
                  [0.5 - 0.5j, 0.5 + 0.5j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0.5 + 0.5j, 0.5 - 0.5j],
                  [0. + 0.j, 0. + 0.j, 0.5 - 0.5j, 0.5 + 0.5j]]))

    assert np.allclose(
        cirq.unitary(IXGate()**0),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    assert np.allclose(
        cirq.unitary(IXGate()**-0.5),
        np.array([[0.5 - 0.5j, 0.5 + 0.5j, 0. + 0.j, 0. + 0.j],
                  [0.5 + 0.5j, 0.5 - 0.5j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0.5 - 0.5j, 0.5 + 0.5j],
                  [0. + 0.j, 0. + 0.j, 0.5 + 0.5j, 0.5 - 0.5j]]))


def test_iy_init():
    assert IYGate(exponent=0.5).exponent == 0.5
    assert IYGate(exponent=5).exponent == 5
    assert (IYGate()**0.5).exponent == 0.5


def test_iy_str():
    assert str(IYGate()) == 'IY'
    assert str(IYGate()**0.5) == 'IY**0.5'
    assert str(IYGate()**-0.25) == 'IY**-0.25'


def test_iy_repr():
    assert repr(IYGate()) == 'cirq.IY'
    assert repr(IYGate()**0.5) == '(cirq.IY**0.5)'
    assert repr(IYGate()**-0.25) == '(cirq.IY**-0.25)'


def test_iy_unitary():
    assert np.allclose(
        cirq.unitary(IYGate()),
        np.array([[0. + 0.j, 0. - 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 1.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. - 1.j],
                  [0. + 0.j, 0. + 0.j, 0. + 1.j, 0. + 0.j]]))

    assert np.allclose(
        cirq.unitary(IYGate()**0.5),
        np.array([[0.5 + 0.5j, -0.5 - 0.5j, 0. + 0.j, 0. + 0.j],
                  [0.5 + 0.5j, 0.5 + 0.5j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0.5 + 0.5j, -0.5 - 0.5j],
                  [0. + 0.j, 0. + 0.j, 0.5 + 0.5j, 0.5 + 0.5j]]))

    assert np.allclose(
        cirq.unitary(IYGate()**0),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    assert np.allclose(
        cirq.unitary(IYGate()**-0.5),
        np.array([[0.5 - 0.5j, 0.5 - 0.5j, 0. + 0.j, 0. + 0.j],
                  [-0.5 + 0.5j, 0.5 - 0.5j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0.5 - 0.5j, 0.5 - 0.5j],
                  [0. + 0.j, 0. + 0.j, -0.5 + 0.5j, 0.5 - 0.5j]]))


def test_iz_init():
    assert IZGate(exponent=0.5).exponent == 0.5
    assert IZGate(exponent=5).exponent == 5
    assert (IZGate()**0.5).exponent == 0.5


def test_iz_str():
    assert str(IZGate()) == 'IZ'
    assert str(IZGate()**0.5) == 'IZ**0.5'
    assert str(IZGate()**-0.25) == 'IZ**-0.25'


def test_iz_repr():
    assert repr(IZGate()) == 'cirq.IZ'
    assert repr(IZGate()**0.5) == '(cirq.IZ**0.5)'
    assert repr(IZGate()**-0.25) == '(cirq.IZ**-0.25)'


def test_iz_unitary():
    assert np.allclose(
        cirq.unitary(IZGate()),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, -1. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, -1. + 0.j]]))

    assert np.allclose(
        cirq.unitary(IZGate()**0.5),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 1.j]]))

    assert np.allclose(
        cirq.unitary(IZGate()**0),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    assert np.allclose(
        cirq.unitary(IZGate()**-0.5),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. - 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. - 1.j]]))
