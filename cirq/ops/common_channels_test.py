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

import cirq

X = np.array([[0, 1], [1, 0]])
Y = np.array( [[0, -1j], [1j, 0]])
Z = np.array( [[1, 0], [0, -1]])


def test_asymmetric_depolarizing_channel():
    d = cirq.asymmetric_depolarize(0.1, 0.2, 0.3)
    np.testing.assert_almost_equal(cirq.channel(d),
                                   (np.sqrt(0.4) * np.eye(2),
                                    np.sqrt(0.1) * X,
                                    np.sqrt(0.2) * Y,
                                    np.sqrt(0.3) * Z))


def test_asymmetric_depolarizing_channel_repr():
    cirq.testing.assert_equivalent_repr(
        cirq.AsymmetricDepolarizingChannel(0.1, 0.2, 0.3))


def test_asymmetric_depolarizing_channel_str():
    assert (str(cirq.asymmetric_depolarize(0.1, 0.2, 0.3))
            == 'AsymmetricDepolarizingChannel(p_x=0.1,p_y=0.2,p_z=0.3)')


def test_asymmetric_depolarizing_channel_eq():
    et = cirq.testing.EqualsTester()
    et.add_equality_group(cirq.asymmetric_depolarize(0.0, 0.0, 0.0))
    et.add_equality_group(cirq.asymmetric_depolarize(0.0, 0.0, 0.1))
    et.add_equality_group(cirq.asymmetric_depolarize(0.0, 0.1, 0.0))
    et.add_equality_group(cirq.asymmetric_depolarize(0.1, 0.0, 0.0))
    et.add_equality_group(cirq.asymmetric_depolarize(0.1, 0.2, 0.3))
    et.add_equality_group(cirq.asymmetric_depolarize(0.3, 0.4, 0.3))
    et.add_equality_group(cirq.asymmetric_depolarize(1.0, 0.0, 0.0))
    et.add_equality_group(cirq.asymmetric_depolarize(0.0, 1.0, 0.0))
    et.add_equality_group(cirq.asymmetric_depolarize(0.0, 0.0, 1.0))


@pytest.mark.parametrize('p_x,p_y,p_z', (
    (-0.1, 0.0, 0.0),
    (0.0, -0.1, 0.0),
    (0.0, 0.0, -0.1),
    (0.1, -0.1, 0.1)))
def test_asymmetric_depolarizing_channel_negative_probability(p_x, p_y, p_z):
    with pytest.raises(ValueError, match='was less than 0'):
        cirq.asymmetric_depolarize(p_x, p_y, p_z)


@pytest.mark.parametrize('p_x,p_y,p_z', (
    (1.1, 0.0, 0.0),
    (0.0, 1.1, 0.0),
    (0.0, 0.0, 1.1),
    (0.1, 0.9, 0.1)))
def test_asymmetric_depolarizing_channel_bigly_probability(p_x, p_y, p_z):
    with pytest.raises(ValueError, match='was greater than 1'):
        cirq.asymmetric_depolarize(p_x, p_y, p_z)


def test_asymmetric_depolarizing_channel_text_diagram():
    a= cirq.asymmetric_depolarize(0.1, 0.2, 0.3)
    assert (cirq.circuit_diagram_info(a) == cirq.CircuitDiagramInfo(
        wire_symbols=('A(0.1,0.2,0.3)',)))


def test_depolarizing_channel():
    d = cirq.depolarize(0.3)
    np.testing.assert_almost_equal(cirq.channel(d),
                                   (np.sqrt(0.7) * np.eye(2),
                                    np.sqrt(0.1) * X,
                                    np.sqrt(0.1) * Y,
                                    np.sqrt(0.1) * Z))


def test_depolarizing_channel_repr():
    cirq.testing.assert_equivalent_repr(cirq.DepolarizingChannel(0.3))


def test_depolarizing_channel_str():
    assert str(cirq.depolarize(0.3)) == 'DepolarizingChannel(p=0.3)'


def test_depolarizing_channel_eq():
    et = cirq.testing.EqualsTester()
    et.add_equality_group(cirq.depolarize(0.0))
    et.add_equality_group(cirq.depolarize(0.1))
    et.add_equality_group(cirq.depolarize(0.9))
    et.add_equality_group(cirq.depolarize(1.0))


def test_depolarizing_channel_invalid_probability():
    with pytest.raises(ValueError, match='was less than 0'):
        cirq.depolarize(-0.1)
    with pytest.raises(ValueError, match='was greater than 1'):
        cirq.depolarize(1.1)


def test_depolarizing_channel_text_diagram():
    assert (cirq.circuit_diagram_info(cirq.depolarize(0.3))
            == cirq.CircuitDiagramInfo(wire_symbols=('D(0.3)',)))


