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
            == 'asymmetric_depolarize(p_x=0.1,p_y=0.2,p_z=0.3)')


def test_asymmetric_depolarizing_channel_eq():
    et = cirq.testing.EqualsTester()
    c = cirq.asymmetric_depolarize(0.0, 0.0, 0.0)
    et.make_equality_group(lambda: c)
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
    assert str(cirq.depolarize(0.3)) == 'depolarize(p=0.3)'


def test_depolarizing_channel_eq():
    et = cirq.testing.EqualsTester()
    c = cirq.depolarize(0.0)
    et.make_equality_group(lambda: c)
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


def test_generalized_amplitude_damping_channel():
    d = cirq.generalized_amplitude_damp(0.1, 0.3)
    np.testing.assert_almost_equal(cirq.channel(d),
              (np.sqrt(0.1) * np.array([[1., 0.], [0., np.sqrt(1.-0.3)]]),
               np.sqrt(0.1) * np.array([[0., np.sqrt(0.3)], [0., 0.]]),
               np.sqrt(0.9) * np.array([[np.sqrt(1. - 0.3), 0.], [0., 1.]]),
               np.sqrt(0.9) * np.array([[0., 0.], [np.sqrt(0.3), 0.]])))


def test_generalized_amplitude_damping_repr():
    cirq.testing.assert_equivalent_repr(
        cirq.GeneralizedAmplitudeDampingChannel(0.1, 0.3))

def test_generalized_amplitude_damping_str():
    assert (str(cirq.generalized_amplitude_damp(0.1, 0.3))
            == 'generalized_amplitude_damp(p=0.1,gamma=0.3)')


def test_generalized_amplitude_damping_channel_eq():
    et = cirq.testing.EqualsTester()
    c = cirq.generalized_amplitude_damp(0.0, 0.0)
    et.make_equality_group(lambda: c)
    et.add_equality_group(cirq.generalized_amplitude_damp(0.1, 0.0))
    et.add_equality_group(cirq.generalized_amplitude_damp(0.0, 0.1))
    et.add_equality_group(cirq.generalized_amplitude_damp(0.6, 0.4))
    et.add_equality_group(cirq.generalized_amplitude_damp(0.8, 0.2))



@pytest.mark.parametrize('p, gamma', (
    (-0.1, 0.0),
    (0.0, -0.1),
    (0.1, -0.1),
    (-0.1, 0.1)))
def test_generalized_amplitude_damping_channel_negative_probability(p, gamma):
    with pytest.raises(ValueError, match='was less than 0'):
        cirq.generalized_amplitude_damp(p, gamma)



@pytest.mark.parametrize('p,gamma', (
    (1.1, 0.0),
    (0.0, 1.1),
    (1.1, 1.1)))
def test_generalized_amplitude_damping_channel_bigly_probability(p, gamma):
    with pytest.raises(ValueError, match='was greater than 1'):
        cirq.generalized_amplitude_damp(p, gamma)



def test_generalized_amplitude_damping_channel_text_diagram():
    a= cirq.generalized_amplitude_damp(0.1, 0.3)
    assert (cirq.circuit_diagram_info(a) == cirq.CircuitDiagramInfo(
        wire_symbols=('GAD(0.1,0.3)',)))


def test_amplitude_damping_channel():
    d = cirq.amplitude_damp(0.3)
    np.testing.assert_almost_equal(cirq.channel(d),
                              (np.array([[1., 0.], [0., np.sqrt(1. - 0.3)]]),
                               np.array([[0., np.sqrt(0.3)], [0., 0.]])))


def test_amplitude_damping_channel_repr():
    cirq.testing.assert_equivalent_repr(
        cirq.AmplitudeDampingChannel(0.3))


def test_amplitude_damping_channel_str():
    assert (str(cirq.amplitude_damp(0.3))
            == 'amplitude_damp(gamma=0.3)')


def test_amplitude_damping_channel_eq():
    et = cirq.testing.EqualsTester()
    c = cirq.amplitude_damp(0.0)
    et.make_equality_group(lambda: c)
    et.add_equality_group(cirq.amplitude_damp(0.1))
    et.add_equality_group(cirq.amplitude_damp(0.4))
    et.add_equality_group(cirq.amplitude_damp(0.6))
    et.add_equality_group(cirq.amplitude_damp(0.8))

def test_amplitude_damping_channel_invalid_probability():
    with pytest.raises(ValueError, match='was less than 0'):
        cirq.amplitude_damp(-0.1)
    with pytest.raises(ValueError, match='was greater than 1'):
        cirq.amplitude_damp(1.1)


def test_amplitude_damping_channel_text_diagram():
    assert (cirq.circuit_diagram_info(cirq.amplitude_damp(0.3))
            == cirq.CircuitDiagramInfo(wire_symbols=('AD(0.3)',)))


def test_phase_damping_channel():
    d = cirq.phase_damp(0.3)
    np.testing.assert_almost_equal(cirq.channel(d),
                              (np.array([[1.0, 0.], [0., np.sqrt(1 - 0.3)]]),
                               np.array([[0., 0.], [0., np.sqrt(0.3)]])))


def test_phase_damping_channel_repr():
    cirq.testing.assert_equivalent_repr(
        cirq.PhaseDampingChannel(0.3))


def test_phase_damping_channel_str():
    assert (str(cirq.phase_damp(0.3))
            == 'phase_damp(gamma=0.3)')


def test_phase_damping_channel_eq():
    et = cirq.testing.EqualsTester()
    c = cirq.phase_damp(0.0)
    et.make_equality_group(lambda: c)
    et.add_equality_group(cirq.phase_damp(0.1))
    et.add_equality_group(cirq.phase_damp(0.4))
    et.add_equality_group(cirq.phase_damp(0.6))
    et.add_equality_group(cirq.phase_damp(0.8))


def test_phase_damping_channel_invalid_probability():
    with pytest.raises(ValueError, match='was less than 0'):
        cirq.phase_damp(-0.1)
    with pytest.raises(ValueError, match='was greater than 1'):
        cirq.phase_damp(1.1)


def test_phase_damping_channel_text_diagram():
    assert (cirq.circuit_diagram_info(cirq.phase_damp(0.3))
            == cirq.CircuitDiagramInfo(wire_symbols=('PD(0.3)',)))


def test_phase_flip_channel():
    d = cirq.phase_flip(0.3)
    np.testing.assert_almost_equal(cirq.channel(d),
                                  (np.sqrt(0.3) * np.eye(2),
                                   np.sqrt(1.-0.3) * Z))


def test_phase_flip_overload():
    d = cirq.phase_flip()
    d2 = cirq.phase_flip(0.3)
    assert str(d) == 'Z'
    assert str(d2) == 'phase_flip(p=0.3)'

def test_phase_flip_channel_repr():
    cirq.testing.assert_equivalent_repr(
        cirq.PhaseFlipChannel(0.3))


def test_phase_flip_channel_str():
    assert (str(cirq.phase_flip(0.3))
            == 'phase_flip(p=0.3)')


def test_phase_flip_channel_eq():
    et = cirq.testing.EqualsTester()
    c = cirq.phase_flip(0.0)
    et.make_equality_group(lambda: c)
    et.add_equality_group(cirq.phase_flip(0.1))
    et.add_equality_group(cirq.phase_flip(0.4))
    et.add_equality_group(cirq.phase_flip(0.6))
    et.add_equality_group(cirq.phase_flip(0.8))



def test_phase_flip_channel_invalid_probability():
    with pytest.raises(ValueError, match='was less than 0'):
        cirq.phase_flip(-0.1)
    with pytest.raises(ValueError, match='was greater than 1'):
        cirq.phase_flip(1.1)


def test_phase_flip_channel_text_diagram():
    assert (cirq.circuit_diagram_info(cirq.phase_flip(0.3))
            == cirq.CircuitDiagramInfo(wire_symbols=('PF(0.3)',)))


def test_bit_flip_channel():
    d = cirq.bit_flip(0.3)
    np.testing.assert_almost_equal(cirq.channel(d),
                                  (np.sqrt(0.3) * np.eye(2),
                                   np.sqrt(1.0 - 0.3) * X))


def test_bit_flip_overload():
    d = cirq.bit_flip()
    d2 = cirq.bit_flip(0.3)
    assert str(d) == 'X'
    assert str(d2) == 'bit_flip(p=0.3)'


def test_bit_flip_channel_repr():
    cirq.testing.assert_equivalent_repr(
        cirq.BitFlipChannel(0.3))


def test_bit_flip_channel_str():
    assert (str(cirq.bit_flip(0.3))
            == 'bit_flip(p=0.3)')


def test_bit_flip_channel_eq():
    et = cirq.testing.EqualsTester()
    c = cirq.bit_flip(0.0)
    et.make_equality_group(lambda: c)
    et.add_equality_group(cirq.bit_flip(0.1))
    et.add_equality_group(cirq.bit_flip(0.4))
    et.add_equality_group(cirq.bit_flip(0.6))
    et.add_equality_group(cirq.bit_flip(0.8))

def test_bit_flip_channel_invalid_probability():
    with pytest.raises(ValueError, match='was less than 0'):
        cirq.bit_flip(-0.1)
    with pytest.raises(ValueError, match='was greater than 1'):
        cirq.bit_flip(1.1)



def test_bit_flip_channel_text_diagram():
    assert (cirq.circuit_diagram_info(cirq.bit_flip(0.3))
            == cirq.CircuitDiagramInfo(wire_symbols=('BF(0.3)',)))


def test_rotation_error_channel():
    d = cirq.rotation_error(np.pi / 6., np.pi/12., np.pi/8.)
    np.testing.assert_almost_equal(cirq.channel(d),
                               (np.exp( X * 0.5 * (0.0 - 1.0j) * np.pi / 6.),
                                np.exp( Y * 0.5 * (0.0 - 1.0j) * np.pi / 12.),
                                np.exp( Z * 0.5 * (0.0 - 1.0j) * np.pi / 8.)))


def test_rotation_error_channel_repr():
    cirq.testing.assert_equivalent_repr(
        cirq.RotationErrorChannel(np.pi / 6., np.pi / 12., np.pi / 8.))


def test_rotation_error_channel_str():
    assert (str(cirq.rotation_error(0.3, 0.4, 0.5))
            == 'rotation_error(eps_x=0.3,eps_y=0.4,eps_z=0.5)')


def test_rotation_error_channel_eq():
    et = cirq.testing.EqualsTester()
    c = cirq.rotation_error(0.0, 0.0, 0.0)
    et.make_equality_group(lambda: c)
    et.add_equality_group(cirq.rotation_error(0.1, 0.1, 0.1))
    et.add_equality_group(cirq.rotation_error(0.9, 0.9, 0.9))
    et.add_equality_group(cirq.rotation_error(0.3, 0.4, 0.5))
    et.add_equality_group(cirq.rotation_error(0.0, 0.0, 0.1))
    et.add_equality_group(cirq.rotation_error(0.0, 0.1, 0.0))
    et.add_equality_group(cirq.rotation_error(0.1, 0.0, 0.0))


def test_rotation_error_channel_text_diagram():
    assert (cirq.circuit_diagram_info(cirq.rotation_error(0.3,0.4,0.5))
            == cirq.CircuitDiagramInfo(wire_symbols=('RE(0.3,0.4,0.5)',)))
