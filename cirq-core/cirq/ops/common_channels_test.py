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

import re

import numpy as np
import pytest

import cirq
from cirq.protocols.act_on_protocol_test import DummySimulationState

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

no_precision = cirq.CircuitDiagramInfoArgs(
    known_qubits=None,
    known_qubit_count=None,
    use_unicode_characters=True,
    precision=None,
    label_map=None,
)

round_to_6_prec = cirq.CircuitDiagramInfoArgs(
    known_qubits=None,
    known_qubit_count=None,
    use_unicode_characters=True,
    precision=6,
    label_map=None,
)

round_to_2_prec = cirq.CircuitDiagramInfoArgs(
    known_qubits=None,
    known_qubit_count=None,
    use_unicode_characters=True,
    precision=2,
    label_map=None,
)


def assert_mixtures_equal(actual, expected):
    """Assert equal for tuple of mixed scalar and array types."""
    for a, e in zip(actual, expected):
        np.testing.assert_almost_equal(a[0], e[0])
        np.testing.assert_almost_equal(a[1], e[1])


def test_asymmetric_depolarizing_channel():
    d = cirq.asymmetric_depolarize(0.1, 0.2, 0.3)
    np.testing.assert_almost_equal(
        cirq.kraus(d),
        (np.sqrt(0.4) * np.eye(2), np.sqrt(0.1) * X, np.sqrt(0.2) * Y, np.sqrt(0.3) * Z),
    )
    cirq.testing.assert_consistent_channel(d)
    cirq.testing.assert_consistent_mixture(d)

    assert cirq.AsymmetricDepolarizingChannel(p_x=0, p_y=0.1, p_z=0).num_qubits() == 1


def test_asymmetric_depolarizing_mixture():
    d = cirq.asymmetric_depolarize(0.1, 0.2, 0.3)
    assert_mixtures_equal(cirq.mixture(d), ((0.4, np.eye(2)), (0.1, X), (0.2, Y), (0.3, Z)))
    assert cirq.has_mixture(d)


def test_asymmetric_depolarizing_channel_repr():
    cirq.testing.assert_equivalent_repr(cirq.AsymmetricDepolarizingChannel(0.1, 0.2, 0.3))


def test_asymmetric_depolarizing_channel_str():
    assert (
        str(cirq.asymmetric_depolarize(0.1, 0.2, 0.3))
        == "asymmetric_depolarize(error_probabilities={'I': 0.3999999999999999, "
        + "'X': 0.1, 'Y': 0.2, 'Z': 0.3})"
    )


def test_asymmetric_depolarizing_channel_eq():

    a = cirq.asymmetric_depolarize(0.0099999, 0.01)
    b = cirq.asymmetric_depolarize(0.01, 0.0099999)
    c = cirq.asymmetric_depolarize(0.0, 0.0, 0.0)

    assert cirq.approx_eq(a, b, atol=1e-2)

    et = cirq.testing.EqualsTester()
    et.make_equality_group(lambda: c)
    et.add_equality_group(cirq.asymmetric_depolarize(0.0, 0.0, 0.1))
    et.add_equality_group(cirq.asymmetric_depolarize(0.0, 0.1, 0.0))
    et.add_equality_group(cirq.asymmetric_depolarize(0.1, 0.0, 0.0))
    et.add_equality_group(cirq.asymmetric_depolarize(0.1, 0.2, 0.3))
    et.add_equality_group(cirq.asymmetric_depolarize(0.3, 0.4, 0.3))
    et.add_equality_group(cirq.asymmetric_depolarize(1.0, 0.0, 0.0))
    et.add_equality_group(cirq.asymmetric_depolarize(0.0, 1.0, 0.0))
    et.add_equality_group(cirq.asymmetric_depolarize(0.0, 0.0, 1.0))


@pytest.mark.parametrize(
    'p_x,p_y,p_z', ((-0.1, 0.0, 0.0), (0.0, -0.1, 0.0), (0.0, 0.0, -0.1), (0.1, -0.1, 0.1))
)
def test_asymmetric_depolarizing_channel_negative_probability(p_x, p_y, p_z):
    with pytest.raises(ValueError, match='was less than 0'):
        cirq.asymmetric_depolarize(p_x, p_y, p_z)


@pytest.mark.parametrize(
    'p_x,p_y,p_z', ((1.1, 0.0, 0.0), (0.0, 1.1, 0.0), (0.0, 0.0, 1.1), (0.1, 0.9, 0.1))
)
def test_asymmetric_depolarizing_channel_bigly_probability(p_x, p_y, p_z):
    with pytest.raises(ValueError, match='was greater than 1'):
        cirq.asymmetric_depolarize(p_x, p_y, p_z)


def test_asymmetric_depolarizing_channel_text_diagram():
    a = cirq.asymmetric_depolarize(1 / 9, 2 / 9, 3 / 9)
    assert cirq.circuit_diagram_info(a, args=no_precision) == cirq.CircuitDiagramInfo(
        wire_symbols=('A(0.1111111111111111,0.2222222222222222,' + '0.3333333333333333)',)
    )
    assert cirq.circuit_diagram_info(a, args=round_to_6_prec) == cirq.CircuitDiagramInfo(
        wire_symbols=('A(0.111111,0.222222,0.333333)',)
    )
    assert cirq.circuit_diagram_info(a, args=round_to_2_prec) == cirq.CircuitDiagramInfo(
        wire_symbols=('A(0.11,0.22,0.33)',)
    )


def test_depolarizing_channel():
    d = cirq.depolarize(0.3)
    np.testing.assert_almost_equal(
        cirq.kraus(d),
        (np.sqrt(0.7) * np.eye(2), np.sqrt(0.1) * X, np.sqrt(0.1) * Y, np.sqrt(0.1) * Z),
    )
    cirq.testing.assert_consistent_channel(d)
    cirq.testing.assert_consistent_mixture(d)


def test_depolarizing_channel_two_qubits():
    d = cirq.depolarize(0.15, n_qubits=2)
    np.testing.assert_almost_equal(
        cirq.kraus(d),
        (
            np.sqrt(0.85) * np.eye(4),
            np.sqrt(0.01) * np.kron(np.eye(2), X),
            np.sqrt(0.01) * np.kron(np.eye(2), Y),
            np.sqrt(0.01) * np.kron(np.eye(2), Z),
            np.sqrt(0.01) * np.kron(X, np.eye(2)),
            np.sqrt(0.01) * np.kron(X, X),
            np.sqrt(0.01) * np.kron(X, Y),
            np.sqrt(0.01) * np.kron(X, Z),
            np.sqrt(0.01) * np.kron(Y, np.eye(2)),
            np.sqrt(0.01) * np.kron(Y, X),
            np.sqrt(0.01) * np.kron(Y, Y),
            np.sqrt(0.01) * np.kron(Y, Z),
            np.sqrt(0.01) * np.kron(Z, np.eye(2)),
            np.sqrt(0.01) * np.kron(Z, X),
            np.sqrt(0.01) * np.kron(Z, Y),
            np.sqrt(0.01) * np.kron(Z, Z),
        ),
    )
    cirq.testing.assert_consistent_channel(d)
    cirq.testing.assert_consistent_mixture(d)

    assert d.num_qubits() == 2
    cirq.testing.assert_has_diagram(
        cirq.Circuit(d(*cirq.LineQubit.range(2))),
        """
0: ───D(0.15)───
      │
1: ───#2────────
        """,
    )


def test_depolarizing_mixture():
    d = cirq.depolarize(0.3)
    assert_mixtures_equal(cirq.mixture(d), ((0.7, np.eye(2)), (0.1, X), (0.1, Y), (0.1, Z)))
    assert cirq.has_mixture(d)


def test_depolarizing_mixture_two_qubits():
    d = cirq.depolarize(0.15, n_qubits=2)
    assert_mixtures_equal(
        cirq.mixture(d),
        (
            (0.85, np.eye(4)),
            (0.01, np.kron(np.eye(2), X)),
            (0.01, np.kron(np.eye(2), Y)),
            (0.01, np.kron(np.eye(2), Z)),
            (0.01, np.kron(X, np.eye(2))),
            (0.01, np.kron(X, X)),
            (0.01, np.kron(X, Y)),
            (0.01, np.kron(X, Z)),
            (0.01, np.kron(Y, np.eye(2))),
            (0.01, np.kron(Y, X)),
            (0.01, np.kron(Y, Y)),
            (0.01, np.kron(Y, Z)),
            (0.01, np.kron(Z, np.eye(2))),
            (0.01, np.kron(Z, X)),
            (0.01, np.kron(Z, Y)),
            (0.01, np.kron(Z, Z)),
        ),
    )
    assert cirq.has_mixture(d)


def test_depolarizing_channel_repr():
    cirq.testing.assert_equivalent_repr(cirq.DepolarizingChannel(0.3))


def test_depolarizing_channel_repr_two_qubits():
    cirq.testing.assert_equivalent_repr(cirq.DepolarizingChannel(0.3, n_qubits=2))


def test_depolarizing_channel_str():
    assert str(cirq.depolarize(0.3)) == 'depolarize(p=0.3)'


def test_depolarizing_channel_str_two_qubits():
    assert str(cirq.depolarize(0.3, n_qubits=2)) == 'depolarize(p=0.3,n_qubits=2)'


def test_deprecated_on_each_for_depolarizing_channel_one_qubit():
    q0 = cirq.LineQubit.range(1)
    op = cirq.DepolarizingChannel(p=0.1, n_qubits=1)

    op.on_each(q0)
    op.on_each([q0])
    with pytest.raises(ValueError, match="Gate was called with type different than Qid"):
        op.on_each('bogus object')


def test_deprecated_on_each_for_depolarizing_channel_two_qubits():
    q0, q1, q2, q3, q4, q5 = cirq.LineQubit.range(6)
    op = cirq.DepolarizingChannel(p=0.1, n_qubits=2)

    op.on_each([(q0, q1)])
    op.on_each([(q0, q1), (q2, q3)])
    op.on_each(zip([q0, q2, q4], [q1, q3, q5]))
    op.on_each((q0, q1))
    op.on_each([q0, q1])
    with pytest.raises(ValueError, match='Inputs to multi-qubit gates must be Sequence'):
        op.on_each(q0, q1)
    with pytest.raises(ValueError, match='All values in sequence should be Qids'):
        op.on_each([('bogus object 0', 'bogus object 1')])
    with pytest.raises(ValueError, match='All values in sequence should be Qids'):
        op.on_each(['01'])
    with pytest.raises(ValueError, match='All values in sequence should be Qids'):
        op.on_each([(False, None)])


def test_depolarizing_channel_apply_two_qubits():
    q0, q1 = cirq.LineQubit.range(2)
    op = cirq.DepolarizingChannel(p=0.1, n_qubits=2)
    op(q0, q1)


def test_asymmetric_depolarizing_channel_apply_two_qubits():
    q0, q1 = cirq.LineQubit.range(2)
    op = cirq.AsymmetricDepolarizingChannel(error_probabilities={'XX': 0.1})
    op(q0, q1)


def test_depolarizing_channel_eq():
    a = cirq.depolarize(p=0.0099999)
    b = cirq.depolarize(p=0.01)
    c = cirq.depolarize(0.0)

    assert cirq.approx_eq(a, b, atol=1e-2)

    et = cirq.testing.EqualsTester()

    et.make_equality_group(lambda: c)
    et.add_equality_group(cirq.depolarize(0.1))
    et.add_equality_group(cirq.depolarize(0.9))
    et.add_equality_group(cirq.depolarize(1.0))


def test_depolarizing_channel_invalid_probability():
    with pytest.raises(ValueError, match=re.escape('p(I) was greater than 1.')):
        cirq.depolarize(-0.1)
    with pytest.raises(ValueError, match=re.escape('p(I) was less than 0.')):
        cirq.depolarize(1.1)


def test_depolarizing_channel_text_diagram():
    d = cirq.depolarize(0.1234567)
    assert cirq.circuit_diagram_info(d, args=round_to_6_prec) == cirq.CircuitDiagramInfo(
        wire_symbols=('D(0.123457)',)
    )
    assert cirq.circuit_diagram_info(d, args=round_to_2_prec) == cirq.CircuitDiagramInfo(
        wire_symbols=('D(0.12)',)
    )
    assert cirq.circuit_diagram_info(d, args=no_precision) == cirq.CircuitDiagramInfo(
        wire_symbols=('D(0.1234567)',)
    )


def test_depolarizing_channel_text_diagram_two_qubits():
    d = cirq.depolarize(0.1234567, n_qubits=2)
    assert cirq.circuit_diagram_info(d, args=round_to_6_prec) == cirq.CircuitDiagramInfo(
        wire_symbols=('D(0.123457)', '#2')
    )
    assert cirq.circuit_diagram_info(d, args=round_to_2_prec) == cirq.CircuitDiagramInfo(
        wire_symbols=('D(0.12)', '#2')
    )
    assert cirq.circuit_diagram_info(d, args=no_precision) == cirq.CircuitDiagramInfo(
        wire_symbols=('D(0.1234567)', '#2')
    )


def test_generalized_amplitude_damping_channel():
    d = cirq.generalized_amplitude_damp(0.1, 0.3)
    np.testing.assert_almost_equal(
        cirq.kraus(d),
        (
            np.sqrt(0.1) * np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - 0.3)]]),
            np.sqrt(0.1) * np.array([[0.0, np.sqrt(0.3)], [0.0, 0.0]]),
            np.sqrt(0.9) * np.array([[np.sqrt(1.0 - 0.3), 0.0], [0.0, 1.0]]),
            np.sqrt(0.9) * np.array([[0.0, 0.0], [np.sqrt(0.3), 0.0]]),
        ),
    )
    cirq.testing.assert_consistent_channel(d)
    assert not cirq.has_mixture(d)


def test_generalized_amplitude_damping_repr():
    cirq.testing.assert_equivalent_repr(cirq.GeneralizedAmplitudeDampingChannel(0.1, 0.3))


def test_generalized_amplitude_damping_str():
    assert (
        str(cirq.generalized_amplitude_damp(0.1, 0.3))
        == 'generalized_amplitude_damp(p=0.1,gamma=0.3)'
    )


def test_generalized_amplitude_damping_channel_eq():
    a = cirq.generalized_amplitude_damp(0.0099999, 0.01)
    b = cirq.generalized_amplitude_damp(0.01, 0.0099999)

    assert cirq.approx_eq(a, b, atol=1e-2)

    et = cirq.testing.EqualsTester()
    c = cirq.generalized_amplitude_damp(0.0, 0.0)
    et.make_equality_group(lambda: c)
    et.add_equality_group(cirq.generalized_amplitude_damp(0.1, 0.0))
    et.add_equality_group(cirq.generalized_amplitude_damp(0.0, 0.1))
    et.add_equality_group(cirq.generalized_amplitude_damp(0.6, 0.4))
    et.add_equality_group(cirq.generalized_amplitude_damp(0.8, 0.2))


@pytest.mark.parametrize('p, gamma', ((-0.1, 0.0), (0.0, -0.1), (0.1, -0.1), (-0.1, 0.1)))
def test_generalized_amplitude_damping_channel_negative_probability(p, gamma):
    with pytest.raises(ValueError, match='was less than 0'):
        cirq.generalized_amplitude_damp(p, gamma)


@pytest.mark.parametrize('p,gamma', ((1.1, 0.0), (0.0, 1.1), (1.1, 1.1)))
def test_generalized_amplitude_damping_channel_bigly_probability(p, gamma):
    with pytest.raises(ValueError, match='was greater than 1'):
        cirq.generalized_amplitude_damp(p, gamma)


def test_generalized_amplitude_damping_channel_text_diagram():
    a = cirq.generalized_amplitude_damp(0.1, 0.39558391)
    assert cirq.circuit_diagram_info(a, args=round_to_6_prec) == cirq.CircuitDiagramInfo(
        wire_symbols=('GAD(0.1,0.395584)',)
    )
    assert cirq.circuit_diagram_info(a, args=round_to_2_prec) == cirq.CircuitDiagramInfo(
        wire_symbols=('GAD(0.1,0.4)',)
    )
    assert cirq.circuit_diagram_info(a, args=no_precision) == cirq.CircuitDiagramInfo(
        wire_symbols=('GAD(0.1,0.39558391)',)
    )


def test_amplitude_damping_channel():
    d = cirq.amplitude_damp(0.3)
    np.testing.assert_almost_equal(
        cirq.kraus(d),
        (
            np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - 0.3)]]),
            np.array([[0.0, np.sqrt(0.3)], [0.0, 0.0]]),
        ),
    )
    cirq.testing.assert_consistent_channel(d)
    assert not cirq.has_mixture(d)


def test_amplitude_damping_channel_repr():
    cirq.testing.assert_equivalent_repr(cirq.AmplitudeDampingChannel(0.3))


def test_amplitude_damping_channel_str():
    assert str(cirq.amplitude_damp(0.3)) == 'amplitude_damp(gamma=0.3)'


def test_amplitude_damping_channel_eq():
    a = cirq.amplitude_damp(0.0099999)
    b = cirq.amplitude_damp(0.01)
    c = cirq.amplitude_damp(0.0)

    assert cirq.approx_eq(a, b, atol=1e-2)

    et = cirq.testing.EqualsTester()
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
    ad = cirq.amplitude_damp(0.38059322)
    assert cirq.circuit_diagram_info(ad, args=round_to_6_prec) == cirq.CircuitDiagramInfo(
        wire_symbols=('AD(0.380593)',)
    )
    assert cirq.circuit_diagram_info(ad, args=round_to_2_prec) == cirq.CircuitDiagramInfo(
        wire_symbols=('AD(0.38)',)
    )
    assert cirq.circuit_diagram_info(ad, args=no_precision) == cirq.CircuitDiagramInfo(
        wire_symbols=('AD(0.38059322)',)
    )


def test_reset_channel():
    r = cirq.reset(cirq.LineQubit(0))
    np.testing.assert_almost_equal(
        cirq.kraus(r), (np.array([[1.0, 0.0], [0.0, 0]]), np.array([[0.0, 1.0], [0.0, 0.0]]))
    )
    cirq.testing.assert_consistent_channel(r)
    assert not cirq.has_mixture(r)

    assert cirq.num_qubits(r) == 1
    assert cirq.qid_shape(r) == (2,)

    r = cirq.reset(cirq.LineQid(0, dimension=3))
    np.testing.assert_almost_equal(
        cirq.kraus(r),
        (
            np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
            np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
            np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]]),
        ),
    )  # yapf: disable
    cirq.testing.assert_consistent_channel(r)
    assert not cirq.has_mixture(r)
    assert cirq.qid_shape(r) == (3,)


def test_reset_channel_equality():
    assert cirq.reset(cirq.LineQubit(0)).gate == cirq.ResetChannel()
    assert cirq.reset(cirq.LineQid(0, 3)).gate == cirq.ResetChannel(3)


def test_reset_channel_repr():
    cirq.testing.assert_equivalent_repr(cirq.ResetChannel())
    cirq.testing.assert_equivalent_repr(cirq.ResetChannel(3))


def test_reset_channel_str():
    assert str(cirq.ResetChannel()) == 'reset'
    assert str(cirq.ResetChannel(3)) == 'reset'


def test_reset_channel_text_diagram():
    assert cirq.circuit_diagram_info(cirq.ResetChannel()) == cirq.CircuitDiagramInfo(
        wire_symbols=('R',)
    )
    assert cirq.circuit_diagram_info(cirq.ResetChannel(3)) == cirq.CircuitDiagramInfo(
        wire_symbols=('R',)
    )


def test_reset_act_on():
    with pytest.raises(TypeError, match="Failed to act"):
        cirq.act_on(cirq.ResetChannel(), DummySimulationState(), qubits=())

    args = cirq.StateVectorSimulationState(
        available_buffer=np.empty(shape=(2, 2, 2, 2, 2), dtype=np.complex64),
        qubits=cirq.LineQubit.range(5),
        prng=np.random.RandomState(),
        initial_state=cirq.one_hot(
            index=(1, 1, 1, 1, 1), shape=(2, 2, 2, 2, 2), dtype=np.complex64
        ),
        dtype=np.complex64,
    )

    cirq.act_on(cirq.ResetChannel(), args, [cirq.LineQubit(1)])
    assert args.log_of_measurement_results == {}
    np.testing.assert_allclose(
        args.target_tensor,
        cirq.one_hot(index=(1, 0, 1, 1, 1), shape=(2, 2, 2, 2, 2), dtype=np.complex64),
    )

    cirq.act_on(cirq.ResetChannel(), args, [cirq.LineQubit(1)])
    assert args.log_of_measurement_results == {}
    np.testing.assert_allclose(
        args.target_tensor,
        cirq.one_hot(index=(1, 0, 1, 1, 1), shape=(2, 2, 2, 2, 2), dtype=np.complex64),
    )


def test_reset_each():
    qubits = cirq.LineQubit.range(8)
    for n in range(len(qubits) + 1):
        ops = cirq.reset_each(*qubits[:n])
        assert len(ops) == n
        for i, op in enumerate(ops):
            assert isinstance(op.gate, cirq.ResetChannel)
            assert op.qubits == (qubits[i],)


def test_reset_consistency():
    two_d_chan = cirq.ResetChannel()
    cirq.testing.assert_has_consistent_apply_channel(two_d_chan)
    three_d_chan = cirq.ResetChannel(dimension=3)
    cirq.testing.assert_has_consistent_apply_channel(three_d_chan)


def test_phase_damping_channel():
    d = cirq.phase_damp(0.3)
    np.testing.assert_almost_equal(
        cirq.kraus(d),
        (
            np.array([[1.0, 0.0], [0.0, np.sqrt(1 - 0.3)]]),
            np.array([[0.0, 0.0], [0.0, np.sqrt(0.3)]]),
        ),
    )
    cirq.testing.assert_consistent_channel(d)
    assert not cirq.has_mixture(d)


def test_phase_damping_channel_repr():
    cirq.testing.assert_equivalent_repr(cirq.PhaseDampingChannel(0.3))


def test_phase_damping_channel_str():
    assert str(cirq.phase_damp(0.3)) == 'phase_damp(gamma=0.3)'


def test_phase_damping_channel_eq():
    a = cirq.phase_damp(0.0099999)
    b = cirq.phase_damp(0.01)
    c = cirq.phase_damp(0.0)

    assert cirq.approx_eq(a, b, atol=1e-2)

    et = cirq.testing.EqualsTester()
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
    pd = cirq.phase_damp(0.1000009)
    assert cirq.circuit_diagram_info(pd, args=round_to_6_prec) == cirq.CircuitDiagramInfo(
        wire_symbols=('PD(0.100001)',)
    )
    assert cirq.circuit_diagram_info(pd, args=round_to_2_prec) == cirq.CircuitDiagramInfo(
        wire_symbols=('PD(0.1)',)
    )
    assert cirq.circuit_diagram_info(pd, args=no_precision) == cirq.CircuitDiagramInfo(
        wire_symbols=('PD(0.1000009)',)
    )


def test_phase_damp_consistency():
    full_damp = cirq.PhaseDampingChannel(gamma=1)
    cirq.testing.assert_has_consistent_apply_channel(full_damp)
    partial_damp = cirq.PhaseDampingChannel(gamma=0.5)
    cirq.testing.assert_has_consistent_apply_channel(partial_damp)
    no_damp = cirq.PhaseDampingChannel(gamma=0)
    cirq.testing.assert_has_consistent_apply_channel(no_damp)


def test_phase_flip_channel():
    d = cirq.phase_flip(0.3)
    np.testing.assert_almost_equal(
        cirq.kraus(d), (np.sqrt(1.0 - 0.3) * np.eye(2), np.sqrt(0.3) * Z)
    )
    cirq.testing.assert_consistent_channel(d)
    cirq.testing.assert_consistent_mixture(d)


def test_phase_flip_mixture():
    d = cirq.phase_flip(0.3)
    assert_mixtures_equal(cirq.mixture(d), ((0.7, np.eye(2)), (0.3, Z)))
    assert cirq.has_mixture(d)


def test_phase_flip_overload():
    d = cirq.phase_flip()
    d2 = cirq.phase_flip(0.3)
    assert str(d) == 'Z'
    assert str(d2) == 'phase_flip(p=0.3)'


def test_phase_flip_channel_repr():
    cirq.testing.assert_equivalent_repr(cirq.PhaseFlipChannel(0.3))


def test_phase_flip_channel_str():
    assert str(cirq.phase_flip(0.3)) == 'phase_flip(p=0.3)'


def test_phase_flip_channel_eq():
    a = cirq.phase_flip(0.0099999)
    b = cirq.phase_flip(0.01)
    c = cirq.phase_flip(0.0)

    assert cirq.approx_eq(a, b, atol=1e-2)

    et = cirq.testing.EqualsTester()
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
    pf = cirq.phase_flip(0.987654)
    assert cirq.circuit_diagram_info(pf, args=round_to_6_prec) == cirq.CircuitDiagramInfo(
        wire_symbols=('PF(0.987654)',)
    )
    assert cirq.circuit_diagram_info(pf, args=round_to_2_prec) == cirq.CircuitDiagramInfo(
        wire_symbols=('PF(0.99)',)
    )
    assert cirq.circuit_diagram_info(pf, no_precision) == cirq.CircuitDiagramInfo(
        wire_symbols=('PF(0.987654)',)
    )


def test_bit_flip_channel():
    d = cirq.bit_flip(0.3)
    np.testing.assert_almost_equal(
        cirq.kraus(d), (np.sqrt(1.0 - 0.3) * np.eye(2), np.sqrt(0.3) * X)
    )
    cirq.testing.assert_consistent_channel(d)
    cirq.testing.assert_consistent_mixture(d)


def test_bit_flip_mixture():
    d = cirq.bit_flip(0.3)
    assert_mixtures_equal(cirq.mixture(d), ((0.7, np.eye(2)), (0.3, X)))
    assert cirq.has_mixture(d)


def test_bit_flip_overload():
    d = cirq.bit_flip()
    d2 = cirq.bit_flip(0.3)
    assert str(d) == 'X'
    assert str(d2) == 'bit_flip(p=0.3)'


def test_bit_flip_channel_repr():
    cirq.testing.assert_equivalent_repr(cirq.BitFlipChannel(0.3))


def test_bit_flip_channel_str():
    assert str(cirq.bit_flip(0.3)) == 'bit_flip(p=0.3)'


def test_bit_flip_channel_eq():

    a = cirq.bit_flip(0.0099999)
    b = cirq.bit_flip(0.01)
    c = cirq.bit_flip(0.0)

    assert cirq.approx_eq(a, b, atol=1e-2)

    et = cirq.testing.EqualsTester()
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
    bf = cirq.bit_flip(0.1234567)
    assert cirq.circuit_diagram_info(bf, args=round_to_6_prec) == cirq.CircuitDiagramInfo(
        wire_symbols=('BF(0.123457)',)
    )
    assert cirq.circuit_diagram_info(bf, args=round_to_2_prec) == cirq.CircuitDiagramInfo(
        wire_symbols=('BF(0.12)',)
    )
    assert cirq.circuit_diagram_info(bf, args=no_precision) == cirq.CircuitDiagramInfo(
        wire_symbols=('BF(0.1234567)',)
    )


def test_stabilizer_supports_depolarize():
    with pytest.raises(TypeError, match="act_on"):
        for _ in range(100):
            cirq.act_on(cirq.depolarize(3 / 4), DummySimulationState(), qubits=())

    q = cirq.LineQubit(0)
    c = cirq.Circuit(cirq.depolarize(3 / 4).on(q), cirq.measure(q, key='m'))
    m = np.sum(cirq.StabilizerSampler().sample(c, repetitions=100)['m'])
    assert 5 < m < 95


def test_default_asymmetric_depolarizing_channel():
    d = cirq.asymmetric_depolarize()
    assert d.p_i == 1.0
    assert d.p_x == 0.0
    assert d.p_y == 0.0
    assert d.p_z == 0.0
    assert d.num_qubits() == 1


def test_bad_error_probabilities_gate():
    with pytest.raises(ValueError, match='AB is not made solely of I, X, Y, Z.'):
        cirq.asymmetric_depolarize(error_probabilities={'AB': 1.0})
    with pytest.raises(ValueError, match='Y must have 2 Pauli gates.'):
        cirq.asymmetric_depolarize(error_probabilities={'IX': 0.8, 'Y': 0.2})


def test_bad_probs():
    with pytest.raises(ValueError, match=re.escape('p(X) was greater than 1.')):
        cirq.asymmetric_depolarize(error_probabilities={'X': 1.1, 'Y': -0.1})
    with pytest.raises(ValueError, match=re.escape('Probabilities do not add up to 1')):
        cirq.asymmetric_depolarize(error_probabilities={'X': 0.7, 'Y': 0.6})


def test_missing_prob_mass():
    with pytest.raises(ValueError, match='Probabilities do not add up to 1'):
        cirq.asymmetric_depolarize(error_probabilities={'X': 0.1, 'I': 0.2})
    d = cirq.asymmetric_depolarize(error_probabilities={'X': 0.1})
    np.testing.assert_almost_equal(d.error_probabilities['I'], 0.9)


def test_multi_asymmetric_depolarizing_channel():
    d = cirq.asymmetric_depolarize(error_probabilities={'II': 0.8, 'XX': 0.2})
    np.testing.assert_almost_equal(
        cirq.kraus(d), (np.sqrt(0.8) * np.eye(4), np.sqrt(0.2) * np.kron(X, X))
    )
    cirq.testing.assert_consistent_channel(d)
    cirq.testing.assert_consistent_mixture(d)
    np.testing.assert_equal(d._num_qubits_(), 2)

    with pytest.raises(ValueError, match="num_qubits should be 1"):
        assert d.p_i == 1.0
    with pytest.raises(ValueError, match="num_qubits should be 1"):
        assert d.p_x == 0.0
    with pytest.raises(ValueError, match="num_qubits should be 1"):
        assert d.p_y == 0.0
    with pytest.raises(ValueError, match="num_qubits should be 1"):
        assert d.p_z == 0.0


def test_multi_asymmetric_depolarizing_mixture():
    d = cirq.asymmetric_depolarize(error_probabilities={'II': 0.8, 'XX': 0.2})
    assert_mixtures_equal(cirq.mixture(d), ((0.8, np.eye(4)), (0.2, np.kron(X, X))))
    assert cirq.has_mixture(d)
    np.testing.assert_equal(d._num_qubits_(), 2)


def test_multi_asymmetric_depolarizing_channel_repr():
    cirq.testing.assert_equivalent_repr(
        cirq.AsymmetricDepolarizingChannel(error_probabilities={'II': 0.8, 'XX': 0.2})
    )


def test_multi_asymmetric_depolarizing_channel_str():
    assert str(cirq.asymmetric_depolarize(error_probabilities={'II': 0.8, 'XX': 0.2})) == (
        "asymmetric_depolarize(error_probabilities={'II': 0.8, 'XX': 0.2})"
    )


def test_multi_asymmetric_depolarizing_channel_text_diagram():
    a = cirq.asymmetric_depolarize(error_probabilities={'II': 2 / 3, 'XX': 1 / 3})
    assert cirq.circuit_diagram_info(a, args=no_precision) == cirq.CircuitDiagramInfo(
        wire_symbols=('A(II:0.6666666666666666, XX:0.3333333333333333)',)
    )
    assert cirq.circuit_diagram_info(a, args=round_to_6_prec) == cirq.CircuitDiagramInfo(
        wire_symbols=('A(II:0.666667, XX:0.333333)',)
    )
    assert cirq.circuit_diagram_info(a, args=round_to_2_prec) == cirq.CircuitDiagramInfo(
        wire_symbols=('A(II:0.67, XX:0.33)',)
    )
    assert cirq.circuit_diagram_info(a, args=no_precision) == cirq.CircuitDiagramInfo(
        wire_symbols=('A(II:0.6666666666666666, XX:0.3333333333333333)',)
    )


def test_reset_stabilizer():
    assert cirq.has_stabilizer_effect(cirq.reset(cirq.LineQubit(0)))
