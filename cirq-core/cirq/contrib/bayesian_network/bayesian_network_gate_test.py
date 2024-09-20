# Copyright 2022 The Cirq Developers
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
import cirq.contrib.bayesian_network as ccb


def test_basic_properties():
    gate = ccb.BayesianNetworkGate([('q0', None), ('q1', None), ('q2', None)], [])

    assert gate._has_unitary_()
    assert gate._qid_shape_() == (2, 2, 2)


def test_incorrect_constructor():
    # Success building.
    ccb.BayesianNetworkGate([('q0', 0.0), ('q1', None)], [('q1', ('q0',), [0.0, 0.0])])

    with pytest.raises(ValueError, match='Initial prob should be between 0 and 1.'):
        ccb.BayesianNetworkGate([('q0', 2016.0913), ('q1', None)], [('q1', ('q0',), [0.0, 0.0])])

    # This is an easy mistake where the tuple for q0 doesn't have the comma at the end.
    with pytest.raises(ValueError, match='Conditional prob params must be a tuple.'):
        ccb.BayesianNetworkGate([('q0', 0.0), ('q1', None)], [('q1', ('q0'), [0.0, 0.0])])

    with pytest.raises(ValueError, match='Incorrect number of conditional probs.'):
        ccb.BayesianNetworkGate([('q0', 0.0), ('q1', None)], [('q1', ('q0',), [0.0])])

    with pytest.raises(ValueError, match='Conditional prob should be between 0 and 1.'):
        ccb.BayesianNetworkGate([('q0', 0.0), ('q1', None)], [('q1', ('q0',), [2016.0913, 0.0])])


def test_repr():
    gate = ccb.BayesianNetworkGate([('q0', 0.0), ('q1', None)], [('q1', ('q0',), [0.0, 0.0])])

    assert repr(gate) == (
        "cirq.BayesianNetworkGate(init_probs=[('q0', 0.0), ('q1', None)],"
        + " arc_probs=[('q1', ('q0',), [0.0, 0.0])])"
    )


@pytest.mark.parametrize('input_prob', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
def test_prob_encoding(input_prob):
    q = cirq.NamedQubit('q')
    gate = ccb.BayesianNetworkGate([('q', input_prob)], [])
    circuit = cirq.Circuit(gate.on(q))
    phi = (
        cirq.Simulator().simulate(circuit, qubit_order=[q], initial_state=0).state_vector(copy=True)
    )
    actual_probs = [abs(x) ** 2 for x in phi]

    np.testing.assert_almost_equal(actual_probs[1], input_prob, decimal=4)


@pytest.mark.parametrize(
    'p0,p1,p2,expected_probs',
    [
        (0.0, 0.0, 0.0, [1, 0, 0, 0, 0, 0, 0, 0]),
        (0.0, 0.0, 1.0, [0, 1, 0, 0, 0, 0, 0, 0]),
        (0.0, 1.0, 0.0, [0, 0, 1, 0, 0, 0, 0, 0]),
        (0.0, 1.0, 1.0, [0, 0, 0, 1, 0, 0, 0, 0]),
        (1.0, 0.0, 0.0, [0, 0, 0, 0, 1, 0, 0, 0]),
        (1.0, 0.0, 1.0, [0, 0, 0, 0, 0, 1, 0, 0]),
        (1.0, 1.0, 0.0, [0, 0, 0, 0, 0, 0, 1, 0]),
        (1.0, 1.0, 1.0, [0, 0, 0, 0, 0, 0, 0, 1]),
    ],
)
@pytest.mark.parametrize('decompose', [True, False])
def test_initial_probs(p0, p1, p2, expected_probs, decompose):
    q0, q1, q2 = cirq.LineQubit.range(3)
    gate = ccb.BayesianNetworkGate([('q0', p0), ('q1', p1), ('q2', p2)], [])
    if decompose:
        circuit = cirq.Circuit(cirq.decompose(gate.on(q0, q1, q2)))
    else:
        circuit = cirq.Circuit(gate.on(q0, q1, q2))

    result = cirq.Simulator().simulate(circuit, qubit_order=[q0, q1, q2], initial_state=0)

    actual_probs = [abs(x) ** 2 for x in result.state_vector(copy=True)]

    np.testing.assert_allclose(actual_probs, expected_probs, atol=1e-6)


@pytest.mark.parametrize(
    'input_prob_q0,input_prob_q1,expected_prob_q2',
    [(0.0, 0.0, 0.1), (0.0, 1.0, 0.2), (1.0, 0.0, 0.3), (1.0, 1.0, 0.4)],
)
@pytest.mark.parametrize('decompose', [True, False])
def test_arc_probs(input_prob_q0, input_prob_q1, expected_prob_q2, decompose):
    q0, q1, q2 = cirq.LineQubit.range(3)
    gate = ccb.BayesianNetworkGate(
        [('q0', input_prob_q0), ('q1', input_prob_q1), ('q2', None)],
        [('q2', ('q0', 'q1'), [0.1, 0.2, 0.3, 0.4])],
    )
    if decompose:
        circuit = cirq.Circuit(cirq.decompose(gate.on(q0, q1, q2)))
    else:
        circuit = cirq.Circuit(gate.on(q0, q1, q2))

    result = cirq.Simulator().simulate(circuit, qubit_order=[q0, q1, q2], initial_state=0)

    probs = [abs(x) ** 2 for x in result.state_vector(copy=True)]

    actual_prob_q2_is_one = sum(probs[1::2])

    np.testing.assert_almost_equal(actual_prob_q2_is_one, expected_prob_q2, decimal=4)


def test_repro_figure_10_of_paper():
    # We try to create the network of figure 10 and check that the probabilities are the same as
    # the ones in table 10 of https://arxiv.org/abs/2004.14803.
    ir = cirq.NamedQubit('q4_IR')
    oi = cirq.NamedQubit('q3_OI')
    sm = cirq.NamedQubit('q2_SM')
    sp = cirq.NamedQubit('q0_SP')

    gate = ccb.BayesianNetworkGate(
        [('ir', 0.25), ('oi', 0.4), ('sm', None), ('sp', None)],
        [('sm', ('ir',), [0.7, 0.2]), ('sp', ('sm', 'oi'), [0.1, 0.5, 0.6, 0.8])],
    )

    qubits = [sp, sm, oi, ir]
    circuit = cirq.Circuit(cirq.decompose_once(gate.on(*qubits)))
    result = cirq.Simulator().simulate(circuit, qubit_order=qubits, initial_state=0)
    probs = np.asarray([abs(x) ** 2 for x in result.state_vector(copy=True)]).reshape(2, 2, 2, 2)

    # p(IR = 0) = 0.7500
    np.testing.assert_almost_equal(np.sum(probs[0, :, :, :]), 0.7500, decimal=6)

    # p(SM = 0) = 0.4250
    np.testing.assert_almost_equal(np.sum(probs[:, :, 0, :]), 0.4250, decimal=6)

    # p(OI = 0) = 0.6000
    np.testing.assert_almost_equal(np.sum(probs[:, 0, :, :]), 0.6000, decimal=6)

    # p(SP = 0) = 0.4985
    np.testing.assert_almost_equal(np.sum(probs[:, :, :, 0]), 0.4985, decimal=6)
