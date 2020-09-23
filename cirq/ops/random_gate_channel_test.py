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


def test_init():
    p = cirq.RandomGateChannel(sub_gate=cirq.X, probability=0.5)
    assert p.sub_gate is cirq.X
    assert p.probability == 0.5

    with pytest.raises(ValueError, match='probability'):
        _ = cirq.RandomGateChannel(sub_gate=cirq.X, probability=2)
    with pytest.raises(ValueError, match='probability'):
        _ = cirq.RandomGateChannel(sub_gate=cirq.X, probability=-1)


def test_eq():
    eq = cirq.testing.EqualsTester()
    q = cirq.LineQubit(0)

    eq.add_equality_group(
        cirq.RandomGateChannel(sub_gate=cirq.X, probability=0.5),
        cirq.X.with_probability(0.5))

    # Each field matters for equality.
    eq.add_equality_group(cirq.Y.with_probability(0.5))
    eq.add_equality_group(cirq.X.with_probability(0.25))

    # `with_probability(1)` doesn't wrap
    eq.add_equality_group(cirq.X, cirq.X.with_probability(1))
    eq.add_equality_group(
        cirq.X.with_probability(1).on(q),
        cirq.X.on(q).with_probability(1),
        cirq.X(q),
    )

    # `with_probability` with `on`.
    eq.add_equality_group(
        cirq.X.with_probability(0.5).on(q),
        cirq.X.on(q).with_probability(0.5),
    )

    # Flattening.
    eq.add_equality_group(
        cirq.RandomGateChannel(sub_gate=cirq.Z, probability=0.25),
        cirq.RandomGateChannel(sub_gate=cirq.RandomGateChannel(sub_gate=cirq.Z,
                                                               probability=0.5),
                               probability=0.5),
        cirq.Z.with_probability(0.5).with_probability(0.5),
        cirq.Z.with_probability(0.25))

    # Supports approximate equality.
    assert cirq.approx_eq(
        cirq.X.with_probability(0.5),
        cirq.X.with_probability(0.50001),
        atol=1e-2,
    )
    assert not cirq.approx_eq(
        cirq.X.with_probability(0.5),
        cirq.X.with_probability(0.50001),
        atol=1e-8,
    )


def test_consistent_protocols():
    cirq.testing.assert_implements_consistent_protocols(
        cirq.RandomGateChannel(sub_gate=cirq.X, probability=1))
    cirq.testing.assert_implements_consistent_protocols(
        cirq.RandomGateChannel(sub_gate=cirq.X, probability=0))
    cirq.testing.assert_implements_consistent_protocols(
        cirq.RandomGateChannel(sub_gate=cirq.X,
                               probability=sympy.Symbol('x') / 2))
    cirq.testing.assert_implements_consistent_protocols(
        cirq.RandomGateChannel(sub_gate=cirq.X, probability=0.5))


def test_diagram():

    class NoDetailsGate(cirq.Gate):

        def num_qubits(self) -> int:
            raise NotImplementedError()

    assert cirq.circuit_diagram_info(NoDetailsGate().with_probability(0.5),
                                     None) is None

    a, b = cirq.LineQubit.range(2)
    cirq.testing.assert_has_diagram(
        cirq.Circuit(cirq.CNOT(a, b).with_probability(0.125)), """
0: ───@[prob=0.125]───
      │
1: ───X───────────────
        """)

    cirq.testing.assert_has_diagram(cirq.Circuit(
        cirq.CNOT(a, b).with_probability(0.125)),
                                    """
0: ───@[prob=0.1]───
      │
1: ───X─────────────
        """,
                                    precision=1)


def test_parameterized():
    op = cirq.X.with_probability(sympy.Symbol('x'))
    assert cirq.is_parameterized(op)
    assert not cirq.has_channel(op)
    assert not cirq.has_mixture(op)

    op2 = cirq.resolve_parameters(op, {'x': 0.5})
    assert op2 == cirq.X.with_probability(0.5)
    assert not cirq.is_parameterized(op2)
    assert cirq.has_channel(op2)
    assert cirq.has_mixture(op2)


def test_mixture():

    class NoDetailsGate(cirq.Gate):

        def num_qubits(self) -> int:
            return 1

    assert not cirq.has_mixture(NoDetailsGate().with_probability(0.5))
    assert cirq.mixture(NoDetailsGate().with_probability(0.5), None) is None

    assert cirq.mixture(cirq.X.with_probability(sympy.Symbol('x')),
                        None) is None

    m = cirq.mixture(cirq.X.with_probability(0.25))
    assert len(m) == 2
    assert m[0][0] == 0.25
    np.testing.assert_allclose(cirq.unitary(cirq.X), m[0][1])
    assert m[1][0] == 0.75
    np.testing.assert_allclose(cirq.unitary(cirq.I), m[1][1])

    m = cirq.mixture(cirq.bit_flip(1 / 4).with_probability(1 / 8))
    assert len(m) == 3
    assert {p for p, _ in m} == {7 / 8, 1 / 32, 3 / 32}


def assert_channel_sums_to_identity(val):
    m = cirq.channel(val)
    s = sum(np.conj(e.T) @ e for e in m)
    np.testing.assert_allclose(s,
                               np.eye(np.product(cirq.qid_shape(val))),
                               atol=1e-8)


def test_channel():

    class NoDetailsGate(cirq.Gate):

        def num_qubits(self) -> int:
            return 1

    assert not cirq.has_channel(NoDetailsGate().with_probability(0.5))
    assert cirq.channel(NoDetailsGate().with_probability(0.5), None) is None
    assert cirq.channel(cirq.X.with_probability(sympy.Symbol('x')),
                        None) is None
    assert_channel_sums_to_identity(cirq.X.with_probability(0.25))
    assert_channel_sums_to_identity(cirq.bit_flip(0.75).with_probability(0.25))
    assert_channel_sums_to_identity(
        cirq.amplitude_damp(0.75).with_probability(0.25))

    m = cirq.channel(cirq.X.with_probability(0.25))
    assert len(m) == 2
    np.testing.assert_allclose(
        m[0],
        cirq.unitary(cirq.X) * np.sqrt(0.25),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        m[1],
        cirq.unitary(cirq.I) * np.sqrt(0.75),
        atol=1e-8,
    )

    m = cirq.channel(cirq.bit_flip(0.75).with_probability(0.25))
    assert len(m) == 3
    np.testing.assert_allclose(
        m[0],
        cirq.unitary(cirq.I) * np.sqrt(0.25) * np.sqrt(0.25),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        m[1],
        cirq.unitary(cirq.X) * np.sqrt(0.25) * np.sqrt(0.75),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        m[2],
        cirq.unitary(cirq.I) * np.sqrt(0.75),
        atol=1e-8,
    )

    m = cirq.channel(cirq.amplitude_damp(0.75).with_probability(0.25))
    assert len(m) == 3
    np.testing.assert_allclose(
        m[0],
        np.array([[1, 0], [0, np.sqrt(1 - 0.75)]]) * np.sqrt(0.25),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        m[1],
        np.array([[0, np.sqrt(0.75)], [0, 0]]) * np.sqrt(0.25),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        m[2],
        cirq.unitary(cirq.I) * np.sqrt(0.75),
        atol=1e-8,
    )


def test_trace_distance():
    t = cirq.trace_distance_bound
    assert 0.999 <= t(cirq.X.with_probability(sympy.Symbol('x')))
    assert t(cirq.X.with_probability(0)) == 0
    assert 0.49 <= t(cirq.X.with_probability(0.5)) <= 0.51
    assert 0.7 <= t(cirq.S.with_probability(sympy.Symbol('x'))) <= 0.71
    assert 0.35 <= t(cirq.S.with_probability(0.5)) <= 0.36


def test_str():
    assert str(cirq.X.with_probability(0.5)) == 'X[prob=0.5]'


def test_stabilizer_supports_probability():
    q = cirq.LineQubit(0)
    c = cirq.Circuit(cirq.X(q).with_probability(0.5), cirq.measure(q, key='m'))
    m = np.sum(cirq.StabilizerSampler().sample(c, repetitions=100)['m'])
    assert 5 < m < 95


def test_unsupported_stabilizer_safety():
    with pytest.raises(TypeError, match="act_on"):
        for _ in range(100):
            cirq.act_on(cirq.X.with_probability(0.5), object())
    with pytest.raises(TypeError, match="act_on"):
        cirq.act_on(cirq.X.with_probability(sympy.Symbol('x')), object())

    q = cirq.LineQubit(0)
    c = cirq.Circuit((cirq.X(q)**0.25).with_probability(0.5),
                     cirq.measure(q, key='m'))
    with pytest.raises(TypeError, match='Failed to act'):
        cirq.StabilizerSampler().sample(c, repetitions=100)
