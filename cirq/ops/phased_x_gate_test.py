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

import cirq


def test_init():
    g = cirq.PhasedXPowGate(0.75, 0.25)
    assert g.phase_exponent == 0.75
    assert g.exponent == 0.25

    assert isinstance(cirq.PhasedXPowGate(phase_exponent=0), cirq.XPowGate)
    assert isinstance(cirq.PhasedXPowGate(phase_exponent=1), cirq.XPowGate)
    assert isinstance(cirq.PhasedXPowGate(phase_exponent=0.5), cirq.YPowGate)
    assert isinstance(cirq.PhasedXPowGate(phase_exponent=1.5), cirq.YPowGate)


def test_extrapolate():
    g = cirq.PhasedXPowGate(phase_exponent=0.25)
    assert g**0.25 == (g**0.5)**0.5

    # The gate is self-inverse, but there are hidden variables tracking the
    # exponent's sign and scale.
    assert g**-1 == g
    assert g.exponent == 1
    assert (g**-1).exponent == -1
    assert g**-0.5 == (g**-1)**0.5 != g**0.5
    assert g == g**3
    assert g**0.5 != (g**3)**0.5 == g**-0.5


def test_eq():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.PhasedXPowGate(phase_exponent=0),
                          cirq.PhasedXPowGate(0, 1),
                          cirq.PhasedXPowGate(exponent=1, phase_exponent=0),
                          cirq.PhasedXPowGate(exponent=1, phase_exponent=1),
                          cirq.PhasedXPowGate(exponent=1, phase_exponent=2),
                          cirq.PhasedXPowGate(exponent=1, phase_exponent=-2),
                          cirq.X)
    eq.add_equality_group(cirq.PhasedXPowGate(0.5, 1),
                          cirq.PhasedXPowGate(2.5, 3),
                          cirq.Y,
                          cirq.PhasedXPowGate(-0.5, 1))
    eq.add_equality_group(cirq.PhasedXPowGate(0.5, 0.25),
                          cirq.Y**0.25)

    eq.make_equality_group(
        lambda: cirq.PhasedXPowGate(exponent=cirq.Symbol('a'),
                                    phase_exponent=0))
    eq.add_equality_group(
        cirq.PhasedXPowGate(exponent=cirq.Symbol('a'),
                            phase_exponent=0.25))
    eq.add_equality_group(cirq.PhasedXPowGate(exponent=0, phase_exponent=0))
    eq.add_equality_group(
        cirq.PhasedXPowGate(exponent=0, phase_exponent=cirq.Symbol('a')))
    eq.add_equality_group(cirq.PhasedXPowGate(exponent=0, phase_exponent=0.5))
    eq.add_equality_group(cirq.PhasedXPowGate(
        exponent=cirq.Symbol('ab'),
        phase_exponent=cirq.Symbol('xy')))


def test_str_repr():
    assert str(cirq.PhasedXPowGate(phase_exponent=0.25)) == 'PhasedX(0.25)'
    assert str(cirq.PhasedXPowGate(phase_exponent=0.25,
                                   exponent=0.5)) == 'PhasedX(0.25)^0.5'
    cirq.testing.assert_equivalent_repr(cirq.PhasedXPowGate(phase_exponent=0))
    cirq.testing.assert_equivalent_repr(cirq.PhasedXPowGate(0.1, 0.3))
    assert repr(cirq.PhasedXPowGate(0.25, 4)) == 'cirq.PhasedXPowGate(0.25, 4)'


def test_decomposition():
    cirq.testing.assert_decompose_is_consistent_with_unitary(
        cirq.PhasedXPowGate(exponent=0.25, phase_exponent=0.75))
    cirq.testing.assert_decompose_is_consistent_with_unitary(
        cirq.PhasedXPowGate(exponent=0.125, phase_exponent=0.25))


def test_parameterize():
    parameterized_gate = cirq.PhasedXPowGate(
        exponent=cirq.Symbol('a'),
        phase_exponent=cirq.Symbol('b'))
    assert cirq.pow(parameterized_gate, 5, default=None) is None
    assert parameterized_gate.default_decompose(
        [cirq.LineQubit(0)]) is NotImplemented
    assert cirq.unitary(parameterized_gate, default=None) is None
    assert cirq.is_parameterized(parameterized_gate)
    resolver = cirq.ParamResolver({'a': 0.1, 'b': 0.2})
    resolved_gate = cirq.resolve_parameters(parameterized_gate, resolver)
    assert resolved_gate == cirq.PhasedXPowGate(exponent=0.1,
                                                phase_exponent=0.2)


def test_trace_bound():
    assert cirq.trace_distance_bound(cirq.PhasedXPowGate(
        phase_exponent=0.25, exponent=.001)) < 0.01
    assert cirq.trace_distance_bound(cirq.PhasedXPowGate(
        phase_exponent=0.25, exponent=cirq.Symbol('a'))) >= 1


def test_diagram():
    q = cirq.NamedQubit('q')
    c = cirq.Circuit.from_ops(
        cirq.PhasedXPowGate(phase_exponent=cirq.Symbol('a'),
                            exponent=cirq.Symbol('b')).on(q),
        cirq.PhasedXPowGate(phase_exponent=0.25,
                            exponent=1).on(q)
    )
    cirq.testing.assert_has_diagram(c, """
q: ───PhasedX(a)^b───PhasedX(0.25)───
""")


def test_phase_by():
    g = cirq.PhasedXPowGate(phase_exponent=0.25)
    g2 = cirq.phase_by(g, 0.25, 0)
    assert g2 == cirq.PhasedXPowGate(phase_exponent=0.75)
    cirq.testing.assert_phase_by_is_consistent_with_unitary(g)

    g = cirq.PhasedXPowGate(phase_exponent=0)
    g2 = cirq.phase_by(g, 0.125, 0)
    assert g2 == cirq.PhasedXPowGate(phase_exponent=0.25)
    cirq.testing.assert_phase_by_is_consistent_with_unitary(g)

    g = cirq.PhasedXPowGate(phase_exponent=0.5)
    g2 = cirq.phase_by(g, 0.125, 0)
    assert g2 == cirq.PhasedXPowGate(phase_exponent=0.75)
    cirq.testing.assert_phase_by_is_consistent_with_unitary(g)
