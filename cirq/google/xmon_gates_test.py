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

import cirq
import cirq.google as cg


def test_is_supported():
    a = cirq.GridQubit(0, 0)
    b = cirq.GridQubit(0, 1)
    c = cirq.GridQubit(1, 0)
    assert cg.XmonGate.is_supported_op(cirq.CZ(a, b))
    assert cg.XmonGate.is_supported_op(cirq.Z(a)**1)
    assert not cg.XmonGate.is_supported_op(cirq.CCZ(a, b, c))
    assert not cg.XmonGate.is_supported_op(cirq.SWAP(a, b))


def test_w_eq():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cg.ExpWGate(),
                          cg.ExpWGate(half_turns=1, axis_half_turns=0),
                          cg.ExpWGate(degs=180, axis_degs=0),
                          cg.ExpWGate(rads=np.pi, axis_rads=0))
    eq.make_equality_group(
        lambda: cg.ExpWGate(half_turns=cirq.Symbol('a')))
    eq.make_equality_group(lambda: cg.ExpWGate(half_turns=0))
    eq.make_equality_group(
        lambda: cg.ExpWGate(half_turns=0, axis_half_turns=cirq.Symbol('a')))
    eq.add_equality_group(
        cg.ExpWGate(half_turns=0, axis_half_turns=0.5),
        cg.ExpWGate(half_turns=0, axis_rads=np.pi / 2))
    eq.make_equality_group(
        lambda: cg.ExpWGate(
            half_turns=cirq.Symbol('ab'),
            axis_half_turns=cirq.Symbol('xy')))

    # Flipping the axis and negating the angle gives the same rotation.
    eq.add_equality_group(
        cg.ExpWGate(half_turns=0.25, axis_half_turns=1.5),
        cg.ExpWGate(half_turns=1.75, axis_half_turns=0.5))
    # ...but not when there are parameters.
    eq.add_equality_group(cg.ExpWGate(
        half_turns=cirq.Symbol('a'),
        axis_half_turns=1.5))
    eq.add_equality_group(cg.ExpWGate(
        half_turns=cirq.Symbol('a'),
        axis_half_turns=0.5))
    eq.add_equality_group(cg.ExpWGate(
        half_turns=0.25,
        axis_half_turns=cirq.Symbol('a')))
    eq.add_equality_group(cg.ExpWGate(
        half_turns=1.75,
        axis_half_turns=cirq.Symbol('a')))

    # Adding or subtracting whole turns/phases gives the same rotation.
    eq.add_equality_group(
        cg.ExpWGate(half_turns=-2.25, axis_half_turns=1.25),
        cg.ExpWGate(half_turns=7.75, axis_half_turns=11.25))


def test_w_str():
    assert str(cg.ExpWGate()) == 'X'
    assert str(cg.ExpWGate(axis_half_turns=0.99999, half_turns=0.5)) == 'X^-0.5'
    assert str(cg.ExpWGate(axis_half_turns=0.5, half_turns=0.25)) == 'Y^0.25'
    assert str(cg.ExpWGate(axis_half_turns=0.25,
                           half_turns=0.5)) == 'W(0.25)^0.5'


def test_w_decomposition():
    q = cirq.NamedQubit('q')
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.Circuit.from_ops(
            cg.ExpWGate(half_turns=0.25, axis_half_turns=0.3).on(q)
        ).to_unitary_matrix(),
        cirq.Circuit.from_ops(
            cirq.Z(q) ** -0.3,
            cirq.X(q) ** 0.25,
            cirq.Z(q) ** 0.3
        ).to_unitary_matrix(),
        atol=1e-8)


def test_w_inverse():
    assert cirq.inverse(cg.ExpWGate(half_turns=cirq.Symbol('a')), None) is None
    assert cirq.inverse(cg.ExpWGate()) == cg.ExpWGate()


def test_w_parameterize():
    parameterized_gate = cg.ExpWGate(half_turns=cirq.Symbol('a'),
                                     axis_half_turns=cirq.Symbol('b'))
    assert cirq.is_parameterized(parameterized_gate)
    assert cirq.unitary(parameterized_gate, None) is None
    resolver = cirq.ParamResolver({'a': 0.1, 'b': 0.2})
    resolved_gate = cirq.resolve_parameters(parameterized_gate, resolver)
    assert resolved_gate == cg.ExpWGate(half_turns=0.1, axis_half_turns=0.2)


def test_w_repr():
    gate = cg.ExpWGate(half_turns=0.1, axis_half_turns=0.2)
    assert repr(gate
                ) == 'cirq.google.ExpWGate(half_turns=0.1, axis_half_turns=0.2)'


def test_trace_bound():
    assert cirq.trace_distance_bound(cg.ExpWGate(half_turns=.001)) < 0.01
    assert cirq.trace_distance_bound(cg.ExpWGate(
        half_turns=cirq.Symbol('a'))) >= 1


def test_cirq_symbol_diagrams():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    c = cirq.Circuit.from_ops(
        cg.ExpWGate(axis_half_turns=cirq.Symbol('a'),
                    half_turns=cirq.Symbol('b')).on(q00),
        cirq.Z(q01)**cirq.Symbol('c'),
        cirq.CZ(q00, q01)**cirq.Symbol('d'),
    )
    cirq.testing.assert_has_diagram(c, """
(0, 0): ───W(a)^b───@─────
                    │
(0, 1): ───Z^c──────@^d───
""")


def test_z_diagram_chars():
    q = cirq.GridQubit(0, 1)
    c = cirq.Circuit.from_ops(
        cirq.Z(q),
        cirq.Z(q)**0.5,
        cirq.Z(q)**0.25,
        cirq.Z(q)**0.125,
        cirq.Z(q)**-0.5,
        cirq.Z(q)**-0.25,
    )
    cirq.testing.assert_has_diagram(c, """
(0, 1): ───Z───S───T───Z^0.125───S^-1───T^-1───
""")


def test_w_diagram_chars():
    q = cirq.GridQubit(0, 1)
    c = cirq.Circuit.from_ops(
        cg.ExpWGate(axis_half_turns=0).on(q),
        cg.ExpWGate(axis_half_turns=0.25).on(q),
        cg.ExpWGate(axis_half_turns=0.5).on(q),
        cg.ExpWGate(axis_half_turns=cirq.Symbol('a')).on(q),
    )
    cirq.testing.assert_has_diagram(c, """
(0, 1): ───X───W(0.25)───Y───W(a)───
""")
