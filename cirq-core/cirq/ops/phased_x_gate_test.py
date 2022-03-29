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

import itertools

import numpy as np
import pytest
import sympy

import cirq


@pytest.mark.parametrize(
    'phase_exponent',
    [
        -0.5,
        0,
        0.1,
        0.25,
        0.5,
        1,
        sympy.Symbol('p'),
        sympy.Symbol('p') + 1,
    ],
)
def test_phased_x_consistent_protocols(phase_exponent):
    cirq.testing.assert_implements_consistent_protocols(
        cirq.PhasedXPowGate(phase_exponent=phase_exponent, exponent=1.0),
    )
    cirq.testing.assert_implements_consistent_protocols(
        cirq.PhasedXPowGate(phase_exponent=phase_exponent, exponent=1.0, global_shift=0.1),
    )


def test_init():
    g = cirq.PhasedXPowGate(phase_exponent=0.75, exponent=0.25, global_shift=0.1)
    assert g.phase_exponent == 0.75
    assert g.exponent == 0.25
    assert g._global_shift == 0.1

    x = cirq.PhasedXPowGate(phase_exponent=0, exponent=0.1, global_shift=0.2)
    assert x.phase_exponent == 0
    assert x.exponent == 0.1
    assert x._global_shift == 0.2

    y = cirq.PhasedXPowGate(phase_exponent=0.5, exponent=0.1, global_shift=0.2)
    assert y.phase_exponent == 0.5
    assert y.exponent == 0.1
    assert y._global_shift == 0.2


@pytest.mark.parametrize(
    'sym',
    [
        sympy.Symbol('a'),
        sympy.Symbol('a') + 1,
    ],
)
def test_no_symbolic_qasm_but_fails_gracefully(sym):
    q = cirq.NamedQubit('q')
    v = cirq.PhasedXPowGate(phase_exponent=sym).on(q)
    assert cirq.qasm(v, args=cirq.QasmArgs(), default=None) is None


def test_extrapolate():
    g = cirq.PhasedXPowGate(phase_exponent=0.25)
    assert g**0.25 == (g**0.5) ** 0.5

    # The gate is self-inverse, but there are hidden variables tracking the
    # exponent's sign and scale.
    assert g**-1 == g
    assert g.exponent == 1
    assert (g**-1).exponent == -1
    assert g**-0.5 == (g**-1) ** 0.5 != g**0.5
    assert g == g**3
    assert g**0.5 != (g**3) ** 0.5 == g**-0.5


def test_eq():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(
        cirq.PhasedXPowGate(phase_exponent=0),
        cirq.PhasedXPowGate(phase_exponent=0, exponent=1),
        cirq.PhasedXPowGate(exponent=1, phase_exponent=0),
        cirq.PhasedXPowGate(exponent=1, phase_exponent=2),
        cirq.PhasedXPowGate(exponent=1, phase_exponent=-2),
        cirq.X,
    )
    eq.add_equality_group(cirq.PhasedXPowGate(exponent=1, phase_exponent=2, global_shift=0.1))

    eq.add_equality_group(
        cirq.PhasedXPowGate(phase_exponent=0.5, exponent=1),
        cirq.PhasedXPowGate(phase_exponent=2.5, exponent=3),
        cirq.Y,
    )
    eq.add_equality_group(cirq.PhasedXPowGate(phase_exponent=0.5, exponent=0.25), cirq.Y**0.25)

    eq.add_equality_group(cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.25, global_shift=0.1))
    eq.add_equality_group(cirq.PhasedXPowGate(phase_exponent=2.25, exponent=0.25, global_shift=0.2))

    eq.make_equality_group(
        lambda: cirq.PhasedXPowGate(exponent=sympy.Symbol('a'), phase_exponent=0)
    )
    eq.make_equality_group(
        lambda: cirq.PhasedXPowGate(exponent=sympy.Symbol('a') + 1, phase_exponent=0)
    )
    eq.add_equality_group(cirq.PhasedXPowGate(exponent=sympy.Symbol('a'), phase_exponent=0.25))
    eq.add_equality_group(cirq.PhasedXPowGate(exponent=sympy.Symbol('a') + 1, phase_exponent=0.25))
    eq.add_equality_group(cirq.PhasedXPowGate(exponent=0, phase_exponent=0))
    eq.add_equality_group(cirq.PhasedXPowGate(exponent=0, phase_exponent=sympy.Symbol('a')))
    eq.add_equality_group(cirq.PhasedXPowGate(exponent=0, phase_exponent=sympy.Symbol('a') + 1))
    eq.add_equality_group(cirq.PhasedXPowGate(exponent=0, phase_exponent=0.5))
    eq.add_equality_group(
        cirq.PhasedXPowGate(exponent=sympy.Symbol('ab'), phase_exponent=sympy.Symbol('xy'))
    )
    eq.add_equality_group(
        cirq.PhasedXPowGate(exponent=sympy.Symbol('ab') + 1, phase_exponent=sympy.Symbol('xy') + 1)
    )

    eq.add_equality_group(
        cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.125, global_shift=-0.5),
        cirq.PhasedXPowGate(phase_exponent=0.25, exponent=4.125, global_shift=-0.5),
    )
    eq.add_equality_group(
        cirq.PhasedXPowGate(phase_exponent=0.25, exponent=2.125, global_shift=-0.5)
    )


def test_approx_eq():
    assert cirq.approx_eq(
        cirq.PhasedXPowGate(phase_exponent=0.1, exponent=0.2, global_shift=0.3),
        cirq.PhasedXPowGate(phase_exponent=0.1, exponent=0.2, global_shift=0.3),
        atol=1e-4,
    )
    assert not cirq.approx_eq(
        cirq.PhasedXPowGate(phase_exponent=0.1, exponent=0.2, global_shift=0.4),
        cirq.PhasedXPowGate(phase_exponent=0.1, exponent=0.2, global_shift=0.3),
        atol=1e-4,
    )
    assert cirq.approx_eq(
        cirq.PhasedXPowGate(phase_exponent=0.1, exponent=0.2, global_shift=0.4),
        cirq.PhasedXPowGate(phase_exponent=0.1, exponent=0.2, global_shift=0.3),
        atol=0.2,
    )


def test_str_repr():
    assert str(cirq.PhasedXPowGate(phase_exponent=0.25)) == 'PhX(0.25)'
    assert str(cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5)) == 'PhX(0.25)**0.5'
    assert repr(
        cirq.PhasedXPowGate(phase_exponent=0.25, exponent=4, global_shift=0.125)
        == 'cirq.PhasedXPowGate(phase_exponent=0.25, '
        'exponent=4, global_shift=0.125)'
    )
    assert (
        repr(cirq.PhasedXPowGate(phase_exponent=0.25)) == 'cirq.PhasedXPowGate(phase_exponent=0.25)'
    )


@pytest.mark.parametrize(
    'resolve_fn, global_shift', [(cirq.resolve_parameters, 0), (cirq.resolve_parameters_once, 0.1)]
)
def test_parameterize(resolve_fn, global_shift):
    parameterized_gate = cirq.PhasedXPowGate(
        exponent=sympy.Symbol('a'), phase_exponent=sympy.Symbol('b'), global_shift=global_shift
    )
    assert cirq.pow(parameterized_gate, 5) == cirq.PhasedXPowGate(
        exponent=sympy.Symbol('a') * 5, phase_exponent=sympy.Symbol('b'), global_shift=global_shift
    )
    assert cirq.unitary(parameterized_gate, default=None) is None
    assert cirq.is_parameterized(parameterized_gate)
    q = cirq.NamedQubit("q")
    parameterized_decomposed_circuit = cirq.Circuit(cirq.decompose(parameterized_gate(q)))
    for resolver in cirq.Linspace('a', 0, 2, 10) * cirq.Linspace('b', 0, 2, 10):
        resolved_gate = resolve_fn(parameterized_gate, resolver)
        assert resolved_gate == cirq.PhasedXPowGate(
            exponent=resolver.value_of('a'),
            phase_exponent=resolver.value_of('b'),
            global_shift=global_shift,
        )
        np.testing.assert_allclose(
            cirq.unitary(resolved_gate(q)),
            cirq.unitary(resolve_fn(parameterized_decomposed_circuit, resolver)),
            atol=1e-8,
        )

    unparameterized_gate = cirq.PhasedXPowGate(
        exponent=0.1, phase_exponent=0.2, global_shift=global_shift
    )
    assert not cirq.is_parameterized(unparameterized_gate)
    assert cirq.is_parameterized(unparameterized_gate ** sympy.Symbol('a'))
    assert cirq.is_parameterized(unparameterized_gate ** (sympy.Symbol('a') + 1))


def test_trace_bound():
    assert (
        cirq.trace_distance_bound(cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.001)) < 0.01
    )
    assert (
        cirq.trace_distance_bound(
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=sympy.Symbol('a'))
        )
        >= 1
    )


def test_diagram():
    q = cirq.NamedQubit('q')
    c = cirq.Circuit(
        cirq.PhasedXPowGate(phase_exponent=sympy.Symbol('a'), exponent=sympy.Symbol('b')).on(q),
        cirq.PhasedXPowGate(
            phase_exponent=sympy.Symbol('a') * 2, exponent=sympy.Symbol('b') + 1
        ).on(q),
        cirq.PhasedXPowGate(phase_exponent=0.25, exponent=1).on(q),
        cirq.PhasedXPowGate(phase_exponent=1, exponent=1).on(q),
    )
    cirq.testing.assert_has_diagram(
        c,
        """
q: ───PhX(a)^b───PhX(2*a)^(b + 1)───PhX(0.25)───PhX(1)───
""",
    )


def test_phase_by():
    g = cirq.PhasedXPowGate(phase_exponent=0.25)
    g2 = cirq.phase_by(g, 0.25, 0)
    assert g2 == cirq.PhasedXPowGate(phase_exponent=0.75)

    g = cirq.PhasedXPowGate(phase_exponent=0)
    g2 = cirq.phase_by(g, 0.125, 0)
    assert g2 == cirq.PhasedXPowGate(phase_exponent=0.25)

    g = cirq.PhasedXPowGate(phase_exponent=0.5)
    g2 = cirq.phase_by(g, 0.125, 0)
    assert g2 == cirq.PhasedXPowGate(phase_exponent=0.75)

    g = cirq.PhasedXPowGate(phase_exponent=0.5)
    g2 = cirq.phase_by(g, sympy.Symbol('b') + 1, 0)
    assert g2 == cirq.PhasedXPowGate(phase_exponent=2 * sympy.Symbol('b') + 2.5)


@pytest.mark.parametrize(
    'exponent,phase_exponent',
    itertools.product(np.arange(-2.5, 2.75, 0.25), repeat=2),
)
def test_exponent_consistency(exponent, phase_exponent):
    """Verifies that instances of PhasedX gate expose consistent exponents."""
    g = cirq.PhasedXPowGate(exponent=exponent, phase_exponent=phase_exponent)
    assert g.exponent in [exponent, -exponent]
    assert g.phase_exponent in [cirq.value.canonicalize_half_turns(g.phase_exponent)]

    g2 = cirq.PhasedXPowGate(exponent=g.exponent, phase_exponent=g.phase_exponent)
    assert g == g2

    u = cirq.protocols.unitary(g)
    u2 = cirq.protocols.unitary(g2)
    assert np.all(u == u2)
