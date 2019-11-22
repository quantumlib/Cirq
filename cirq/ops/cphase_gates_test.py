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
import sympy

import cirq


def test_cz00_str():
    assert str(cirq.CZ00) == 'CZ00'
    assert str(cirq.CZ00**0.5) == 'CZ00**0.5'
    assert str(cirq.CZ00**-0.25) == 'CZ00**-0.25'


def test_cz00_repr():
    assert repr(cirq.CZ00) == 'cirq.CZ00'
    assert repr(cirq.CZ00**0.5) == '(cirq.CZ00**0.5)'
    assert repr(cirq.CZ00**-0.25) == '(cirq.CZ00**-0.25)'
    assert repr(cirq.CZPowGate00(
        exponent=1,
        global_shift=-0.5)) == 'cirq.CZPowGate00(exponent=1, global_shift=-0.5)'


def test_cz00_unitary():
    assert np.allclose(
        cirq.unitary(cirq.CZ00),
        np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    assert np.allclose(
        cirq.unitary(cirq.CZ00**0.5),
        np.array([[1j, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    assert np.allclose(
        cirq.unitary(cirq.CZ00**0),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    assert np.allclose(
        cirq.unitary(cirq.CZ00**-0.5),
        np.array([[-1j, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))


def test_trace_distance():
    foo = sympy.Symbol('foo')
    scz00 = cirq.CZ00**foo
    assert cirq.trace_distance_bound(scz00) == 1.0

    assert cirq.approx_eq(cirq.trace_distance_bound(cirq.CZ00**(1 / 9)),
                          np.sin(np.pi / 18))


def test_phase_by():
    g = cirq.CZPowGate00(exponent=0.25)
    g2 = cirq.phase_by(g, 0.25, 0)
    assert g2 == g


def test_text_diagrams():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    circuit = cirq.Circuit(
        cirq.X(a),
        cirq.Y(a),
        cirq.Z(a),
        cirq.Z(a)**sympy.Symbol('x'),
        cirq.rx(sympy.Symbol('x')).on(a),
        cirq.CZ00(a, b),
    )

    cirq.testing.assert_has_diagram(
        circuit, """
a: ───X───Y───Z───Z^x───Rx(x)───@─────
                                │
b: ─────────────────────────────@00───
""")
    cirq.testing.assert_has_diagram(circuit,
                                    """
a: ---X---Y---Z---Z^x---Rx(x)---@-----
                                |
b: -----------------------------@00---

    """,
                                    use_unicode_characters=False)


def test_czpowgate00_consistent():
    gate = cirq.CZPowGate00(exponent=0.5, global_shift=0.1)
    cirq.testing.assert_implements_consistent_protocols(gate)
