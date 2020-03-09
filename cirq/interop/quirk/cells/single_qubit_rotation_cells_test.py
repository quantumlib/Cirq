# Copyright 2019 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sympy

import cirq
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns


def test_fixed_single_qubit_rotations():
    a, b, c, d = cirq.LineQubit.range(4)

    assert_url_to_circuit_returns(
        '{"cols":[["H","X","Y","Z"]]}',
        cirq.Circuit(cirq.H(a), cirq.X(b), cirq.Y(c), cirq.Z(d)))

    assert_url_to_circuit_returns(
        '{"cols":[["X^½","X^⅓","X^¼"],'
        '["X^⅛","X^⅟₁₆","X^⅟₃₂"],'
        '["X^-½","X^-⅓","X^-¼"],'
        '["X^-⅛","X^-⅟₁₆","X^-⅟₃₂"]]}',
        cirq.Circuit(
            cirq.X(a)**(1 / 2),
            cirq.X(b)**(1 / 3),
            cirq.X(c)**(1 / 4),
            cirq.X(a)**(1 / 8),
            cirq.X(b)**(1 / 16),
            cirq.X(c)**(1 / 32),
            cirq.X(a)**(-1 / 2),
            cirq.X(b)**(-1 / 3),
            cirq.X(c)**(-1 / 4),
            cirq.X(a)**(-1 / 8),
            cirq.X(b)**(-1 / 16),
            cirq.X(c)**(-1 / 32),
        ))

    assert_url_to_circuit_returns(
        '{"cols":[["Y^½","Y^⅓","Y^¼"],'
        '["Y^⅛","Y^⅟₁₆","Y^⅟₃₂"],'
        '["Y^-½","Y^-⅓","Y^-¼"],'
        '["Y^-⅛","Y^-⅟₁₆","Y^-⅟₃₂"]]}',
        cirq.Circuit(
            cirq.Y(a)**(1 / 2),
            cirq.Y(b)**(1 / 3),
            cirq.Y(c)**(1 / 4),
            cirq.Y(a)**(1 / 8),
            cirq.Y(b)**(1 / 16),
            cirq.Y(c)**(1 / 32),
            cirq.Y(a)**(-1 / 2),
            cirq.Y(b)**(-1 / 3),
            cirq.Y(c)**(-1 / 4),
            cirq.Y(a)**(-1 / 8),
            cirq.Y(b)**(-1 / 16),
            cirq.Y(c)**(-1 / 32),
        ))

    assert_url_to_circuit_returns(
        '{"cols":[["Z^½","Z^⅓","Z^¼"],'
        '["Z^⅛","Z^⅟₁₆","Z^⅟₃₂"],'
        '["Z^⅟₆₄","Z^⅟₁₂₈"],'
        '["Z^-½","Z^-⅓","Z^-¼"],'
        '["Z^-⅛","Z^-⅟₁₆"]]}',
        cirq.Circuit(
            cirq.Z(a)**(1 / 2),
            cirq.Z(b)**(1 / 3),
            cirq.Z(c)**(1 / 4),
            cirq.Z(a)**(1 / 8),
            cirq.Z(b)**(1 / 16),
            cirq.Z(c)**(1 / 32),
            cirq.Z(a)**(1 / 64),
            cirq.Z(b)**(1 / 128),
            cirq.Moment([
                cirq.Z(a)**(-1 / 2),
                cirq.Z(b)**(-1 / 3),
                cirq.Z(c)**(-1 / 4),
            ]),
            cirq.Z(a)**(-1 / 8),
            cirq.Z(b)**(-1 / 16),
        ))


def test_dynamic_single_qubit_rotations():
    a, b, c = cirq.LineQubit.range(3)
    t = sympy.Symbol('t')

    # Dynamic single qubit rotations.
    assert_url_to_circuit_returns(
        '{"cols":[["X^t","Y^t","Z^t"],["X^-t","Y^-t","Z^-t"]]}',
        cirq.Circuit(
            cirq.X(a)**t,
            cirq.Y(b)**t,
            cirq.Z(c)**t,
            cirq.X(a)**-t,
            cirq.Y(b)**-t,
            cirq.Z(c)**-t,
        ))
    assert_url_to_circuit_returns(
        '{"cols":[["e^iXt","e^iYt","e^iZt"],["e^-iXt","e^-iYt","e^-iZt"]]}',
        cirq.Circuit(
            cirq.rx(2 * sympy.pi * t).on(a),
            cirq.ry(2 * sympy.pi * t).on(b),
            cirq.rz(2 * sympy.pi * t).on(c),
            cirq.rx(2 * sympy.pi * -t).on(a),
            cirq.ry(2 * sympy.pi * -t).on(b),
            cirq.rz(2 * sympy.pi * -t).on(c),
        ))


def test_formulaic_gates():
    a, b = cirq.LineQubit.range(2)
    t = sympy.Symbol('t')

    assert_url_to_circuit_returns(
        '{"cols":[["X^ft",{"id":"X^ft","arg":"t*t"}]]}',
        cirq.Circuit(
            cirq.X(a)**sympy.sin(sympy.pi * t),
            cirq.X(b)**(t * t),
        ))
    assert_url_to_circuit_returns(
        '{"cols":[["Y^ft",{"id":"Y^ft","arg":"t*t"}]]}',
        cirq.Circuit(
            cirq.Y(a)**sympy.sin(sympy.pi * t),
            cirq.Y(b)**(t * t),
        ))
    assert_url_to_circuit_returns(
        '{"cols":[["Z^ft",{"id":"Z^ft","arg":"t*t"}]]}',
        cirq.Circuit(
            cirq.Z(a)**sympy.sin(sympy.pi * t),
            cirq.Z(b)**(t * t),
        ))
    assert_url_to_circuit_returns(
        '{"cols":[["Rxft",{"id":"Rxft","arg":"t*t"}]]}',
        cirq.Circuit(
            cirq.rx(sympy.pi * t * t).on(a),
            cirq.rx(t * t).on(b),
        ))
    assert_url_to_circuit_returns(
        '{"cols":[["Ryft",{"id":"Ryft","arg":"t*t"}]]}',
        cirq.Circuit(
            cirq.ry(sympy.pi * t * t).on(a),
            cirq.ry(t * t).on(b),
        ))
    assert_url_to_circuit_returns(
        '{"cols":[["Rzft",{"id":"Rzft","arg":"t*t"}]]}',
        cirq.Circuit(
            cirq.rz(sympy.pi * t * t).on(a),
            cirq.rz(t * t).on(b),
        ))
