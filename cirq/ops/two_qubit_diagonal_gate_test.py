# Copyright 2020 The Cirq Developers
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


@pytest.mark.parametrize(
    'gate', ((cirq.TwoQubitDiagonalGate(
        [2, 3, 5, 7]), cirq.TwoQubitDiagonalGate([0, 0, 0, 0]),
              cirq.TwoQubitDiagonalGate([2, 3, 5, sympy.Symbol('a')]),
              cirq.TwoQubitDiagonalGate([0.34, 0.12, 0, 0.96]))))
def test_consistent_protocols(gate):
    cirq.testing.assert_implements_consistent_protocols(gate)


def test_unitary():
    diagonal_angles = [2, 3, 5, 7]
    assert cirq.has_unitary(cirq.TwoQubitDiagonalGate(diagonal_angles))
    np.testing.assert_allclose(
        cirq.unitary(cirq.TwoQubitDiagonalGate(diagonal_angles)),
        np.diag([np.exp(1j * angle) for angle in diagonal_angles]),
        atol=1e-8)


def test_diagram():
    a, b = cirq.LineQubit.range(2)

    diagonal_circuit = cirq.Circuit(
        cirq.TwoQubitDiagonalGate([2, 3, 5, 7])(a, b))
    cirq.testing.assert_has_diagram(
        diagonal_circuit, """
0: ───diag(2, 3, 5, 7)───
      │
1: ───#2─────────────────
""")
    cirq.testing.assert_has_diagram(diagonal_circuit,
                                    """
0: ---diag(2, 3, 5, 7)---
      |
1: ---#2-----------------
""",
                                    use_unicode_characters=False)


def test_diagonal_exponent():
    diagonal_angles = [2, 3, 5, 7]
    diagonal_gate = cirq.TwoQubitDiagonalGate(diagonal_angles)

    sqrt_diagonal_gate = diagonal_gate**.5

    expected_angles = [prime / 2 for prime in diagonal_angles]
    assert cirq.approx_eq(sqrt_diagonal_gate,
                          cirq.TwoQubitDiagonalGate(expected_angles))

    assert cirq.pow(cirq.TwoQubitDiagonalGate(diagonal_angles), "test",
                    None) is None


def test_protocols_mul_not_implemented():
    diagonal_angles = [2, 3, None, 7]
    diagonal_gate = cirq.TwoQubitDiagonalGate(diagonal_angles)
    with pytest.raises(TypeError):
        cirq.protocols.pow(diagonal_gate, 3)


def test_resolve():
    diagonal_angles = [2, 3, 5, 7]
    diagonal_gate = cirq.TwoQubitDiagonalGate(
        diagonal_angles[:2] +
        [sympy.Symbol('a'), sympy.Symbol('b')])
    assert cirq.is_parameterized(diagonal_gate)

    diagonal_gate = cirq.resolve_parameters(diagonal_gate, {'a': 5})
    assert diagonal_gate == cirq.TwoQubitDiagonalGate(diagonal_angles[:3] +
                                                      [sympy.Symbol('b')])
    assert cirq.is_parameterized(diagonal_gate)

    diagonal_gate = cirq.resolve_parameters(diagonal_gate, {'b': 7})
    assert diagonal_gate == cirq.TwoQubitDiagonalGate(diagonal_angles)
    assert not cirq.is_parameterized(diagonal_gate)
