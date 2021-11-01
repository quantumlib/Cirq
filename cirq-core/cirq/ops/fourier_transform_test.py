# Copyright 2019 The Cirq Developers
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
import pytest, sympy

import cirq


def test_phase_gradient():
    np.testing.assert_allclose(
        cirq.unitary(cirq.PhaseGradientGate(num_qubits=2, exponent=1)), np.diag([1, 1j, -1, -1j])
    )

    for k in range(4):
        cirq.testing.assert_implements_consistent_protocols(
            cirq.PhaseGradientGate(num_qubits=k, exponent=1)
        )


@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_phase_gradient_symbolic(resolve_fn):
    a = cirq.PhaseGradientGate(num_qubits=2, exponent=0.5)
    b = cirq.PhaseGradientGate(num_qubits=2, exponent=sympy.Symbol('t'))
    assert not cirq.is_parameterized(a)
    assert cirq.is_parameterized(b)
    assert cirq.has_unitary(a)
    assert not cirq.has_unitary(b)
    assert resolve_fn(a, {'t': 0.25}) is a
    assert resolve_fn(b, {'t': 0.5}) == a
    assert resolve_fn(b, {'t': 0.25}) == cirq.PhaseGradientGate(num_qubits=2, exponent=0.25)


def test_str():
    assert str(cirq.PhaseGradientGate(num_qubits=2, exponent=0.5)) == 'Grad[2]^0.5'
    assert str(cirq.PhaseGradientGate(num_qubits=2, exponent=1)) == 'Grad[2]'


def test_phase_gradient_gate_repr():
    a = cirq.PhaseGradientGate(num_qubits=2, exponent=0.5)
    cirq.testing.assert_equivalent_repr(a)


def test_quantum_fourier_transform_gate_repr():
    b = cirq.QuantumFourierTransformGate(num_qubits=2, without_reverse=False)
    cirq.testing.assert_equivalent_repr(b)


def test_pow():
    a = cirq.PhaseGradientGate(num_qubits=2, exponent=0.5)
    assert a ** 0.5 == cirq.PhaseGradientGate(num_qubits=2, exponent=0.25)
    assert a ** sympy.Symbol('t') == cirq.PhaseGradientGate(
        num_qubits=2, exponent=0.5 * sympy.Symbol('t')
    )


def test_qft():
    np.testing.assert_allclose(
        cirq.unitary(cirq.qft(*cirq.LineQubit.range(2))),
        np.array(
            [
                [1, 1, 1, 1],
                [1, 1j, -1, -1j],
                [1, -1, 1, -1],
                [1, -1j, -1, 1j],
            ]
        )
        / 2,
        atol=1e-8,
    )

    np.testing.assert_allclose(
        cirq.unitary(cirq.qft(*cirq.LineQubit.range(2), without_reverse=True)),
        np.array(
            [
                [1, 1, 1, 1],
                [1, -1, 1, -1],
                [1, 1j, -1, -1j],
                [1, -1j, -1, 1j],
            ]
        )
        / 2,
        atol=1e-8,
    )

    np.testing.assert_allclose(
        cirq.unitary(cirq.qft(*cirq.LineQubit.range(4))),
        np.array([[np.exp(2j * np.pi * i * j / 16) for i in range(16)] for j in range(16)]) / 4,
        atol=1e-8,
    )

    np.testing.assert_allclose(
        cirq.unitary(cirq.qft(*cirq.LineQubit.range(2)) ** -1),
        np.array(
            [
                [1, 1, 1, 1],
                [1, -1j, -1, 1j],
                [1, -1, 1, -1],
                [1, 1j, -1, -1j],
            ]
        )
        / 2,
        atol=1e-8,
    )

    for k in range(4):
        for b in [False, True]:
            cirq.testing.assert_implements_consistent_protocols(
                cirq.QuantumFourierTransformGate(num_qubits=k, without_reverse=b)
            )


def test_inverse():
    a, b, c = cirq.LineQubit.range(3)
    assert cirq.qft(a, b, c, inverse=True) == cirq.qft(a, b, c) ** -1
    assert cirq.qft(a, b, c, inverse=True, without_reverse=True) == cirq.inverse(
        cirq.qft(a, b, c, without_reverse=True)
    )


def test_circuit_diagram():
    cirq.testing.assert_has_diagram(
        cirq.Circuit(cirq.decompose_once(cirq.qft(*cirq.LineQubit.range(4)))),
        """
0: ───H───Grad^0.5───────#2─────────────#3─────────────×───
          │              │              │              │
1: ───────@──────────H───Grad^0.5───────#2─────────×───┼───
                         │              │          │   │
2: ──────────────────────@──────────H───Grad^0.5───×───┼───
                                        │              │
3: ─────────────────────────────────────@──────────H───×───
        """,
    )

    cirq.testing.assert_has_diagram(
        cirq.Circuit(cirq.decompose_once(cirq.qft(*cirq.LineQubit.range(4), without_reverse=True))),
        """
0: ───H───Grad^0.5───────#2─────────────#3─────────────
          │              │              │
1: ───────@──────────H───Grad^0.5───────#2─────────────
                         │              │
2: ──────────────────────@──────────H───Grad^0.5───────
                                        │
3: ─────────────────────────────────────@──────────H───
        """,
    )

    cirq.testing.assert_has_diagram(
        cirq.Circuit(
            cirq.qft(*cirq.LineQubit.range(4)), cirq.inverse(cirq.qft(*cirq.LineQubit.range(4)))
        ),
        """
0: ───qft───qft^-1───
      │     │
1: ───#2────#2───────
      │     │
2: ───#3────#3───────
      │     │
3: ───#4────#4───────
        """,
    )
