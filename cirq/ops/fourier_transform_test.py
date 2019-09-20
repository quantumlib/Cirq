# Copyright 2018 The Cirq Developers
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

import numpy as np

import cirq


def test_phase_gradient():
    np.testing.assert_allclose(
        cirq.unitary(cirq.PhaseGradientGate(num_qubits=2, exponent=1)),
        np.diag([1, 1j, -1, -1j])
    )

    for k in range(4):
        cirq.testing.assert_implements_consistent_protocols(
            cirq.PhaseGradientGate(num_qubits=k, exponent=1))


def test_qft():
    np.testing.assert_allclose(
        cirq.unitary(cirq.QFT(*cirq.LineQubit.range(2))),
        np.array([
            [1, 1, 1, 1],
            [1, 1j, -1, -1j],
            [1, -1, 1, -1],
            [1, -1j, -1, 1j],
        ]) / 2,
        atol=1e-8)

    np.testing.assert_allclose(
        cirq.unitary(cirq.QFT(*cirq.LineQubit.range(2), without_reverse=True)),
        np.array([
            [1, 1, 1, 1],
            [1, -1, 1, -1],
            [1, 1j, -1, -1j],
            [1, -1j, -1, 1j],
        ]) / 2,
        atol=1e-8)

    np.testing.assert_allclose(
        cirq.unitary(cirq.QFT(*cirq.LineQubit.range(4))),
        np.array([
            [np.exp(2j * np.pi * i * j / 16) for i in range(16)]
            for j in range(16)
        ]) / 4,
        atol=1e-8)

    np.testing.assert_allclose(
        cirq.unitary(cirq.QFT(*cirq.LineQubit.range(2))**-1),
        np.array([
            [1, 1, 1, 1],
            [1, -1j, -1, 1j],
            [1, -1, 1, -1],
            [1, 1j, -1, -1j],
        ]) / 2,
        atol=1e-8)

    for k in range(4):
        cirq.testing.assert_implements_consistent_protocols(
            cirq.QuantumFourierTransformGate(num_qubits=k))


def test_circuit_diagram():
    cirq.testing.assert_has_diagram(
        cirq.Circuit.from_ops(
            cirq.decompose_once(cirq.QFT(*cirq.LineQubit.range(4)))),
        """
0: ───H───Grad^0.5───────#2─────────────#3─────────────×───
          │              │              │              │
1: ───────@──────────H───Grad^0.5───────#2─────────×───┼───
                         │              │          │   │
2: ──────────────────────@──────────H───Grad^0.5───×───┼───
                                        │              │
3: ─────────────────────────────────────@──────────H───×───
        """)

    cirq.testing.assert_has_diagram(
        cirq.Circuit.from_ops(
            cirq.decompose_once(cirq.QFT(*cirq.LineQubit.range(4),
                                         without_reverse=True))),
        """
0: ───H───Grad^0.5───────#2─────────────#3─────────────
          │              │              │
1: ───────@──────────H───Grad^0.5───────#2─────────────
                         │              │
2: ──────────────────────@──────────H───Grad^0.5───────
                                        │
3: ─────────────────────────────────────@──────────H───
        """)

    cirq.testing.assert_has_diagram(
        cirq.Circuit.from_ops(
            cirq.QFT(*cirq.LineQubit.range(4)),
            cirq.inverse(cirq.QFT(*cirq.LineQubit.range(4)))),
        """
0: ───QFT───QFT^-1───
      │     │
1: ───#2────#2───────
      │     │
2: ───#3────#3───────
      │     │
3: ───#4────#4───────
        """)
