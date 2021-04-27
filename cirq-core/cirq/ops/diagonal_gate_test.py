# Copyright 2021 The Cirq Developers
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

from typing import List
import numpy as np
import pytest
import sympy

import cirq

_candidate_angles: List[float] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]


@pytest.mark.parametrize(
    'gate',
    (
        (
            cirq.DiagonalGate([2, 3, 5, 7]),
            cirq.DiagonalGate([0, 0, 0, 0]),
            cirq.DiagonalGate([2, 3, 5, sympy.Symbol('a')]),
            cirq.DiagonalGate([0.34, 0.12, 0, 0.96]),
            cirq.DiagonalGate(_candidate_angles[:8]),
            cirq.DiagonalGate(_candidate_angles[:16]),
        )
    ),
)
def test_consistent_protocols(gate):
    cirq.testing.assert_implements_consistent_protocols(gate)


@pytest.mark.parametrize('n', [1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_decomposition_unitary(n):
    diagonal_angles = np.random.randn(2 ** n)
    diagonal_gate = cirq.DiagonalGate(diagonal_angles)
    decomposed_circ = cirq.Circuit(cirq.decompose(diagonal_gate(*cirq.LineQubit.range(n))))

    expected_f = [np.exp(1j * angle) for angle in diagonal_angles]
    decomposed_f = cirq.unitary(decomposed_circ).diagonal()

    np.testing.assert_allclose(decomposed_f, expected_f)


@pytest.mark.parametrize('n', [1, 2, 3, 4])
def test_diagonal_exponent(n):
    diagonal_angles = _candidate_angles[: 2 ** n]
    diagonal_gate = cirq.DiagonalGate(diagonal_angles)

    sqrt_diagonal_gate = diagonal_gate ** 0.5

    expected_angles = [prime / 2 for prime in diagonal_angles]
    np.testing.assert_allclose(expected_angles, sqrt_diagonal_gate._diag_angles_radians, atol=1e-8)

    assert cirq.pow(cirq.DiagonalGate(diagonal_angles), "test", None) is None


@pytest.mark.parametrize('n', [1, 2, 3, 4])
def test_decomposition_diagonal_exponent(n):
    diagonal_angles = np.random.randn(2 ** n)
    diagonal_gate = cirq.DiagonalGate(diagonal_angles)
    sqrt_diagonal_gate = diagonal_gate ** 0.5
    decomposed_circ = cirq.Circuit(cirq.decompose(sqrt_diagonal_gate(*cirq.LineQubit.range(n))))

    expected_f = [np.exp(1j * angle / 2) for angle in diagonal_angles]
    decomposed_f = cirq.unitary(decomposed_circ).diagonal()

    np.testing.assert_allclose(decomposed_f, expected_f)


def test_decomposition_with_parameterization():
    diagonal_gate = cirq.DiagonalGate([2, 3, 5, sympy.Symbol('a')])
    op = diagonal_gate(*cirq.LineQubit.range(2))

    # We do not support the decomposition of parameterized case yet.
    # So cirq.decompose should do nothing.
    assert cirq.decompose(op) == [op]


def test_diagram():
    a, b, c, d = cirq.LineQubit.range(4)

    diagonal_circuit = cirq.Circuit(cirq.DiagonalGate(_candidate_angles[:16])(a, b, c, d))
    cirq.testing.assert_has_diagram(
        diagonal_circuit,
        """
0: ───diag(2, 3, ..., 47, 53)───
      │
1: ───#2────────────────────────
      │
2: ───#3────────────────────────
      │
3: ───#4────────────────────────
""",
    )

    diagonal_circuit = cirq.Circuit(cirq.DiagonalGate(_candidate_angles[:8])(a, b, c))
    cirq.testing.assert_has_diagram(
        diagonal_circuit,
        """
0: ───diag(2, 3, ..., 17, 19)───
      │
1: ───#2────────────────────────
      │
2: ───#3────────────────────────
""",
    )

    diagonal_circuit = cirq.Circuit(cirq.DiagonalGate(_candidate_angles[:4])(a, b))
    cirq.testing.assert_has_diagram(
        diagonal_circuit,
        """
0: ───diag(2, 3, 5, 7)───
      │
1: ───#2─────────────────
""",
    )


@pytest.mark.parametrize('n', [1, 2, 3, 4])
def test_unitary(n):
    diagonal_angles = _candidate_angles[: 2 ** n]
    assert cirq.has_unitary(cirq.DiagonalGate(diagonal_angles))
    np.testing.assert_allclose(
        cirq.unitary(cirq.DiagonalGate(diagonal_angles)).diagonal(),
        [np.exp(1j * angle) for angle in diagonal_angles],
        atol=1e-8,
    )


@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_resolve(resolve_fn):
    diagonal_angles = [2, 3, 5, 7, 11, 13, 17, 19]
    diagonal_gate = cirq.DiagonalGate(diagonal_angles[:6] + [sympy.Symbol('a'), sympy.Symbol('b')])
    assert cirq.is_parameterized(diagonal_gate)

    diagonal_gate = resolve_fn(diagonal_gate, {'a': 17})
    assert diagonal_gate == cirq.DiagonalGate(diagonal_angles[:7] + [sympy.Symbol('b')])
    assert cirq.is_parameterized(diagonal_gate)

    diagonal_gate = resolve_fn(diagonal_gate, {'b': 19})
    assert diagonal_gate == cirq.DiagonalGate(diagonal_angles)
    assert not cirq.is_parameterized(diagonal_gate)
