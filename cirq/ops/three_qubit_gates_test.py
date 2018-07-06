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

import cirq


def test_matrix():
    np.testing.assert_allclose(cirq.CCX.matrix(), np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ]))

    np.testing.assert_allclose(cirq.CCZ.matrix(), np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, -1],
    ]))

    np.testing.assert_allclose(cirq.CSWAP.matrix(), np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ]))


def test_str():
    assert str(cirq.CCX) == 'TOFFOLI'
    assert str(cirq.TOFFOLI) == 'TOFFOLI'
    assert str(cirq.CSWAP) == 'FREDKIN'
    assert str(cirq.FREDKIN) == 'FREDKIN'
    assert str(cirq.CCZ) == 'CCZ'


def test_eq():
    a, b, c, d = cirq.LineQubit.range(4)
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.CCZ(a, b, c),
                          cirq.CCZ(a, c, b),
                          cirq.CCZ(b, c, a))
    eq.add_equality_group(cirq.CCZ(a, b, d))
    eq.add_equality_group(cirq.TOFFOLI(a, b, c), cirq.CCX(a, b, c))
    eq.add_equality_group(cirq.TOFFOLI(a, c, b), cirq.TOFFOLI(c, a, b))
    eq.add_equality_group(cirq.TOFFOLI(a, b, d))
    eq.add_equality_group(cirq.CSWAP(a, b, c), cirq.FREDKIN(a, b, c))
    eq.add_equality_group(cirq.CSWAP(b, a, c), cirq.CSWAP(b, c, a))


@pytest.mark.parametrize('gate', [
    cirq.CCX, cirq.CSWAP, cirq.CCZ,
])
def test_decomposition_matches_matrix(gate):
    cirq.testing.assert_allclose_up_to_global_phase(
        gate.matrix(),
        cirq.Circuit.from_ops(
            gate.default_decompose(cirq.LineQubit.range(3))
        ).to_unitary_matrix(),
        atol=1e-8)


@pytest.mark.parametrize('gate', [
    cirq.CCX, cirq.CSWAP, cirq.CCZ,
])
def test_decomposition_respects_locality(gate):
    a = cirq.google.GridQubit(0, 0)
    b = cirq.google.GridQubit(1, 0)
    c = cirq.google.GridQubit(0, 1)

    for x, y, z in itertools.permutations([a, b, c]):
        circuit = cirq.Circuit.from_ops(gate(x, y, z))
        cirq.google.ConvertToXmonGates().optimize_circuit(circuit)
        cirq.google.Foxtail.validate_circuit(circuit)


def test_diagram():
    a, b, c, d = cirq.LineQubit.range(4)
    circuit = cirq.Circuit.from_ops(
        cirq.TOFFOLI(a, b, c),
        cirq.CCX(a, c, b),
        cirq.CCZ(a, d, b),
        cirq.CSWAP(a, c, d),
        cirq.FREDKIN(a, b, c)
    )
    assert circuit.to_text_diagram() == """
0: ───@───@───@───@───@───
      │   │   │   │   │
1: ───@───X───@───┼───×───
      │   │   │   │   │
2: ───X───@───┼───×───×───
              │   │
3: ───────────@───×───────
""".strip()
    assert circuit.to_text_diagram(use_unicode_characters=False) == """
0: ---@---@---@---@------@------
      |   |   |   |      |
1: ---@---X---@---|------swap---
      |   |   |   |      |
2: ---X---@---|---swap---swap---
              |   |
3: -----------@---swap----------
""".strip()
