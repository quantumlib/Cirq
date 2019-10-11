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

import numpy as np
import pytest

import cirq
from cirq.contrib.quirk.cells.testing import assert_url_to_circuit_returns
from cirq.contrib.quirk.url_to_circuit import quirk_url_to_circuit


def test_input_rotation_cells():
    with pytest.raises(ValueError, match='classical constant'):
        _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":'
                                 '[["Z^(A/2^n)",{"id":"setA","arg":3}]]}')
    with pytest.raises(ValueError, match="Missing input 'a'"):
        _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":'
                                 '[["X^(A/2^n)"]]}')

    assert_url_to_circuit_returns(
        '{"cols":[["Z^(A/2^n)","inputA2"]]}',
        diagram="""
0: ───Z^(A/2^2)───
      │
1: ───A0──────────
      │
2: ───A1──────────
        """,
        unitary=np.diag([1, 1, 1, 1, 1j**0, 1j**0.5, 1j**1, 1j**1.5]))
    assert_url_to_circuit_returns('{"cols":[["Z^(-A/2^n)","inputA1"]]}',
                                  unitary=np.diag([1, 1, 1, -1j]))

    assert_url_to_circuit_returns(
        '{"cols":[["H"],["X^(A/2^n)","inputA2"],["H"]]}',
        unitary=np.diag([1, 1, 1, 1, 1j**0, 1j**0.5, 1j**1, 1j**1.5]))
    assert_url_to_circuit_returns(
        '{"cols":[["H"],["X^(-A/2^n)","inputA2"],["H"]]}',
        unitary=np.diag([1, 1, 1, 1, 1j**0, 1j**-0.5, 1j**-1, 1j**-1.5]))

    assert_url_to_circuit_returns(
        '{"cols":[["X^-½"],["Y^(A/2^n)","inputA2"],["X^½"]]}',
        unitary=np.diag([1, 1, 1, 1, 1j**0, 1j**0.5, 1j**1, 1j**1.5]))
    assert_url_to_circuit_returns(
        '{"cols":[["X^-½"],["Y^(-A/2^n)","inputA2"],["X^½"]]}',
        unitary=np.diag([1, 1, 1, 1, 1j**0, 1j**-0.5, 1j**-1, 1j**-1.5]))

    assert_url_to_circuit_returns('{"cols":[["•","Z^(A/2^n)","inputA2"]]}',
                                  diagram="""
0: ───@───────────
      │
1: ───Z^(A/2^2)───
      │
2: ───A0──────────
      │
3: ───A1──────────
        """,
                                  unitary=np.diag([1 + 0j] * 13 +
                                                  [1j**0.5, 1j, 1j**1.5]))

    assert_url_to_circuit_returns('{"cols":[["X^(-A/2^n)","inputA2"]]}',
                                  diagram="""
0: ───X^(-A/2^2)───
      │
1: ───A0───────────
      │
2: ───A1───────────
        """)

    assert_url_to_circuit_returns('{"cols":[["•","X^(-A/2^n)","inputA2"]]}',
                                  diagram="""
0: ───@────────────
      │
1: ───X^(-A/2^2)───
      │
2: ───A0───────────
      │
3: ───A1───────────
        """)


def test_input_rotation_cells_repr():
    circuit = quirk_url_to_circuit('http://algassert.com/quirk#circuit='
                                   '{"cols":[["•","X^(-A/2^n)","inputA2"]]}')
    op = circuit[0].operations[0]
    cirq.testing.assert_equivalent_repr(op)
