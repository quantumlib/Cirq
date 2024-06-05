# Copyright 2024 The Cirq Developers
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

# Checking Consistency


def test_cxswap_consistent():
    gate = cirq.CXSWAPGate()
    cirq.testing.assert_implements_consistent_protocols(gate)


def test_czswap_consistent():
    gate = cirq.CZSWAPGate()
    cirq.testing.assert_implements_consistent_protocols(gate)


# Verifying Unitaries


def test_cxswap_unitary():
    # fmt: off
    np.testing.assert_allclose(
        cirq.unitary(cirq.CXSWAPGate()),
        np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 1, 0, 0],
            ]
        ),
        atol=1e-8,
    )


def test_czswap_unitary():
    # fmt: off
    np.testing.assert_allclose(
        cirq.unitary(cirq.CZSWAPGate()),
        np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, -1],
            ]
        ),
        atol=1e-8,
    )


def test_cswap_circuit():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.CXSWAPGate().on(a, b), cirq.CZSWAPGate().on(a, b))
    cirq.testing.assert_has_diagram(
        c,
        """
0: ───CXSWAPGate───CZSWAPGate───
      │            │
1: ───CXSWAPGate───CZSWAPGate───
    """,
    )


# Verifying __repr__()
def test_cswap_repr():
    assert repr(cirq.CXSWAPGate()) == 'cirq.CXSWAPGate()'
    assert repr(cirq.CZSWAPGate()) == 'cirq.CZSWAPGate()'


# Verifying __eq__()
def test_cswap_eq():
    gate = cirq.CXSWAPGate()
    gate1 = cirq.CXSWAPGate()
    gate2 = cirq.CZSWAPGate()
    gate3 = cirq.CZSWAPGate()
    assert gate == gate1
    assert gate2 == gate3
    assert gate != gate3
    assert gate2 != gate1
