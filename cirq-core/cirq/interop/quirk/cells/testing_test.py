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
import pytest

import cirq
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns


def test_assert_url_to_circuit_returns_circuit():
    assert_url_to_circuit_returns(
        '{"cols":[["X"]]}', circuit=cirq.Circuit(cirq.X(cirq.LineQubit(0)))
    )

    with pytest.raises(AssertionError, match='circuit differs'):
        assert_url_to_circuit_returns(
            '{"cols":[["X"]]}', circuit=cirq.Circuit(cirq.Y(cirq.LineQubit(0)))
        )


def test_assert_url_to_circuit_returns_unitary():
    assert_url_to_circuit_returns('{"cols":[["X"]]}', unitary=cirq.unitary(cirq.X))

    with pytest.raises(AssertionError, match='Not equal to tolerance'):
        assert_url_to_circuit_returns('{"cols":[["X"]]}', unitary=np.eye(2))


def test_assert_url_to_circuit_returns_diagram():
    assert_url_to_circuit_returns('{"cols":[["X"]]}', diagram='0: ───X───')

    with pytest.raises(AssertionError, match='text diagram differs'):
        assert_url_to_circuit_returns('{"cols":[["X"]]}', diagram='not even close')


def test_assert_url_to_circuit_returns_maps():
    assert_url_to_circuit_returns('{"cols":[["X"]]}', maps={0: 1})
    assert_url_to_circuit_returns('{"cols":[["X"]]}', maps={0: 1, 1: 0})

    with pytest.raises(AssertionError, match='was mapped to 0b1'):
        assert_url_to_circuit_returns('{"cols":[["X"]]}', maps={0: 0})

    with pytest.raises(AssertionError, match='was mapped to None'):
        assert_url_to_circuit_returns('{"cols":[["H"]]}', maps={0: 0})


def test_assert_url_to_circuit_returns_output_amplitudes_from_quirk():
    assert_url_to_circuit_returns(
        '{"cols":[["X","Z"]]}',
        output_amplitudes_from_quirk=[
            {"r": 0, "i": 0},
            {"r": 1, "i": 0},
            {"r": 0, "i": 0},
            {"r": 0, "i": 0},
        ],
    )

    with pytest.raises(AssertionError, match='Not equal to tolerance'):
        assert_url_to_circuit_returns(
            '{"cols":[["X","Z"]]}',
            output_amplitudes_from_quirk=[
                {"r": 0, "i": 0},
                {"r": 0, "i": 0},
                {"r": 0, "i": 1},
                {"r": 0, "i": 0},
            ],
        )


def test_assert_url_to_circuit_misc():
    a, b = cirq.LineQubit.range(2)

    assert_url_to_circuit_returns(
        '{"cols":[["X","X"],["X"]]}',
        cirq.Circuit(cirq.X(a), cirq.X(b), cirq.X(a)),
        output_amplitudes_from_quirk=[
            {"r": 0, "i": 0},
            {"r": 0, "i": 0},
            {"r": 1, "i": 0},
            {"r": 0, "i": 0},
        ],
    )

    assert_url_to_circuit_returns(
        '{"cols":[["X","X"],["X"]]}', cirq.Circuit(cirq.X(a), cirq.X(b), cirq.X(a))
    )

    with pytest.raises(AssertionError, match='Not equal to tolerance'):
        assert_url_to_circuit_returns(
            '{"cols":[["X","X"],["X"]]}',
            cirq.Circuit(cirq.X(a), cirq.X(b), cirq.X(a)),
            output_amplitudes_from_quirk=[
                {"r": 0, "i": 0},
                {"r": 0, "i": -1},
                {"r": 0, "i": 0},
                {"r": 0, "i": 0},
            ],
        )

    with pytest.raises(AssertionError, match='differs from expected circuit'):
        assert_url_to_circuit_returns(
            '{"cols":[["X","X"],["X"]]}', cirq.Circuit(cirq.X(a), cirq.Y(b), cirq.X(a))
        )
