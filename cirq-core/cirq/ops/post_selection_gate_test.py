# Copyright 2023 The Cirq Developers
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


@pytest.mark.parametrize(
    'controls, expected',
    [([(0,)], [1, 0]), ([(1,)], [0, 1]), ([(0,), (1,)], [np.sqrt(0.5), np.sqrt(0.5)])],
)
def test_basic(controls, expected):
    q = cirq.LineQubit(0)
    c = cirq.Circuit(cirq.H(q), cirq.PostSelectionGate(qid_shape=(2,), controls=controls).on(q))
    sv = cirq.final_state_vector(c)
    assert np.allclose(sv, expected)


@pytest.mark.parametrize(
    'controls, expected',
    [
        ([(0, 0)], [[1, 0], [0, 0]]),
        ([(1, 1)], [[0, 0], [0, 1]]),
        ([(0, 0), (1, 0), (0, 1)], [[1, 0], [0, 0]]),
        ([(0, 0), (1, 0), (0, 1), (1, 1)], [[np.sqrt(0.5), 0], [0, np.sqrt(0.5)]]),
    ],
)
def test_multiple(controls, expected):
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit(
        cirq.H(q0),
        cirq.CX(q0, q1),
        cirq.PostSelectionGate(qid_shape=(2, 2), controls=controls).on(q0, q1),
    )
    sv = cirq.final_state_vector(c).reshape((2, 2))
    assert np.allclose(sv, expected)


@pytest.mark.parametrize(
    'dimension, controls, expected',
    [
        (3, [(0,)], [1, 0, 0]),
        (3, [(1,)], [0, 1, 0]),
        (3, [(2,)], [0, 0, 1]),
        (3, [(0,), (1,), (2,)], [2 / 3, 2 / 3, 1 / 3]),
        (3, [(0,), (1,)], [np.sqrt(0.5), np.sqrt(0.5), 0]),
        (4, [(0,)], [1, 0, 0, 0]),
        (4, [(1,)], [0, 1, 0, 0]),
        (4, [(2,)], [0, 0, 1, 0]),
        (4, [(3,)], [0, 0, 0, 1]),
        (4, [(0,), (1,)], [np.sqrt(0.5), np.sqrt(0.5), 0, 0]),
        (4, [(2,), (3,)], [0, 0, np.sqrt(0.5), np.sqrt(0.5)]),
    ],
)
def test_qudits(dimension, controls, expected):
    q = cirq.LineQid(0, dimension=dimension)
    c = cirq.Circuit(
        cirq.XPowGate(exponent=0.5, dimension=dimension).on(q),
        cirq.PostSelectionGate(qid_shape=(dimension,), controls=controls).on(q),
    )
    sv = cirq.final_state_vector(c)
    assert np.allclose(np.abs(sv), np.array(expected))
