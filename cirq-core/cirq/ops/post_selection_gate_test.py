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
    'subspaces, expected',
    [([(0,)], [1, 0]), ([(1,)], [0, 1]), ([(0,), (1,)], [np.sqrt(0.5), np.sqrt(0.5)])],
)
def test_basic(subspaces, expected):
    q = cirq.LineQubit(0)
    c = cirq.Circuit(cirq.H(q), cirq.PostSelectionGate(qid_shape=(2,), subspaces=subspaces).on(q))
    sv = cirq.final_state_vector(c)
    assert np.allclose(sv, expected)


def test_error():
    q = cirq.LineQubit(0)
    c = cirq.Circuit(cirq.PostSelectionGate(qid_shape=(2,), subspaces=[(1,)]).on(q))
    with pytest.raises(ValueError, match='Waveform does not contain any post-selected values'):
        _ = cirq.final_state_vector(c)


def test_repr():
    g = cirq.PostSelectionGate(qid_shape=(2,), subspaces=[(0,)])
    assert repr(g) == 'cirq.PostSelectionGate(qid_shape=(2,), subspaces=((0,),))'


@pytest.mark.parametrize(
    'subspaces, expected',
    [
        ([(0,)], [[1, 0], [0, 0]]),
        ([(1,)], [[0, 0], [0, 1]]),
        ([(0,), (1,)], [[0.5, 0.5], [0.5, 0.5]]),
    ],
)
def test_density_matrix(subspaces, expected):
    q = cirq.LineQubit(0)
    c = cirq.Circuit(cirq.H(q), cirq.PostSelectionGate(qid_shape=(2,), subspaces=subspaces).on(q))
    sv = cirq.final_density_matrix(c)
    assert np.allclose(sv, expected)


@pytest.mark.parametrize(
    'subspaces, expected',
    [
        ([(0, 0)], [[1, 0], [0, 0]]),
        ([(1, 1)], [[0, 0], [0, 1]]),
        ([(0, 0), (1, 0), (0, 1)], [[1, 0], [0, 0]]),
        ([(0, 0), (1, 0), (0, 1), (1, 1)], [[np.sqrt(0.5), 0], [0, np.sqrt(0.5)]]),
    ],
)
def test_multiple_qubits(subspaces, expected):
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit(
        cirq.H(q0),
        cirq.CX(q0, q1),
        cirq.PostSelectionGate(qid_shape=(2, 2), subspaces=subspaces).on(q0, q1),
    )
    sv = cirq.final_state_vector(c).reshape((2, 2))
    assert np.allclose(sv, expected)


@pytest.mark.parametrize(
    'dimension, subspaces, expected',
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
def test_qudits(dimension, subspaces, expected):
    q = cirq.LineQid(0, dimension=dimension)
    c = cirq.Circuit(
        cirq.XPowGate(exponent=0.5, dimension=dimension).on(q),
        cirq.PostSelectionGate(qid_shape=(dimension,), subspaces=subspaces).on(q),
    )
    sv = cirq.final_state_vector(c)
    assert np.allclose(np.abs(sv), expected)
