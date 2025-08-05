# Copyright 2025 The Cirq Developers
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

_PAULIS = [cirq.I, cirq.X, cirq.Y, cirq.Z]


def _random_probs(n: int, seed: int | None = None):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        probs = rng.random((4, 4))
        probs /= probs.sum()
        yield probs


@pytest.mark.parametrize('probs', _random_probs(3, 0))
def test_pauli_insertion_with_probabilities(probs):
    c = cirq.Circuit(cirq.ZZ(*cirq.LineQubit.range(2)) ** 0.324)
    transformer = cirq.transformers.PauliInsertionTransformer(cirq.ZZPowGate, probs)
    count = np.zeros((4, 4))
    for _ in range(100):
        nc = transformer(c)
        assert len(nc) == 2
        u, v = nc[0]
        i = _PAULIS.index(u.gate)
        j = _PAULIS.index(v.gate)
        count[i, j] += 1
    count = count / count.sum()
    np.testing.assert_allclose(count, probs, atol=0.1)


@pytest.mark.parametrize('probs', _random_probs(3, 0))
def test_pauli_insertion_with_probabilities_doesnot_create_moment(probs):
    c = cirq.Circuit.from_moments([], [cirq.ZZ(*cirq.LineQubit.range(2)) ** 0.324])
    transformer = cirq.transformers.PauliInsertionTransformer(cirq.ZZPowGate, probs)
    count = np.zeros((4, 4))
    for _ in range(100):
        nc = transformer(c)
        assert len(nc) == 2
        u, v = nc[0]
        i = _PAULIS.index(u.gate)
        j = _PAULIS.index(v.gate)
        count[i, j] += 1
    count = count / count.sum()
    np.testing.assert_allclose(count, probs, atol=0.1)
