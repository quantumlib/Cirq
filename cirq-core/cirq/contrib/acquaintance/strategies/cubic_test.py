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

import pytest

import cirq
import cirq.contrib.acquaintance as cca
import cirq.contrib.acquaintance.strategies.cubic as ccasc


def test_skip_and_wrap_around():
    assert ccasc.skip_and_wrap_around(range(3)) == (0, 2, 1)
    assert ccasc.skip_and_wrap_around(range(4)) == (0, 3, 1, 2)
    assert ccasc.skip_and_wrap_around('abcde') == tuple('aebdc')
    assert ccasc.skip_and_wrap_around('abcdef') == tuple('afbecd')


@pytest.mark.parametrize('n_qubits', range(3, 9))
def test_cubic_acquaintance_strategy(n_qubits):
    qubits = tuple(cirq.LineQubit.range(n_qubits))
    strategy = cca.cubic_acquaintance_strategy(qubits)
    initial_mapping = {q: i for i, q in enumerate(qubits)}
    opps = cca.get_logical_acquaintance_opportunities(strategy, initial_mapping)
    assert set(len(opp) for opp in opps) == set([3])
    expected_opps = set(frozenset(ijk) for ijk in itertools.combinations(range(n_qubits), 3))
    assert opps == expected_opps
