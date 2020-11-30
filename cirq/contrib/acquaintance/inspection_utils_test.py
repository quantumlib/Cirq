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

from itertools import product, combinations

import pytest

import cirq
import cirq.contrib.acquaintance as cca


@pytest.mark.parametrize('n_qubits, acquaintance_size', product(range(2, 6), range(2, 5)))
def test_get_logical_acquaintance_opportunities(n_qubits, acquaintance_size):
    qubits = cirq.LineQubit.range(n_qubits)
    acquaintance_strategy = cca.complete_acquaintance_strategy(qubits, acquaintance_size)
    initial_mapping = {q: i for i, q in enumerate(qubits)}
    opps = cca.get_logical_acquaintance_opportunities(acquaintance_strategy, initial_mapping)
    assert opps == set(frozenset(s) for s in combinations(range(n_qubits), acquaintance_size))
