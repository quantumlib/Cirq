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

import pytest

import cirq
import cirq.contrib.acquaintance as cca


@pytest.mark.parametrize(
    'circuit_dag,sorted_nodes',
    [(dag, cca.random_topological_sort(dag)) for dag in [
        cirq.CircuitDag.from_circuit(cirq.testing.random_circuit(10, 10, 0.5))
        for _ in range(5)
    ] for _ in range(5)])
def test_topological_sort(circuit_dag, sorted_nodes):
    sorted_nodes = list(sorted_nodes)
    assert cca.is_topologically_sorted(circuit_dag,
                                       (node.val for node in sorted_nodes))

    assert not cca.is_topologically_sorted(
        circuit_dag, (node.val for node in sorted_nodes[:-1]))

    assert not cca.is_topologically_sorted(
        circuit_dag, (node.val for node in sorted_nodes + sorted_nodes[:2]))

    v, w = next(iter(circuit_dag.edges))
    i = sorted_nodes.index(v)
    j = sorted_nodes.index(w, i + 1)
    sorted_nodes[i], sorted_nodes[j] = sorted_nodes[j], sorted_nodes[j]

    assert cca.is_topologically_sorted(
        circuit_dag, (node.val for node in sorted_nodes)) == (v.val == w.val)
