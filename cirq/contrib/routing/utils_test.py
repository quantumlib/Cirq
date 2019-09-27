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
import pytest

import cirq
import cirq.contrib.routing as ccr
import networkx as nx


def test_ops_are_consistent_with_device_graph():
    device_graph = ccr.get_linear_device_graph(3)
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.ZZ(qubits[0], qubits[2]))
    assert not ccr.ops_are_consistent_with_device_graph(
        circuit.all_operations(), device_graph)
    assert not ccr.ops_are_consistent_with_device_graph(
        [cirq.X(cirq.GridQubit(0, 0))], device_graph)


def test_get_circuit_connectivity():
    a, b, c, d = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(
        cirq.CZ(a, b),
        cirq.CZ(b, c),
        cirq.CZ(c, d),
        cirq.CZ(d, a),
    )
    graph = ccr.get_circuit_connectivity(circuit)
    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 4
    is_planar, _ = nx.check_planarity(graph)
    assert is_planar


def test_get_circuit_connectivity_3q():
    a, b, c = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.CCNOT(a, b, c))
    with pytest.raises(ValueError):
        ccr.get_circuit_connectivity(circuit)
