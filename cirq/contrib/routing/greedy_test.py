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
import networkx as nx

import cirq
import cirq.contrib.routing as ccr
from cirq.contrib.routing.greedy import route_circuit_greedily


def test_bad_args():
    circuit = cirq.testing.random_circuit(4, 2, 0.5, random_state=5)
    device_graph = ccr.get_grid_device_graph(3, 2)
    with pytest.raises(ValueError):
        route_circuit_greedily(circuit, device_graph, max_search_radius=0)

    with pytest.raises(ValueError):
        route_circuit_greedily(circuit, device_graph, max_num_empty_steps=0)


def test_random_choice():
    x, y, z = (cirq.NamedQubit(s) for s in 'xyz')
    a, b = (cirq.NamedQubit(s) for s in 'ab')
    circuit = cirq.Circuit(cirq.CZ(a, b))
    device_graph = nx.Graph([(x, y), (y, z)])
    initial_mapping = {x: a, z: b}
    route_circuit_greedily(circuit,
                           device_graph,
                           initial_mapping=initial_mapping,
                           max_search_radius=1)


def test_loop_break():
    pass
