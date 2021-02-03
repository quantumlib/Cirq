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

from multiprocessing import Process

import pytest

import cirq
import cirq.contrib.routing as ccr
from cirq.contrib.routing.greedy import route_circuit_greedily


def test_bad_args():
    """Test zero valued arguments in greedy router."""
    circuit = cirq.testing.random_circuit(4, 2, 0.5, random_state=5)
    device_graph = ccr.get_grid_device_graph(3, 2)
    with pytest.raises(ValueError):
        route_circuit_greedily(circuit, device_graph, max_search_radius=0)

    with pytest.raises(ValueError):
        route_circuit_greedily(circuit, device_graph, max_num_empty_steps=0)


def create_circuit_and_device():
    """Construct a small circuit and a device with line connectivity
    to test the greedy router. This instance hangs router in Cirq 8.2.
    """
    num_qubits = 6
    gate_domain = {cirq.ops.CNOT: 2}
    circuit = cirq.testing.random_circuit(num_qubits, 15, 0.5, gate_domain, random_state=37)
    device_graph = ccr.get_linear_device_graph(num_qubits)
    return circuit, device_graph


def create_hanging_routing_instance(circuit, device_graph):
    """Create a test problem instance."""
    route_circuit_greedily(circuit, device_graph, max_search_radius=2, random_state=1)


def test_router_hanging():
    """Run a separate process and check if greedy router hits timeout (5s)."""
    circuit, device_graph = create_circuit_and_device()
    process = Process(target=create_hanging_routing_instance, args=[circuit, device_graph])
    process.start()
    process.join(timeout=5)
    try:
        assert not process.is_alive(), "Greedy router timeout"
    finally:
        process.terminate()
