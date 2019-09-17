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


@pytest.mark.parametrize('circuit,device_graph,algo',
                         [(cirq.testing.random_circuit(10, 30, 0.5),
                           ccr.get_grid_device_graph(4, 3), algo)
                          for algo in ccr.ROUTERS
                          for _ in range(5)])
def test_route_circuit(circuit, device_graph, algo):
    swap_network = ccr.route_circuit(circuit, device_graph, algo_name=algo)
    reverse_mapping = {l: p for p, l in swap_network.initial_mapping.items()}
    logical_qubits = sorted(circuit.all_qubits())
    physical_qubits = [reverse_mapping[l] for l in logical_qubits]
    assert set(swap_network.initial_mapping).issubset(physical_qubits)
    assert sorted(swap_network.initial_mapping.values()) == logical_qubits
    assert ccr.ops_are_consistent_with_device_graph(
        swap_network.circuit.all_operations(), device_graph)
    assert ccr.is_valid_routing(circuit, swap_network)


def test_router_bad_args():
    circuit = cirq.Circuit()
    device_graph = ccr.get_linear_device_graph(5)
    with pytest.raises(ValueError):
        ccr.route_circuit(circuit, device_graph)

    algo_name = 'greedy'
    with pytest.raises(ValueError):
        ccr.route_circuit(circuit,
                          device_graph,
                          algo_name=algo_name,
                          router=ccr.ROUTERS[algo_name])

    circuit = cirq.Circuit.from_ops(cirq.CCZ(*cirq.LineQubit.range(3)))
    with pytest.raises(ValueError):
        ccr.route_circuit(circuit, device_graph, algo_name=algo_name)

    circuit = cirq.Circuit.from_ops(
        cirq.CZ(cirq.LineQubit(i), cirq.LineQubit(i + 1)) for i in range(5))
    with pytest.raises(ValueError):
        ccr.route_circuit(circuit, device_graph, algo_name=algo_name)
