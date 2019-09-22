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

import numpy as np
import pytest

import cirq
import cirq.contrib.acquaintance as cca
import cirq.contrib.routing as ccr


@pytest.mark.parametrize(
    'circuit,device_graph,algo', [(cirq.testing.random_circuit(
        10, 30, 0.5), ccr.get_grid_device_graph(4, 3), algo)
                                  for algo in ccr.ROUTERS
                                  for _ in range(5)] +
    [(cirq.Circuit(), ccr.get_grid_device_graph(4, 3), 'greedy')])
def test_route_circuit(circuit, device_graph, algo):
    swap_network = ccr.route_circuit(circuit, device_graph, algo_name=algo)
    assert set(swap_network.initial_mapping).issubset(device_graph)
    assert (sorted(swap_network.initial_mapping.values()) == sorted(
        circuit.all_qubits()))
    assert ccr.ops_are_consistent_with_device_graph(
        swap_network.circuit.all_operations(), device_graph)
    assert ccr.is_valid_routing(circuit, swap_network)


@pytest.mark.parametrize(
    'circuit,device_graph,algo,make_bad', [(cirq.testing.random_circuit(
        4, 8, 0.5), ccr.get_grid_device_graph(3, 2), 'greedy', make_bad)
                                           for make_bad in (False, True)
                                           for _ in range(5)] +
    [(cirq.Circuit(), ccr.get_grid_device_graph(3, 2), 'greedy', False)])
def test_route_circuit_via_unitaries(circuit, device_graph, algo, make_bad):
    swap_network = ccr.route_circuit(circuit, device_graph, algo_name=algo)

    logical_qubits = sorted(circuit.all_qubits())
    if len(logical_qubits) < 2:
        return
    reverse_mapping = {l: p for p, l in swap_network.initial_mapping.items()}
    physical_qubits = [reverse_mapping[l] for l in logical_qubits]
    physical_qubits += list(set(device_graph).difference(physical_qubits))
    n_unused_qubits = len(physical_qubits) - len(logical_qubits)

    if make_bad:
        swap_network.circuit += [cirq.CNOT(*physical_qubits[:2])]
    cca.return_to_initial_mapping(swap_network.circuit)

    logical_unitary = circuit.unitary(qubit_order=logical_qubits)
    logical_unitary = np.kron(logical_unitary, np.eye(1 << n_unused_qubits))
    physical_unitary = swap_network.circuit.unitary(qubit_order=physical_qubits)

    assert ccr.is_valid_routing(circuit, swap_network) == (not make_bad)
    assert np.allclose(physical_unitary, logical_unitary) == (not make_bad)


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
