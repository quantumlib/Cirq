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

import itertools
import random

import numpy as np
import pytest

import cirq
import cirq.contrib.acquaintance as cca
import cirq.contrib.routing as ccr


def random_seed():
    return random.randint(0, 2**32)


@pytest.mark.parametrize(
    'n_moments,algo,circuit_seed,routing_seed',
    [(20, algo, random_seed(), random_seed()) for algo in ccr.ROUTERS for _ in range(5)]
    + [(0, 'greedy', random_seed(), random_seed())]
    + [(10, 'greedy', random_seed(), None)],
)
def test_route_circuit(n_moments, algo, circuit_seed, routing_seed):
    circuit = cirq.testing.random_circuit(10, n_moments, 0.5, random_state=circuit_seed)
    device_graph = ccr.get_grid_device_graph(4, 3)
    swap_network = ccr.route_circuit(
        circuit, device_graph, algo_name=algo, random_state=routing_seed
    )
    assert set(swap_network.initial_mapping).issubset(device_graph)
    assert sorted(swap_network.initial_mapping.values()) == sorted(circuit.all_qubits())
    assert ccr.ops_are_consistent_with_device_graph(
        swap_network.circuit.all_operations(), device_graph
    )
    assert ccr.is_valid_routing(circuit, swap_network)


@pytest.mark.parametrize(
    'algo,seed', [(algo, random_seed()) for algo in ccr.ROUTERS for _ in range(3)]
)
def test_route_circuit_reproducible_with_seed(algo, seed):
    circuit = cirq.testing.random_circuit(8, 20, 0.5, random_state=seed)
    device_graph = ccr.get_grid_device_graph(4, 3)
    wrappers = (lambda s: s, np.random.RandomState)

    swap_networks = []
    for wrapper, _ in itertools.product(wrappers, range(3)):
        swap_network = ccr.route_circuit(
            circuit, device_graph, algo_name=algo, random_state=wrapper(seed)
        )
        swap_networks.append(swap_network)

    eq = cirq.testing.equals_tester.EqualsTester()
    eq.add_equality_group(*swap_networks)


@pytest.mark.parametrize('algo', ccr.ROUTERS.keys())
def test_route_circuit_reproducible_between_runs(algo):
    seed = 23
    circuit = cirq.testing.random_circuit(6, 5, 0.5, random_state=seed)
    device_graph = ccr.get_grid_device_graph(2, 3)

    swap_network = ccr.route_circuit(circuit, device_graph, algo_name=algo, random_state=seed)
    swap_network_str = str(swap_network).lstrip('\n').rstrip()
    expected_swap_network_str = """
               ┌──┐       ┌────┐       ┌──────┐
(0, 0): ───4────Z─────4────@───────4──────────────4───
                           │
(0, 1): ───2────@─────2────┼1↦0────5────@─────────5───
                │          ││           │
(0, 2): ───5────┼─────5────┼0↦1────2────┼iSwap────2───
                │          │            ││
(1, 0): ───3────┼T────3────@───────3────┼┼────────3───
                │                       ││
(1, 1): ───1────@─────1────────────1────X┼────────1───
                                         │
(1, 2): ───0────X─────0────────────0─────iSwap────0───
               └──┘       └────┘       └──────┘
    """.lstrip(
        '\n'
    ).rstrip()
    assert swap_network_str == expected_swap_network_str


@pytest.mark.parametrize(
    'n_moments,algo,seed,make_bad',
    [
        (8, algo, random_seed(), make_bad)
        for algo in ccr.ROUTERS
        for make_bad in (False, True)
        for _ in range(5)
    ]
    + [(0, 'greedy', random_seed(), False)],
)
def test_route_circuit_via_unitaries(n_moments, algo, seed, make_bad):
    circuit = cirq.testing.random_circuit(4, n_moments, 0.5, random_state=seed)
    device_graph = ccr.get_grid_device_graph(3, 2)

    swap_network = ccr.route_circuit(circuit, device_graph, algo_name=algo, random_state=seed)

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
        ccr.route_circuit(circuit, device_graph, algo_name=algo_name, router=ccr.ROUTERS[algo_name])

    circuit = cirq.Circuit(cirq.CCZ(*cirq.LineQubit.range(3)))
    with pytest.raises(ValueError):
        ccr.route_circuit(circuit, device_graph, algo_name=algo_name)

    circuit = cirq.Circuit(cirq.CZ(cirq.LineQubit(i), cirq.LineQubit(i + 1)) for i in range(5))
    with pytest.raises(ValueError):
        ccr.route_circuit(circuit, device_graph, algo_name=algo_name)
