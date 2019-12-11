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

from typing import Callable, Optional

import networkx as nx

from cirq import circuits, protocols
from cirq.contrib.routing.greedy import route_circuit_greedily
from cirq.contrib.routing.swap_network import SwapNetwork

ROUTERS = {
    'greedy': route_circuit_greedily,
}


def route_circuit(circuit: circuits.Circuit,
                  device_graph: nx.Graph,
                  *,
                  algo_name: Optional[str] = None,
                  router: Optional[Callable[..., SwapNetwork]] = None,
                  **kwargs) -> SwapNetwork:
    """Routes a circuit on a given device.

    Args:
        circuit: The circuit to route.
        device_graph: The device's graph, in which each vertex is a qubit and
            each edge indicates the ability to do an operation on those qubits.
        algo_name: The name of a routing algorithm. Must be in ROUTERS.
        router: The function that actually does the routing.
        **kwargs: Arguments to pass to the routing algorithm.

    Exactly one of algo_name and router must be specified.
    """

    if any(protocols.num_qubits(op) > 2 for op in circuit.all_operations()):
        raise ValueError('Can only route circuits with operations that act on'
                         ' at most 2 qubits.')

    if len(list(circuit.all_qubits())) > device_graph.number_of_nodes():
        raise ValueError('Number of logical qubits is greater than number'
                         ' of physical qubits.')

    if not (algo_name is None or router is None):
        raise ValueError('At most one of algo_name or router can be specified.')
    if algo_name is not None:
        router = ROUTERS[algo_name]
    elif router is None:
        raise ValueError(f'No routing algorithm specified.')
    return router(circuit, device_graph, **kwargs)
