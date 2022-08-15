# Copyright 2022 The Cirq Developers
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

import networkx as nx
import pytest

import cirq
from cirq.transformers.routing import construct_grid_device
from cirq.transformers.routing import LineInitialMapper


@pytest.mark.parametrize(
    "qubits, n_moments, op_density, random_state",
    [(5*size, 10*size, 0.4, seed) for size in range(1,3) for seed in range(20) for density in [0.4, 0.5, 0.6]],
)
def test_random_circuits_grid_device(qubits: int, n_moments: int, op_density: float, random_state: int):
    c_orig = cirq.testing.random_circuit(
        qubits=qubits,
        n_moments=n_moments,
        op_density=op_density,
        random_state=random_state,
    )
    device = construct_grid_device(7)
    device_graph = device.metadata.nx_graph
    mapper = LineInitialMapper(circuit=c_orig, device_graph=device_graph)
    mapping = mapper.initial_mapping()
    c_mapped = c_orig.transform_qubits(mapping)

    # all qubits in the input circuit are placed on the device
    assert set(mapping.keys()) == set(c_orig.all_qubits())

    # the first two moments are executable 
    device.validate_circuit(c_mapped[:2])

    # the induced graph of the device on the physical qubits in the map is connected
    assert nx.is_connected(nx.induced_subgraph(device_graph, mapping.values()))

