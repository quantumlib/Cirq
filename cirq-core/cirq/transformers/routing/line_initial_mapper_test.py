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


def construct_small_circuit():
    return cirq.Circuit(
        [
            cirq.Moment(cirq.CNOT(cirq.NamedQubit('1'), cirq.NamedQubit('3'))),
            cirq.Moment(cirq.CNOT(cirq.NamedQubit('2'), cirq.NamedQubit('3'))),
            cirq.Moment(
                cirq.CNOT(cirq.NamedQubit('4'), cirq.NamedQubit('3')), cirq.X(cirq.NamedQubit('5'))
            ),
        ]
    )


def construct_large_circuit(k: int):
    q = cirq.LineQubit.range(k)
    return cirq.Circuit([cirq.CNOT(q[i], q[i + 1]) for i in range(k - 1)])


# TODO: finish this
def test_large_circuit_grid_device():
    # also call with grid device too small in size for # of qubits (expect failure)
    pass


# TODO: finish this
def test_small_circuit_grid_device():
    # call initial_mapping() twice (expect the same thing)
    pass


@pytest.mark.parametrize(
    "qubits, n_moments, op_density, random_state",
    [
        (10 * size, 20 * size, 0.4, seed)
        for size in range(1, 3)
        for seed in range(10)
        for density in [0.4, 0.5, 0.6]
    ],
)
def test_random_circuits_grid_device(
    qubits: int, n_moments: int, op_density: float, random_state: int
):
    c_orig = cirq.testing.random_circuit(
        qubits=qubits, n_moments=n_moments, op_density=op_density, random_state=random_state
    )
    device = cirq.testing.construct_grid_device(7)
    device_graph = device.metadata.nx_graph
    mapper = cirq.LineInitialMapper(circuit=c_orig, device_graph=device_graph)
    mapping = mapper.initial_mapping()
    c_mapped = c_orig.transform_qubits(mapping)

    # all qubits in the input circuit are placed on the device
    assert set(mapping.keys()) == set(c_orig.all_qubits())

    # the first two moments are executable
    device.validate_circuit(c_mapped[:2])

    # the induced graph of the device on the physical qubits in the map is connected
    assert nx.is_connected(nx.induced_subgraph(device_graph, mapping.values()))


def test_value_equality():
    # equality across device_graphs already tested in mapping_manager_test. Check for equality
    # across circuits
    device_graph = cirq.testing.construct_grid_device(7).metadata.nx_graph
    equals_tester = cirq.testing.EqualsTester()
    small_one = construct_small_circuit()
    small_two = construct_small_circuit()
    equals_tester.add_equality_group(
        cirq.LineInitialMapper(device_graph, small_one),
        cirq.LineInitialMapper(device_graph, small_two),
    )
    equals_tester.add_equality_group(
        cirq.LineInitialMapper(device_graph, construct_large_circuit(30))
    )


def test_str():
    circuit = construct_small_circuit()
    device_graph = cirq.testing.construct_grid_device(7).metadata.nx_graph
    mapper = cirq.LineInitialMapper(device_graph, circuit)
    assert (
        str(mapper)
        == f'cirq.LineInitialMapper(nx.Graph({dict(device_graph.adjacency())}), {repr(circuit)})'
    )


def test_repr():
    circuit = construct_small_circuit()
    device_graph = cirq.testing.construct_grid_device(7).metadata.nx_graph
    mapper = cirq.LineInitialMapper(device_graph, circuit)
    cirq.testing.assert_equivalent_repr(mapper, setup_code='import cirq\nimport networkx as nx')
