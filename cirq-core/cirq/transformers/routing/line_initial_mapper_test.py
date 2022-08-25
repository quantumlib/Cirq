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


def construct_step_circuit(k: int):
    q = cirq.LineQubit.range(k)
    return cirq.Circuit([cirq.CNOT(q[i], q[i + 1]) for i in range(k - 1)])


def test_line_breaking_on_grid_device():
    # tests
    #   -if strategy is able to map into several small lines if fails to map onto one long line
    #   -if # of physical qubits <= # of logical qubits then strategy should succeed

    step_circuit = construct_step_circuit(49)
    device = cirq.testing.construct_grid_device(7, 7)
    device_graph = device.metadata.nx_graph
    mapper = cirq.LineInitialMapper(device_graph)
    mapping = mapper.initial_mapping(step_circuit)

    # all qubits in the input circuit are placed on the device
    assert set(mapping.keys()) == set(step_circuit.all_qubits())

    # the induced graph of the device on the physical qubits in the map is connected
    assert nx.is_connected(nx.induced_subgraph(device_graph, mapping.values()))

    step_circuit = construct_step_circuit(50)
    with pytest.raises(ValueError, match="No available physical qubits left on the device"):
        mapper.initial_mapping(step_circuit)


def test_small_circuit_on_grid_device():
    circuit = construct_small_circuit()
    device_graph = cirq.testing.construct_grid_device(7, 7).metadata.nx_graph
    mapper = cirq.LineInitialMapper(device_graph)
    mapping = mapper.initial_mapping(circuit)

    assert mapper.center == cirq.GridQubit(3, 3)

    expected_circuit = cirq.Circuit(
        [
            cirq.Moment(cirq.CNOT(cirq.GridQubit(1, 3), cirq.GridQubit(2, 3))),
            cirq.Moment(cirq.CNOT(cirq.GridQubit(3, 3), cirq.GridQubit(2, 3))),
            cirq.Moment(
                cirq.CNOT(cirq.GridQubit(2, 2), cirq.GridQubit(2, 3)), cirq.X(cirq.GridQubit(2, 4))
            ),
        ]
    )
    cirq.testing.assert_same_circuits(circuit.transform_qubits(mapping), expected_circuit)


def test_small_circuit_on_ring_device():
    circuit = construct_small_circuit()
    device_graph = cirq.testing.construct_ring_device(10, directed=True).metadata.nx_graph

    mapper = cirq.LineInitialMapper(device_graph)
    mapping = mapper.initial_mapping(circuit)
    assert mapper.center == cirq.LineQubit(0)

    expected_circuit = cirq.Circuit(
        [
            cirq.Moment(cirq.CNOT(cirq.LineQubit(2), cirq.LineQubit(1))),
            cirq.Moment(cirq.CNOT(cirq.LineQubit(0), cirq.LineQubit(1))),
            cirq.Moment(cirq.CNOT(cirq.LineQubit(3), cirq.LineQubit(1)), cirq.X(cirq.LineQubit(4))),
        ]
    )
    cirq.testing.assert_same_circuits(circuit.transform_qubits(mapping), expected_circuit)


@pytest.mark.parametrize(
    "qubits, n_moments, op_density, random_state",
    [
        (5 * size, 20 * size, density, seed)
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
    device = cirq.testing.construct_grid_device(7, 7)
    device_graph = device.metadata.nx_graph
    mapper = cirq.LineInitialMapper(device_graph)
    mapping = mapper.initial_mapping(c_orig)

    assert len(set(mapping.values())) == len(mapping.values())
    assert set(mapping.keys()) == set(c_orig.all_qubits())
    assert nx.is_connected(nx.induced_subgraph(device_graph, mapping.values()))


@pytest.mark.parametrize(
    "qubits, n_moments, op_density, random_state",
    [(49, size, 0.5, seed) for size in [50, 100] for seed in range(3)],
)
def test_large_random_circuits_grid_device(
    qubits: int, n_moments: int, op_density: float, random_state: int
):
    c_orig = cirq.testing.random_circuit(
        qubits=qubits, n_moments=n_moments, op_density=op_density, random_state=random_state
    )
    device = cirq.testing.construct_grid_device(7, 7)
    device_graph = device.metadata.nx_graph
    mapper = cirq.LineInitialMapper(device_graph)
    mapping = mapper.initial_mapping(c_orig)

    assert len(set(mapping.values())) == len(mapping.values())
    assert set(mapping.keys()) == set(c_orig.all_qubits())
    assert nx.is_connected(nx.induced_subgraph(device_graph, mapping.values()))


def test_two_circuits_same_instance():
    circ0, circ1 = (
        cirq.testing.random_circuit(qubits=25, n_moments=50, op_density=0.6, random_state=i)
        for i in [0, 1]
    )
    device = cirq.testing.construct_grid_device(7, 7)
    device_graph = device.metadata.nx_graph
    mapper = cirq.LineInitialMapper(device_graph)

    mapping0 = mapper.initial_mapping(circ0)
    mapping1 = mapper.initial_mapping(circ1)

    assert len(set(mapping0.values())) == len(mapping0.values())
    assert set(mapping0.keys()) == set(circ0.all_qubits())
    assert nx.is_connected(nx.induced_subgraph(device_graph, mapping0.values()))

    assert len(set(mapping1.values())) == len(mapping1.values())
    assert set(mapping1.keys()) == set(circ1.all_qubits())
    assert nx.is_connected(nx.induced_subgraph(device_graph, mapping1.values()))


def test_repr():
    device_graph = cirq.testing.construct_grid_device(7, 7).metadata.nx_graph
    mapper = cirq.LineInitialMapper(device_graph)
    cirq.testing.assert_equivalent_repr(mapper, setup_code='import cirq\nimport networkx as nx')
