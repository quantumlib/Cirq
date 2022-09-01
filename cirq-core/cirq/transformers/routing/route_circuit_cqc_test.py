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

import cirq
import pytest


@pytest.mark.parametrize(
    "n_qubits, n_moments, op_density, seed",
    [
        (8, size, op_density, seed)
        for size in [50, 100]
        for seed in range(2)
        for op_density in [0.3, 0.5, 0.7]
    ],
)
def test_route_small_circuit_random(n_qubits, n_moments, op_density, seed):
    c_orig = cirq.testing.random_circuit(
        qubits=n_qubits, n_moments=n_moments, op_density=op_density, random_state=seed
    )
    device = cirq.testing.construct_grid_device(4, 4)
    device_graph = device.metadata.nx_graph
    router = cirq.RouteCQC(device_graph)
    c_routed_preserved = router(c_orig, tag_inserted_swaps=True, preserve_moment_structure=True)
    c_routed_efficient = router(c_orig, tag_inserted_swaps=True, preserve_moment_structure=False)
    device.validate_circuit(c_routed_preserved)
    device.validate_circuit(c_routed_efficient)


def test_high_qubit_count():
    c_orig = cirq.testing.random_circuit(qubits=40, n_moments=350, op_density=0.4, random_state=0)
    device = cirq.testing.construct_grid_device(7, 7)
    device_graph = device.metadata.nx_graph
    router = cirq.RouteCQC(device_graph)
    c_routed = router(c_orig)
    device.validate_circuit(c_routed)


def test_multi_qubit_gate_inputs():
    device = cirq.testing.construct_grid_device(4, 4)
    device_graph = device.metadata.nx_graph
    router = cirq.RouteCQC(device_graph)
    q = cirq.LineQubit.range(5)

    invalid_subcircuit_op = cirq.CircuitOperation(
        cirq.Circuit(cirq.X(q[1]), cirq.CCZ(q[0], q[1], q[2]), cirq.Y(q[1])).freeze()
    ).with_tags('<mapped_circuit_op>')
    invalid_circuit = cirq.Circuit(cirq.H(q[0]), cirq.H(q[2]), invalid_subcircuit_op)
    with pytest.raises(
        ValueError, match="Input circuit must only have ops that act on 1 or 2 qubits."
    ):
        router(invalid_circuit, context=cirq.TransformerContext(deep=True))
    with pytest.raises(
        ValueError, match="Input circuit must only have ops that act on 1 or 2 qubits."
    ):
        router(invalid_circuit, context=cirq.TransformerContext(deep=False))

    invalid_circuit = cirq.Circuit(cirq.CCX(q[0], q[1], q[2]))
    with pytest.raises(
        ValueError, match="Input circuit must only have ops that act on 1 or 2 qubits."
    ):
        router(invalid_circuit, context=cirq.TransformerContext(deep=True))
    with pytest.raises(
        ValueError, match="Input circuit must only have ops that act on 1 or 2 qubits."
    ):
        router(invalid_circuit, context=cirq.TransformerContext(deep=False))

    valid_subcircuit_op = cirq.CircuitOperation(
        cirq.Circuit(cirq.X(q[1]), cirq.CZ(q[0], q[1]), cirq.CZ(q[1], q[2]), cirq.Y(q[1])).freeze()
    ).with_tags('<mapped_circuit_op>')
    valid_circuit = cirq.Circuit(cirq.H(q[0]), cirq.H(q[2]), valid_subcircuit_op)
    with pytest.raises(
        ValueError, match="Input circuit must only have ops that act on 1 or 2 qubits."
    ):
        router(invalid_circuit, context=cirq.TransformerContext(deep=False))
    c_routed = router(valid_circuit, context=cirq.TransformerContext(deep=True))
    device.validate_circuit(c_routed)


def test_directed_device():
    device = cirq.testing.construct_ring_device(10, directed=True)
    device_graph = device.metadata.nx_graph
    with pytest.raises(ValueError, match="Device graph must be undirected."):
        cirq.RouteCQC(device_graph)


def test_empty_circuit():
    device = cirq.testing.construct_grid_device(5, 5)
    device_graph = device.metadata.nx_graph
    empty_circuit = cirq.Circuit()
    router = cirq.RouteCQC(device_graph)
    empty_circuit_routed = router(empty_circuit)

    device.validate_circuit(empty_circuit_routed)
    assert len(list(empty_circuit.all_operations())) == len(
        list(empty_circuit_routed.all_operations())
    )


def test_already_valid_circuit():
    device = cirq.testing.construct_ring_device(10)
    device_graph = device.metadata.nx_graph
    valid_circuit = cirq.Circuit(
        cirq.Moment(cirq.CNOT(cirq.LineQubit(i), cirq.LineQubit(i + 1))) for i in range(9)
    )
    hard_coded_mapper = cirq.HardCodedInitialMapper(
        {cirq.LineQubit(i): cirq.LineQubit(i) for i in range(10)}
    )
    router = cirq.RouteCQC(device_graph)
    valid_circuit_routed = router(valid_circuit, initial_mapper=hard_coded_mapper)

    device.validate_circuit(valid_circuit_routed)
    assert len(list(valid_circuit.all_operations())) == len(
        list(valid_circuit_routed.all_operations())
    )
    # TODO: add assert same cirucits using cirq.testing.assert_same_circuits


def test_repr():
    device = cirq.testing.construct_ring_device(10)
    device_graph = device.metadata.nx_graph
    router = cirq.RouteCQC(device_graph)
    cirq.testing.assert_equivalent_repr(router, setup_code='import cirq\nimport networkx as nx')
