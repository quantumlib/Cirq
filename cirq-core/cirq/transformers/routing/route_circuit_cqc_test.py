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

from __future__ import annotations

import networkx as nx
import pytest

import cirq


def test_directed_device() -> None:
    device = cirq.testing.construct_ring_device(10, directed=True)
    device_graph = device.metadata.nx_graph
    # Directed graphs should now be accepted
    router = cirq.RouteCQC(device_graph)
    # Test that we can route a simple circuit on a directed graph
    q = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.CNOT(q[0], q[1]), cirq.CNOT(q[1], q[2]))
    hard_coded_mapper = cirq.HardCodedInitialMapper({q[i]: q[i] for i in range(3)})
    routed_circuit = router(circuit, initial_mapper=hard_coded_mapper)
    device.validate_circuit(routed_circuit)


def test_directed_device_swap_decomposition() -> None:
    # Create a simple directed graph: q0 -> q1 (one-way only)
    device_graph = nx.DiGraph()
    q = cirq.LineQubit.range(2)
    device_graph.add_edge(q[0], q[1])

    router = cirq.RouteCQC(device_graph)

    # A circuit that requires a SWAP to execute (qubits need to be swapped)
    circuit = cirq.Circuit(cirq.CNOT(q[1], q[0]))  # Reverse direction not available

    hard_coded_mapper = cirq.HardCodedInitialMapper({q[0]: q[0], q[1]: q[1]})
    routed_circuit = router(circuit, initial_mapper=hard_coded_mapper, tag_inserted_swaps=True)

    # Expected: Hadamard-based SWAP decomposition followed by CNOT
    # SWAP decomposition for unidirectional edge: CNOT-H⊗H-CNOT-H⊗H-CNOT
    expected = cirq.Circuit(
        cirq.CNOT(q[0], q[1]).with_tags(cirq.RoutingSwapTag()),
        cirq.H(q[0]).with_tags(cirq.RoutingSwapTag()),
        cirq.H(q[1]).with_tags(cirq.RoutingSwapTag()),
        cirq.CNOT(q[0], q[1]).with_tags(cirq.RoutingSwapTag()),
        cirq.H(q[0]).with_tags(cirq.RoutingSwapTag()),
        cirq.H(q[1]).with_tags(cirq.RoutingSwapTag()),
        cirq.CNOT(q[0], q[1]).with_tags(cirq.RoutingSwapTag()),
        cirq.CNOT(q[0], q[1]),  # The original CNOT after swap
    )
    cirq.testing.assert_same_circuits(routed_circuit, expected)


@pytest.mark.parametrize(
    "n_qubits, n_moments, op_density, seed",
    [
        (8, size, op_density, seed)
        for size in [50, 100]
        for seed in range(3)
        for op_density in [0.3, 0.5, 0.7]
    ],
)
def test_route_small_circuit_random(n_qubits, n_moments, op_density, seed) -> None:
    c_orig = cirq.testing.random_circuit(
        qubits=n_qubits, n_moments=n_moments, op_density=op_density, random_state=seed
    )
    device = cirq.testing.construct_grid_device(4, 4)
    device_graph = device.metadata.nx_graph
    router = cirq.RouteCQC(device_graph)
    c_routed, imap, swap_map = router.route_circuit(c_orig, tag_inserted_swaps=True)
    device.validate_circuit(c_routed)
    cirq.testing.assert_circuits_have_same_unitary_given_final_permutation(
        c_routed, c_orig.transform_qubits(imap), swap_map
    )


def test_high_qubit_count() -> None:
    c_orig = cirq.testing.random_circuit(qubits=40, n_moments=350, op_density=0.4, random_state=0)
    device = cirq.testing.construct_grid_device(7, 7)
    device_graph = device.metadata.nx_graph
    router = cirq.RouteCQC(device_graph)
    c_routed = router(c_orig)
    device.validate_circuit(c_routed)


def test_multi_qubit_gate_inputs() -> None:
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


def test_circuit_with_measurement_gates() -> None:
    device = cirq.testing.construct_ring_device(3)
    device_graph = device.metadata.nx_graph
    q = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.MeasurementGate(2).on(q[0], q[2]), cirq.MeasurementGate(3).on(*q))
    hard_coded_mapper = cirq.HardCodedInitialMapper({q[i]: q[i] for i in range(3)})
    router = cirq.RouteCQC(device_graph)
    routed_circuit = router(circuit, initial_mapper=hard_coded_mapper)
    cirq.testing.assert_same_circuits(routed_circuit, circuit)


def test_circuit_with_two_qubit_intermediate_measurement_gate() -> None:
    device = cirq.testing.construct_ring_device(2)
    device_graph = device.metadata.nx_graph
    router = cirq.RouteCQC(device_graph)
    qs = cirq.LineQubit.range(2)
    hard_coded_mapper = cirq.HardCodedInitialMapper({qs[i]: qs[i] for i in range(2)})
    circuit = cirq.Circuit([cirq.Moment(cirq.measure(qs)), cirq.Moment(cirq.H.on_each(qs))])
    routed_circuit = router(
        circuit, initial_mapper=hard_coded_mapper, context=cirq.TransformerContext(deep=True)
    )
    device.validate_circuit(routed_circuit)


def test_circuit_with_multi_qubit_intermediate_measurement_gate_and_with_default_key() -> None:
    device = cirq.testing.construct_ring_device(3)
    device_graph = device.metadata.nx_graph
    router = cirq.RouteCQC(device_graph)
    qs = cirq.LineQubit.range(3)
    hard_coded_mapper = cirq.HardCodedInitialMapper({qs[i]: qs[i] for i in range(3)})
    circuit = cirq.Circuit([cirq.Moment(cirq.measure(qs)), cirq.Moment(cirq.H.on_each(qs))])
    routed_circuit = router(
        circuit, initial_mapper=hard_coded_mapper, context=cirq.TransformerContext(deep=True)
    )
    expected = cirq.Circuit([cirq.Moment(cirq.measure_each(qs)), cirq.Moment(cirq.H.on_each(qs))])
    cirq.testing.assert_same_circuits(routed_circuit, expected)


def test_circuit_with_multi_qubit_intermediate_measurement_gate_with_custom_key() -> None:
    device = cirq.testing.construct_ring_device(3)
    device_graph = device.metadata.nx_graph
    router = cirq.RouteCQC(device_graph)
    qs = cirq.LineQubit.range(3)
    hard_coded_mapper = cirq.HardCodedInitialMapper({qs[i]: qs[i] for i in range(3)})
    circuit = cirq.Circuit(
        [cirq.Moment(cirq.measure(qs, key="test")), cirq.Moment(cirq.H.on_each(qs))]
    )
    with pytest.raises(ValueError):
        _ = router(
            circuit, initial_mapper=hard_coded_mapper, context=cirq.TransformerContext(deep=True)
        )


def test_circuit_with_non_unitary_and_global_phase() -> None:
    device = cirq.testing.construct_ring_device(4)
    device_graph = device.metadata.nx_graph
    q = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        [
            cirq.Moment(cirq.CNOT(q[0], q[1]), cirq.global_phase_operation(-1)),
            cirq.Moment(cirq.CNOT(q[1], q[2])),
            cirq.Moment(cirq.depolarize(0.1, 2).on(q[0], q[2]), cirq.depolarize(0.1).on(q[1])),
        ]
    )
    hard_coded_mapper = cirq.HardCodedInitialMapper({q[i]: q[i] for i in range(3)})
    router = cirq.RouteCQC(device_graph)
    routed_circuit = router(circuit, initial_mapper=hard_coded_mapper)
    expected = cirq.Circuit(
        [
            cirq.Moment(cirq.CNOT(q[0], q[1]), cirq.global_phase_operation(-1)),
            cirq.Moment(cirq.CNOT(q[1], q[2])),
            cirq.Moment(cirq.depolarize(0.1).on(q[1])),
            cirq.Moment(cirq.SWAP(q[0], q[1])),
            cirq.Moment(cirq.depolarize(0.1, 2).on(q[1], q[2])),
        ]
    )
    cirq.testing.assert_same_circuits(routed_circuit, expected)


def test_circuit_with_tagged_ops() -> None:
    device = cirq.testing.construct_ring_device(4)
    device_graph = device.metadata.nx_graph
    q = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        [
            cirq.Moment(cirq.CNOT(q[0], q[1]).with_tags("u")),
            cirq.Moment(cirq.CNOT(q[1], q[2])),
            cirq.Moment(cirq.CNOT(q[0], q[2]).with_tags("u")),
            cirq.Moment(cirq.X(q[0]).with_tags("u")),
            cirq.Moment(cirq.X(q[0]).with_tags("u")),
        ]
    )
    hard_coded_mapper = cirq.HardCodedInitialMapper({q[i]: q[i] for i in range(3)})
    router = cirq.RouteCQC(device_graph)
    routed_circuit = router(circuit, initial_mapper=hard_coded_mapper)
    expected = cirq.Circuit(
        [
            cirq.Moment(cirq.TaggedOperation(cirq.CNOT(q[0], q[1]), 'u')),
            cirq.Moment(cirq.CNOT(q[1], q[2])),
            cirq.Moment(cirq.SWAP(q[0], q[1])),
            cirq.Moment(cirq.TaggedOperation(cirq.CNOT(q[1], q[2]), 'u')),
            cirq.Moment(cirq.TaggedOperation(cirq.X(q[1]), 'u')),
            cirq.Moment(cirq.TaggedOperation(cirq.X(q[1]), 'u')),
        ]
    )
    cirq.testing.assert_same_circuits(routed_circuit, expected)


def test_already_valid_circuit() -> None:
    device = cirq.testing.construct_ring_device(10)
    device_graph = device.metadata.nx_graph
    circuit = cirq.Circuit(
        [cirq.Moment(cirq.CNOT(cirq.LineQubit(i), cirq.LineQubit(i + 1))) for i in range(9)],
        cirq.X(cirq.LineQubit(1)),
    )
    hard_coded_mapper = cirq.HardCodedInitialMapper(
        {cirq.LineQubit(i): cirq.LineQubit(i) for i in range(10)}
    )
    router = cirq.RouteCQC(device_graph)
    routed_circuit = router(circuit, initial_mapper=hard_coded_mapper)

    device.validate_circuit(routed_circuit)
    cirq.testing.assert_same_circuits(routed_circuit, circuit)


def test_empty_circuit() -> None:
    device = cirq.testing.construct_grid_device(5, 5)
    device_graph = device.metadata.nx_graph
    empty_circuit = cirq.Circuit()
    router = cirq.RouteCQC(device_graph)
    empty_circuit_routed = router(empty_circuit)

    device.validate_circuit(empty_circuit_routed)
    assert len(list(empty_circuit.all_operations())) == len(
        list(empty_circuit_routed.all_operations())
    )


def test_directed_device_with_tag_inserted_swaps() -> None:
    # Use a directed ring device
    device = cirq.testing.construct_ring_device(10, directed=True)
    device_graph = device.metadata.nx_graph
    router = cirq.RouteCQC(device_graph)

    q = cirq.LineQubit.range(3)
    # Force routing with non-adjacent qubits to trigger swaps
    circuit = cirq.Circuit(cirq.CNOT(q[0], q[2]))

    hard_coded_mapper = cirq.HardCodedInitialMapper({q[i]: q[i] for i in range(3)})
    routed_circuit, _, _ = router.route_circuit(
        circuit, initial_mapper=hard_coded_mapper, tag_inserted_swaps=True
    )

    # Verify that operations with RoutingSwapTag exist
    ops_list = list(routed_circuit.all_operations())
    tagged_ops = [op for op in ops_list if cirq.RoutingSwapTag() in op.tags]

    # Should have tagged operations from the decomposed swap
    assert len(tagged_ops) > 0, "No operations were tagged with RoutingSwapTag!"

    # Verify Hadamard gates are present and tagged
    tagged_hadamards = [op for op in tagged_ops if op.gate == cirq.H]
    assert len(tagged_hadamards) > 0, "Expected tagged Hadamard gates!"


def test_directed_device_reverse_only_edge() -> None:
    # Create a mixed graph with both bidirectional and reverse-only edges
    device_graph = nx.DiGraph()
    q = cirq.LineQubit.range(4)
    # Create a path: 0<->1->2<-3 (1->2 is forward-only, 3->2 is reverse-only)
    device_graph.add_edges_from(
        [
            (q[0], q[1]),
            (q[1], q[0]),  # bidirectional
            (q[1], q[2]),  # forward-only
            (q[3], q[2]),  # reverse-only (no 2->3 edge)
        ]
    )

    router = cirq.RouteCQC(device_graph)

    # Create a circuit that requires a swap on the reverse-only edge
    # Map qubits so we need to swap on edge 3<-2
    circuit = cirq.Circuit(cirq.CNOT(q[2], q[3]))

    hard_coded_mapper = cirq.HardCodedInitialMapper({q[i]: q[i] for i in range(4)})
    # Need tag_inserted_swaps=True for directionality decomposition
    routed_circuit, _, _ = router.route_circuit(
        circuit, initial_mapper=hard_coded_mapper, tag_inserted_swaps=True
    )

    # Verify Hadamard gates are present
    ops_list = list(routed_circuit.all_operations())
    has_hadamards = any(op.gate == cirq.H for op in ops_list)
    assert has_hadamards, "Expected Hadamard gates for reverse-only edge decomposition!"


def test_directed_device_with_tag_inserted_swaps_reverse_only() -> None:
    # Create a mixed graph with reverse-only edge
    device_graph = nx.DiGraph()
    q = cirq.LineQubit.range(4)
    # Create a path: 0<->1->2<-3
    device_graph.add_edges_from(
        [
            (q[0], q[1]),
            (q[1], q[0]),  # bidirectional
            (q[1], q[2]),  # forward-only
            (q[3], q[2]),  # reverse-only
        ]
    )

    router = cirq.RouteCQC(device_graph)

    # Create a circuit requiring swap on reverse-only edge
    circuit = cirq.Circuit(cirq.CNOT(q[2], q[3]))

    hard_coded_mapper = cirq.HardCodedInitialMapper({q[i]: q[i] for i in range(4)})
    routed_circuit, _, _ = router.route_circuit(
        circuit, initial_mapper=hard_coded_mapper, tag_inserted_swaps=True
    )

    # Verify that operations with RoutingSwapTag exist
    ops_list = list(routed_circuit.all_operations())
    tagged_ops = [op for op in ops_list if cirq.RoutingSwapTag() in op.tags]

    # Should have tagged Hadamards and CNOTs from the decomposed swap
    assert len(tagged_ops) > 0, "No operations were tagged with RoutingSwapTag!"

    # Verify both Hadamards and CNOTs are tagged
    tagged_hadamards = [op for op in tagged_ops if op.gate == cirq.H]
    tagged_cnots = [op for op in tagged_ops if isinstance(op.gate, cirq.CNotPowGate)]

    assert len(tagged_hadamards) > 0, "Expected tagged Hadamard gates!"
    assert len(tagged_cnots) > 0, "Expected tagged CNOT gates!"


def test_directional_swap_decomposition_preserves_unitary() -> None:
    """Test that the directional SWAP decomposition is mathematically equivalent to SWAP."""
    # Create a unidirectional graph
    device_graph = nx.DiGraph()
    q = cirq.LineQubit.range(2)
    device_graph.add_edge(q[0], q[1])  # Only forward edge

    router = cirq.RouteCQC(device_graph)

    # Circuit that needs a SWAP on the unidirectional edge
    circuit = cirq.Circuit(cirq.CNOT(q[1], q[0]))

    hard_coded_mapper = cirq.HardCodedInitialMapper({q[0]: q[0], q[1]: q[1]})
    routed_circuit, initial_map, swap_map = router.route_circuit(
        circuit, initial_mapper=hard_coded_mapper, tag_inserted_swaps=True
    )

    # Verify the routed circuit is mathematically equivalent to the original
    cirq.testing.assert_circuits_have_same_unitary_given_final_permutation(
        routed_circuit, circuit.transform_qubits(initial_map), swap_map
    )


def test_swap_on_qubits_not_in_graph_unchanged() -> None:
    """Test that SWAPs on qubits not in the graph are left unchanged."""
    # Create a graph with only q0 <-> q1 edge
    device_graph = nx.Graph()
    q = cirq.LineQubit.range(3)
    device_graph.add_edge(q[0], q[1])

    router = cirq.RouteCQC(device_graph)

    # Create a circuit with a tagged SWAP on q0-q2 (not in graph)
    # This tests the fallback case in _replace_swaps_with_directional_decomposition
    swap_on_non_edge = cirq.SWAP(q[0], q[2]).with_tags(cirq.RoutingSwapTag())
    circuit = cirq.Circuit(swap_on_non_edge)

    # Directly call the private method to test the fallback path
    result = router._replace_swaps_with_directional_decomposition(circuit, True)

    # The SWAP should be unchanged since there's no edge between q0 and q2
    assert list(result.all_operations()) == [swap_on_non_edge]


def test_repr() -> None:
    device = cirq.testing.construct_ring_device(10)
    device_graph = device.metadata.nx_graph
    router = cirq.RouteCQC(device_graph)
    cirq.testing.assert_equivalent_repr(router, setup_code='import cirq\nimport networkx as nx')
