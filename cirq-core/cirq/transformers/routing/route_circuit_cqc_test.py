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

"""Tests for circuit routing with the RouteCQC transformer."""

from __future__ import annotations

import networkx as nx
import pytest

import cirq


def test_directed_device() -> None:
    """Tests that directed device graphs are now accepted by the router."""
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
    """Tests that directed graphs use Hadamard decomposition for SWAPs on one-way edges."""
    # 1. Create Directed Ring: 0 -> 1 -> 2 ...
    device = cirq.testing.construct_ring_device(10, directed=True)
    device_graph = device.metadata.nx_graph
    router = cirq.RouteCQC(device_graph)

    q = cirq.LineQubit.range(3)

    # 2. Force routing with non-adjacent qubits
    # In a ring: 0->1->2, so 0 and 2 are NOT adjacent
    # This forces the router to insert SWAPs to bring them together
    circuit = cirq.Circuit(cirq.CNOT(q[0], q[2]))

    hard_coded_mapper = cirq.HardCodedInitialMapper({q[i]: q[i] for i in range(3)})
    routed_circuit = router(circuit, initial_mapper=hard_coded_mapper)

    # 3. CRITICAL CHECK: Look for Hadamard gates
    # If your logic works, the output MUST contain H gates to reverse the CNOT.
    ops_list = list(routed_circuit.all_operations())
    has_hadamards = any(op.gate == cirq.H for op in ops_list)

    assert has_hadamards, "Router failed to use Hadamard decomposition for directed SWAP!"

    # 4. Standard validation
    device.validate_circuit(routed_circuit)


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
    """Tests routing of random circuits on a grid device with various parameters."""
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
    """Tests routing of a large circuit with 40 qubits on a 7x7 grid device."""
    c_orig = cirq.testing.random_circuit(qubits=40, n_moments=350, op_density=0.4, random_state=0)
    device = cirq.testing.construct_grid_device(7, 7)
    device_graph = device.metadata.nx_graph
    router = cirq.RouteCQC(device_graph)
    c_routed = router(c_orig)
    device.validate_circuit(c_routed)


def test_multi_qubit_gate_inputs() -> None:
    """Tests that circuits with >2-qubit gates raise appropriate errors."""
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
    """Tests routing of circuits with measurement gates."""
    device = cirq.testing.construct_ring_device(3)
    device_graph = device.metadata.nx_graph
    q = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.MeasurementGate(2).on(q[0], q[2]), cirq.MeasurementGate(3).on(*q))
    hard_coded_mapper = cirq.HardCodedInitialMapper({q[i]: q[i] for i in range(3)})
    router = cirq.RouteCQC(device_graph)
    routed_circuit = router(circuit, initial_mapper=hard_coded_mapper)
    cirq.testing.assert_same_circuits(routed_circuit, circuit)


def test_circuit_with_two_qubit_intermediate_measurement_gate() -> None:
    """Tests routing with intermediate 2-qubit measurement gates."""
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
    """Tests routing with multi-qubit intermediate measurement gates using default keys."""
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
    """Tests that multi-qubit intermediate measurements with custom keys raise errors."""
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
    """Tests routing of circuits with non-unitary operations and global phase gates."""
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
    """Tests that tagged operations maintain their tags through routing."""
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
    """Tests routing of a circuit that is already valid for the device."""
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
    """Tests routing of an empty circuit."""
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
    """Tests that tag_inserted_swaps=True works correctly for directed edges."""
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
    """Tests that reverse-only edges use Hadamard decomposition."""
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
    routed_circuit, _, _ = router.route_circuit(circuit, initial_mapper=hard_coded_mapper)

    # Verify Hadamard gates are present
    ops_list = list(routed_circuit.all_operations())
    has_hadamards = any(op.gate == cirq.H for op in ops_list)
    assert has_hadamards, "Expected Hadamard gates for reverse-only edge decomposition!"


def test_directed_device_with_tag_inserted_swaps_reverse_only() -> None:
    """Tests that tag_inserted_swaps=True works for reverse-only directed edges."""
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


def test_emit_swap_reverse_only_edge_with_tags() -> None:
    """Direct unit test of _emit_swap to cover lines 364-393 (Case C).

    This test directly calls _emit_swap with (q0, q1) arguments where only
    the reverse edge (q1, q0) exists in the directed graph, triggering the
    Case C code path with tag_inserted_swaps=True.
    """
    q0 = cirq.LineQubit(0)
    q1 = cirq.LineQubit(1)

    # Graph with only q1 -> q0 edge (reverse-only from q0's perspective)
    g = nx.DiGraph()
    g.add_edge(q1, q0)

    circuit_ops: list[cirq.Operation] = []

    # Call _emit_swap with (q0, q1) - this should trigger Case C (lines 364-393)
    # because has_edge(q1, q0) is True but has_edge(q0, q1) is False
    cirq.RouteCQC._emit_swap(  # pylint: disable=protected-access
        circuit_ops, q0, q1, g, tag_inserted_swaps=True
    )

    # Verify the decomposition: should be 3 CNOTs + 4 Hadamards, all tagged
    assert len(circuit_ops) == 7, f"Expected 7 operations, got {len(circuit_ops)}"

    # All operations should be tagged with RoutingSwapTag
    tagged_ops = [op for op in circuit_ops if cirq.RoutingSwapTag() in op.tags]
    assert len(tagged_ops) == 7, "All operations should be tagged with RoutingSwapTag"

    # Verify we have both Hadamards and CNOTs
    hadamards = [op for op in circuit_ops if isinstance(op.gate, cirq.HPowGate)]
    cnots = [op for op in circuit_ops if isinstance(op.gate, cirq.CNotPowGate)]

    assert len(hadamards) == 4, f"Expected 4 Hadamard gates, got {len(hadamards)}"
    assert len(cnots) == 3, f"Expected 3 CNOT gates, got {len(cnots)}"

    # Verify the CNOT directions are all (q1, q0) since that's the only available edge
    for cnot in cnots:
        assert cnot.qubits == (q1, q0), f"Expected CNOT(q1, q0), got CNOT{cnot.qubits}"


def test_no_edge_between_qubits_raises_error() -> None:
    """Tests that _emit_swap raises ValueError when qubits have no connecting edge."""
    # Test the _emit_swap method directly since the router has other checks
    # that prevent getting to this error in normal routing
    device_graph = nx.Graph()
    q = cirq.LineQubit.range(3)
    # Create a graph where q[0] and q[2] are not connected
    device_graph.add_edge(q[0], q[1])
    # Note: q[2] is isolated (no edges)

    circuit_ops: list[cirq.Operation] = []

    # This should raise a ValueError about no edge between qubits
    with pytest.raises(ValueError, match="No edge between"):
        cirq.RouteCQC._emit_swap(  # pylint: disable=protected-access
            circuit_ops, q[0], q[2], device_graph, tag_inserted_swaps=False
        )


def test_repr() -> None:
    """Tests the string representation of the RouteCQC transformer."""
    device = cirq.testing.construct_ring_device(10)
    device_graph = device.metadata.nx_graph
    router = cirq.RouteCQC(device_graph)
    cirq.testing.assert_equivalent_repr(router, setup_code='import cirq\nimport networkx as nx')
