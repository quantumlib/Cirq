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

# Unit tests: keep N small and VF2 bounded.
_MAX_MATCHES = 1  # stop at first embedding
_TIMEOUT_STEPS = 2_000  # hard cap search work


def construct_star_circuit_5q():
    # Interaction graph edges: (1-3), (2-3), (4-3). Center has degree 3.
    return cirq.Circuit(
        [
            cirq.Moment(cirq.CNOT(cirq.NamedQubit("1"), cirq.NamedQubit("3"))),
            cirq.Moment(cirq.CNOT(cirq.NamedQubit("2"), cirq.NamedQubit("3"))),
            cirq.Moment(cirq.CNOT(cirq.NamedQubit("4"), cirq.NamedQubit("3"))),
            cirq.Moment(cirq.X(cirq.NamedQubit("5"))),
        ]
    )


def construct_path_circuit(k: int):
    q = cirq.LineQubit.range(k)
    return cirq.Circuit(cirq.CNOT(q[i], q[i + 1]) for i in range(k - 1))


def _interaction_graph(circuit: cirq.AbstractCircuit) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(sorted(circuit.all_qubits()))
    for op in circuit.all_operations():
        if cirq.num_qubits(op) != 2:
            continue
        a, b = op.qubits
        if a != b:
            g.add_edge(a, b)
    return g


def _assert_is_monomorphism(
    circuit: cirq.AbstractCircuit, device_graph: nx.Graph, mapping: dict[cirq.Qid, cirq.Qid]
) -> None:
    # Injective + total.
    assert len(set(mapping.values())) == len(mapping.values())
    assert set(mapping.keys()) == set(circuit.all_qubits())

    # Edge-preserving.
    cg = _interaction_graph(circuit)
    dg = device_graph.to_undirected() if nx.is_directed(device_graph) else device_graph
    for u, v in cg.edges:
        pu, pv = mapping[u], mapping[v]
        assert dg.has_edge(pu, pv)


def test_path_embeds_on_small_grid() -> None:
    # Small (<= 10) always embeddable on 4x4 grid, should be very fast.
    circuit = construct_path_circuit(10)
    device = cirq.testing.construct_grid_device(4, 4)  # 16 physical
    g = device.metadata.nx_graph

    mapper = cirq.GraphMonomorphismMapper(g, max_matches=_MAX_MATCHES, timeout_steps=_TIMEOUT_STEPS)
    mapping = mapper.initial_mapping(circuit)

    _assert_is_monomorphism(circuit, g, mapping)
    device.validate_circuit(circuit.transform_qubits(mapping))


def test_star_embeds_on_small_grid() -> None:
    # Degree-3 center requires a physical node with degree >= 3 (grid has it).
    circuit = construct_star_circuit_5q()
    device = cirq.testing.construct_grid_device(4, 4)
    g = device.metadata.nx_graph

    mapper = cirq.GraphMonomorphismMapper(g, max_matches=_MAX_MATCHES, timeout_steps=_TIMEOUT_STEPS)
    mapping = mapper.initial_mapping(circuit)

    _assert_is_monomorphism(circuit, g, mapping)
    device.validate_circuit(circuit.transform_qubits(mapping))


def test_star_fails_on_ring() -> None:
    # Ring max degree is 2, but star requires degree 3 -> impossible.
    circuit = construct_star_circuit_5q()
    device = cirq.testing.construct_ring_device(10, directed=True)
    g = device.metadata.nx_graph

    mapper = cirq.GraphMonomorphismMapper(g, max_matches=_MAX_MATCHES, timeout_steps=500)
    with pytest.raises(ValueError, match="No graph monomorphism embedding found"):
        mapper.initial_mapping(circuit)


def test_path_embeds_on_ring() -> None:
    # A path (max degree 2) should embed on a ring.
    circuit = construct_path_circuit(6)
    device = cirq.testing.construct_ring_device(10, directed=True)
    g = device.metadata.nx_graph

    mapper = cirq.GraphMonomorphismMapper(g, max_matches=_MAX_MATCHES, timeout_steps=_TIMEOUT_STEPS)
    mapping = mapper.initial_mapping(circuit)

    _assert_is_monomorphism(circuit, g, mapping)
    device.validate_circuit(circuit.transform_qubits(mapping))


def test_more_logical_than_physical_fails_fast() -> None:
    # Keep small, but verify the early size check.
    circuit = construct_path_circuit(17)  # 17 logical
    device = cirq.testing.construct_grid_device(4, 4)  # 16 physical
    g = device.metadata.nx_graph

    mapper = cirq.GraphMonomorphismMapper(g, max_matches=_MAX_MATCHES, timeout_steps=_TIMEOUT_STEPS)
    with pytest.raises(ValueError, match="more qubits than the device graph can host"):
        mapper.initial_mapping(circuit)


def test_repr() -> None:
    g = cirq.testing.construct_grid_device(4, 4).metadata.nx_graph
    mapper = cirq.GraphMonomorphismMapper(g, max_matches=123, timeout_steps=456)
    cirq.testing.assert_equivalent_repr(mapper, setup_code="import cirq\nimport networkx as nx")
