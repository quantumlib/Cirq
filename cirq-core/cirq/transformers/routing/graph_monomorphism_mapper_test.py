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

from collections.abc import Iterator
from typing import cast

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
    circuit = construct_path_circuit(10)
    device = cirq.testing.construct_grid_device(4, 4)  # 16 physical
    g = device.metadata.nx_graph

    mapper = cirq.GraphMonomorphismMapper(g, max_matches=_MAX_MATCHES, timeout_steps=_TIMEOUT_STEPS)
    mapping = mapper.initial_mapping(circuit)

    _assert_is_monomorphism(circuit, g, mapping)
    device.validate_circuit(circuit.transform_qubits(mapping))


def test_star_embeds_on_small_grid() -> None:
    circuit = construct_star_circuit_5q()
    device = cirq.testing.construct_grid_device(4, 4)
    g = device.metadata.nx_graph

    mapper = cirq.GraphMonomorphismMapper(g, max_matches=_MAX_MATCHES, timeout_steps=_TIMEOUT_STEPS)
    mapping = mapper.initial_mapping(circuit)

    _assert_is_monomorphism(circuit, g, mapping)
    device.validate_circuit(circuit.transform_qubits(mapping))


def test_star_fails_on_ring() -> None:
    circuit = construct_star_circuit_5q()
    device = cirq.testing.construct_ring_device(10, directed=True)
    g = device.metadata.nx_graph

    mapper = cirq.GraphMonomorphismMapper(g, max_matches=_MAX_MATCHES, timeout_steps=500)
    with pytest.raises(ValueError, match="No graph monomorphism embedding found"):
        mapper.initial_mapping(circuit)


def test_path_embeds_on_ring() -> None:
    circuit = construct_path_circuit(6)
    device = cirq.testing.construct_ring_device(10, directed=True)
    g = device.metadata.nx_graph

    mapper = cirq.GraphMonomorphismMapper(g, max_matches=_MAX_MATCHES, timeout_steps=_TIMEOUT_STEPS)
    mapping = mapper.initial_mapping(circuit)

    _assert_is_monomorphism(circuit, g, mapping)
    device.validate_circuit(circuit.transform_qubits(mapping))


def test_more_logical_than_physical_fails_fast() -> None:
    circuit = construct_path_circuit(17)  # 17 logical
    device = cirq.testing.construct_grid_device(4, 4)  # 16 physical
    g = device.metadata.nx_graph

    mapper = cirq.GraphMonomorphismMapper(g, max_matches=_MAX_MATCHES, timeout_steps=_TIMEOUT_STEPS)
    with pytest.raises(ValueError, match="more qubits than the device graph can host"):
        mapper.initial_mapping(circuit)


def test_empty_circuit_returns_empty_mapping() -> None:
    g = cirq.testing.construct_grid_device(2, 2).metadata.nx_graph
    mapper = cirq.GraphMonomorphismMapper(g, max_matches=_MAX_MATCHES, timeout_steps=_TIMEOUT_STEPS)
    assert mapper.initial_mapping(cirq.Circuit()) == {}


def test_make_interaction_graph_skips_self_edges() -> None:
    g = cirq.testing.construct_grid_device(2, 2).metadata.nx_graph
    mapper = cirq.GraphMonomorphismMapper(g, max_matches=_MAX_MATCHES, timeout_steps=_TIMEOUT_STEPS)

    class TwoQubitSelfOp(cirq.Operation):
        def __init__(self, q: cirq.Qid) -> None:
            self._q = q

        @property
        def qubits(self) -> tuple[cirq.Qid, cirq.Qid]:
            return (self._q, self._q)

        def with_qubits(self, *new_qubits: cirq.Qid) -> "TwoQubitSelfOp":
            assert len(new_qubits) == 1
            return TwoQubitSelfOp(new_qubits[0])

        def _num_qubits_(self) -> int:
            return 2

    q = cirq.NamedQubit("q")
    a, b = cirq.NamedQubit("a"), cirq.NamedQubit("b")

    ops = [TwoQubitSelfOp(q), cirq.CZ(a, b)]
    qubits = {q, a, b}

    class FakeCircuit:
        def all_qubits(self):
            return qubits

        def all_operations(self):
            return iter(ops)

    # cover with_qubits for incremental coverage
    _ = TwoQubitSelfOp(q).with_qubits(q)

    cg = mapper._make_circuit_interaction_graph(cast(cirq.AbstractCircuit, FakeCircuit()))

    assert a in cg.nodes and b in cg.nodes and q in cg.nodes
    assert cg.has_edge(a, b)
    assert not cg.has_edge(q, q)
    assert cg.degree(q) == 0


def test_timeout_steps_breaks_out_and_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Forces coverage of:
    # if self.timeout_steps is not None and steps > self.timeout_steps: break
    g = cirq.testing.construct_grid_device(2, 2).metadata.nx_graph
    mapper = cirq.GraphMonomorphismMapper(g, max_matches=10_000, timeout_steps=0)

    circuit = construct_path_circuit(2)

    class FakeMatcher:
        def subgraph_isomorphisms_iter(self) -> Iterator[dict[cirq.Qid, cirq.Qid]]:
            # Infinite-ish generator; timeout should stop it immediately.
            while True:
                yield {}

    monkeypatch.setattr(
        nx.algorithms.isomorphism, "GraphMatcher", lambda *_args, **_kw: FakeMatcher()
    )

    with pytest.raises(ValueError, match="No graph monomorphism embedding found"):
        mapper.initial_mapping(circuit)


def test_defensive_incomplete_mapping_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    # Covers:
    # if len(logical_to_physical) != circuit_g.number_of_nodes(): continue
    g = cirq.testing.construct_grid_device(2, 2).metadata.nx_graph
    mapper = cirq.GraphMonomorphismMapper(g, max_matches=10, timeout_steps=_TIMEOUT_STEPS)

    circuit = construct_path_circuit(2)  # 2 logical nodes

    class FakeMatcher:
        def subgraph_isomorphisms_iter(self) -> Iterator[dict[cirq.Qid, cirq.Qid]]:
            # physical -> logical mapping with only 1 logical node
            # -> after inversion it's incomplete
            yield {cirq.LineQubit(0): cirq.LineQubit(0)}

    monkeypatch.setattr(
        nx.algorithms.isomorphism, "GraphMatcher", lambda *_args, **_kw: FakeMatcher()
    )

    with pytest.raises(ValueError, match="No graph monomorphism embedding found"):
        mapper.initial_mapping(circuit)


def test_score_embedding_uses_default_large_distance() -> None:
    g = cirq.testing.construct_grid_device(2, 2).metadata.nx_graph
    mapper = cirq.GraphMonomorphismMapper(g, max_matches=_MAX_MATCHES, timeout_steps=_TIMEOUT_STEPS)

    # Pick an actual physical node from the graph.
    pq = next(iter(mapper.device_graph.nodes))
    lq = cirq.NamedQubit("lq")

    # dist_to_center missing pq -> should fall back to 10**9.
    score = mapper._score_embedding({lq: pq}, dist_to_center={})
    assert score[0] >= 10**9


def test_value_equality_values_and_repr() -> None:
    g = cirq.testing.construct_grid_device(2, 2).metadata.nx_graph
    mapper = cirq.GraphMonomorphismMapper(g, max_matches=123, timeout_steps=456)

    # Covers _value_equality_values_ explicitly.
    vals = mapper._value_equality_values_()
    assert vals[2] == 123
    assert vals[3] == 456

    # Covers __repr__ (and keeps the existing equivalent repr check).
    cirq.testing.assert_equivalent_repr(mapper, setup_code="import cirq\nimport networkx as nx")
