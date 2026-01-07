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

"""Maps logical to physical qubits by finding a graph monomorphism into the device graph.

This mapper builds an *interaction graph* from the circuit (logical qubits as nodes, and an edge
between two logical qubits if they participate in any 2-qubit operation). It then attempts to find
an injective mapping of logical nodes into physical nodes such that every logical edge maps to a
physical edge (i.e. a subgraph/monomorphism embedding).

If multiple embeddings exist, it chooses the one that (heuristically) is most "central" on the
device by minimizing total distance-to-center and then (tie-break) maximizing total degree.

If no monomorphism exists, it raises ValueError (so a router can fall back to a different strategy).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import networkx as nx

from cirq import protocols, value
from cirq.transformers.routing import initial_mapper

if TYPE_CHECKING:
    import cirq


@value.value_equality
class GraphMonomorphismMapper(initial_mapper.AbstractInitialMapper):
    """Places logical qubits onto physical qubits via graph monomorphism (subgraph embedding)."""

    def __init__(
        self,
        device_graph: nx.Graph,
        *,
        max_matches: int = 5_000,
        timeout_steps: Optional[int] = None,
    ) -> None:
        """Initializes a GraphMonomorphismMapper.

        Args:
            device_graph: Device connectivity graph (physical qubits are nodes). If directed, it is
                treated as undirected for the purposes of placement.
            max_matches: Maximum number of candidate embeddings to consider before choosing the best
                mapping found so far.
            timeout_steps: Optional hard cap on internal iteration steps (additional guardrail).
        """
        # For placement, treat connectivity as undirected adjacency.
        # If you need strict directionality, you'd do a DiGraph monomorphism with edge constraints.
        ug = nx.Graph()
        ug.add_nodes_from(sorted(list(device_graph.nodes(data=True))))
        ug.add_edges_from(sorted(tuple(sorted(e)) for e in device_graph.edges))
        self.device_graph = ug

        # Center is used only as a heuristic scoring anchor.
        # (nx.center returns nodes with minimum eccentricity.)
        self.center = nx.center(self.device_graph)[0]
        self.max_matches = int(max_matches)
        self.timeout_steps = None if timeout_steps is None else int(timeout_steps)

    def _make_circuit_interaction_graph(self, circuit: cirq.AbstractCircuit) -> nx.Graph:
        """Builds the circuit interaction graph from 2-qubit operations."""
        g = nx.Graph()
        logical_qubits = sorted(circuit.all_qubits())
        g.add_nodes_from(logical_qubits)

        for op in circuit.all_operations():
            if protocols.num_qubits(op) != 2:
                continue
            q0, q1 = op.qubits
            if q0 == q1:
                continue
            # Coalesce repeated interactions into a single simple edge.
            g.add_edge(q0, q1)

        return g

    def _score_embedding(
        self,
        logical_to_physical: dict[cirq.Qid, cirq.Qid],
        dist_to_center: dict[cirq.Qid, int],
    ) -> tuple[int, int]:
        """Scores an embedding; lower score is better.

        The score is a tuple used for lexicographic comparison:
            (sum of distances to the device center, -sum of device degrees).

        Args:
            logical_to_physical: Mapping from logical qubits to physical qubits.
            dist_to_center: Precomputed shortest-path distance from each physical qubit to the
                device center.

        Returns:
            A score tuple. Lower is preferred; ties are broken by favoring higher-degree placements.
        """
        total_dist = 0
        total_degree = 0
        for _, pq in logical_to_physical.items():
            total_dist += dist_to_center.get(pq, 10**9)
            total_degree += self.device_graph.degree(pq)
        return (total_dist, -total_degree)

    def initial_mapping(self, circuit: cirq.AbstractCircuit) -> dict[cirq.Qid, cirq.Qid]:
        """Finds an initial mapping by embedding the circuit interaction graph into the device graph.

        Args:
            circuit: The input circuit with logical qubits.

        Returns:
            A dictionary mapping logical qubits in the circuit (keys) to physical qubits on the
            device (values).

        Raises:
            ValueError: If no graph monomorphism embedding exists, or if the circuit has more qubits
                than the device graph can host.
        """
        circuit_g = self._make_circuit_interaction_graph(circuit)

        # Trivial fast path: no qubits.
        if circuit_g.number_of_nodes() == 0:
            return {}

        # If the circuit has more logical qubits than device has physical qubits, impossible.
        if circuit_g.number_of_nodes() > self.device_graph.number_of_nodes():
            raise ValueError("Circuit has more qubits than the device graph can host.")

        # Precompute distances to the device center for scoring.
        dist_to_center = dict(nx.single_source_shortest_path_length(self.device_graph, self.center))

        # NetworkX subgraph isomorphism:
        # GraphMatcher(G_big, G_small).subgraph_isomorphisms_iter()
        # yields mappings: big_node -> small_node.
        matcher = nx.algorithms.isomorphism.GraphMatcher(self.device_graph, circuit_g)

        best_map: Optional[dict[cirq.Qid, cirq.Qid]] = None
        best_score: Optional[tuple[int, int]] = None

        steps = 0
        matches_seen = 0

        for big_to_small in matcher.subgraph_isomorphisms_iter():
            # Optional guardrails.
            steps += 1
            if self.timeout_steps is not None and steps > self.timeout_steps:
                break

            # Invert to get logical -> physical.
            # big_to_small: physical -> logical
            logical_to_physical = {lq: pq for pq, lq in big_to_small.items()}

            # Ensure all logical nodes are mapped (they should be, but be defensive).
            if len(logical_to_physical) != circuit_g.number_of_nodes():
                continue

            score = self._score_embedding(logical_to_physical, dist_to_center)
            if best_score is None or score < best_score:
                best_score = score
                best_map = logical_to_physical

            matches_seen += 1
            if matches_seen >= self.max_matches:
                break

        if best_map is None:
            raise ValueError(
                "No graph monomorphism embedding found for circuit interaction graph "
                "into device graph."
            )

        return best_map

    def _value_equality_values_(self):
        return (
            tuple(self.device_graph.nodes),
            tuple(self.device_graph.edges),
            self.max_matches,
            self.timeout_steps,
        )

    def __repr__(self):
        graph_type = type(self.device_graph).__name__
        return (
            "cirq.GraphMonomorphismMapper("
            f"nx.{graph_type}({dict(self.device_graph.adjacency())}), "
            f"max_matches={self.max_matches}, timeout_steps={self.timeout_steps})"
        )