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

import itertools
from typing import (
    Callable,
    cast,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TYPE_CHECKING,
)

import numpy as np
import networkx as nx

from cirq import circuits, ops, value
import cirq.contrib.acquaintance as cca
from cirq.contrib.routing.initialization import get_initial_mapping
from cirq.contrib.routing.swap_network import SwapNetwork
from cirq.contrib.routing.utils import get_time_slices, ops_are_consistent_with_device_graph

if TYPE_CHECKING:
    import cirq

SWAP = cca.SwapPermutationGate()
QidPair = Tuple[ops.Qid, ops.Qid]


def route_circuit_greedily(
    circuit: circuits.Circuit, device_graph: nx.Graph, **kwargs
) -> SwapNetwork:
    """Greedily routes a circuit on a given device.

    Alternates between heuristically picking a few SWAPs to change the mapping
    and applying all logical operations possible given the new mapping, until
    all logical operations have been applied.

    The SWAP selection heuristic is as follows. In every iteration, the
    remaining two-qubit gates are partitioned into time slices. (See
    utils.get_time_slices for details.) For each set of candidate SWAPs, the new
    mapping is computed. For each time slice and every two-qubit gate therein,
    the distance of the two logical qubits in the device graph under the new
    mapping is calculated. A candidate set 'S' of SWAPs is taken out of
    consideration if for some other set 'T' there is a time slice such that all
    of the distances for 'T' are at most those for 'S' (and they are not all
    equal).

    If more than one candidate remains, the size of the set of SWAPs considered
    is increased by one and the process is repeated. If after considering SWAP
    sets of size up to 'max_search_radius', more than one candidate remains,
    then the pairs of qubits in the first time slice are considered, and those
    farthest away under the current mapping are brought together using SWAPs
    using a shortest path in the device graph.

    Args:
        circuit: The circuit to route.
        device_graph: The device's graph, in which each vertex is a qubit
            and each edge indicates the ability to do an operation on those
            qubits.
        **kwargs: Further keyword args, including
            max_search_radius: The maximum number of disjoint device edges to
                consider routing on.
            max_num_empty_steps: The maximum number of swap sets to apply
                without allowing a new logical operation to be performed.
            initial_mapping: The initial mapping of physical to logical qubits
                to use. Defaults to a greedy initialization.
            can_reorder: A predicate that determines if two operations may be
                reordered.
            random_state: Random state or random state seed.
    """

    router = _GreedyRouter(circuit, device_graph, **kwargs)
    router.route()

    swap_network = router.swap_network
    swap_network.circuit = circuits.Circuit(swap_network.circuit.all_operations())
    return swap_network


class _GreedyRouter:
    """Keeps track of the state of a greedy circuit routing procedure."""

    def __init__(
        self,
        circuit,
        device_graph: nx.Graph,
        *,
        max_search_radius: int = 1,
        max_num_empty_steps: int = 5,
        initial_mapping: Optional[Dict[ops.Qid, ops.Qid]] = None,
        can_reorder: Callable[
            [ops.Operation, ops.Operation], bool
        ] = circuits.circuit_dag._disjoint_qubits,
        random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
    ):

        self.prng = value.parse_random_state(random_state)

        self.device_graph = device_graph
        self.physical_distances: Dict[QidPair, int] = {
            (a, b): d
            for a, neighbor_distances in nx.shortest_path_length(device_graph)
            for b, d in neighbor_distances.items()
        }

        self.remaining_dag = circuits.CircuitDag.from_circuit(circuit, can_reorder=can_reorder)
        self.logical_qubits = list(self.remaining_dag.all_qubits())
        self.physical_qubits = list(self.device_graph.nodes)
        self.edge_sets: Dict[int, List[Sequence[QidPair]]] = {}

        self.physical_ops: List[ops.Operation] = []

        self.set_initial_mapping(initial_mapping)

        if max_search_radius < 1:
            raise ValueError('max_search_radius must be a positive integer.')
        self.max_search_radius = max_search_radius

        if max_num_empty_steps < 1:
            raise ValueError('max_num_empty_steps must be a positive integer.')
        self.max_num_empty_steps = max_num_empty_steps

    def get_edge_sets(self, edge_set_size: int) -> Iterable[Sequence[QidPair]]:
        """Returns matchings of the device graph of a given size."""
        if edge_set_size not in self.edge_sets:
            self.edge_sets[edge_set_size] = [
                cast(Sequence[QidPair], edge_set)
                for edge_set in itertools.combinations(self.device_graph.edges, edge_set_size)
                if all(set(e).isdisjoint(f) for e, f in itertools.combinations(edge_set, 2))
            ]
        return self.edge_sets[edge_set_size]

    def log_to_phys(self, *qubits: 'cirq.Qid') -> Iterable[ops.Qid]:
        """Returns an iterator over the physical qubits mapped to by the given
        logical qubits."""
        return (self._log_to_phys[q] for q in qubits)

    def phys_to_log(self, *qubits: 'cirq.Qid') -> Iterable[Optional[ops.Qid]]:
        """Returns an iterator over the logical qubits that map to the given
        physical qubits."""
        return (self._phys_to_log[q] for q in qubits)

    def apply_swap(self, *physical_edges: QidPair):
        """Applies SWAP on the given edges."""
        self.update_mapping(*physical_edges)
        self.physical_ops += [SWAP(*e) for e in physical_edges]

    def update_mapping(self, *physical_edges: QidPair):
        """Updates the mapping in accordance with SWAPs on the given physical
        edges."""
        for physical_edge in physical_edges:
            old_logical_edge = tuple(self.phys_to_log(*physical_edge))
            new_logical_edge = old_logical_edge[::-1]
            for p, l in zip(physical_edge, new_logical_edge):
                self._phys_to_log[p] = l
                if l is not None:
                    self._log_to_phys[l] = p

    def set_initial_mapping(self, initial_mapping: Optional[Dict[ops.Qid, ops.Qid]] = None):
        """Sets the internal state according to an initial mapping.

        Args:
            initial_mapping: The mapping to use. If not given, one is found
                greedily.
        """

        if initial_mapping is None:
            time_slices = get_time_slices(self.remaining_dag)
            if not time_slices:
                initial_mapping = dict(zip(self.device_graph, self.logical_qubits))
            else:
                logical_graph = time_slices[0]
                logical_graph.add_nodes_from(self.logical_qubits)
                initial_mapping = get_initial_mapping(logical_graph, self.device_graph, self.prng)
        self.initial_mapping = initial_mapping
        self._phys_to_log = {q: initial_mapping.get(q) for q in self.physical_qubits}
        self._log_to_phys = {l: p for p, l in self._phys_to_log.items() if l is not None}
        self._assert_mapping_consistency()

    def _assert_mapping_consistency(self):
        assert sorted(self._log_to_phys) == sorted(self.logical_qubits)
        assert sorted(self._phys_to_log) == sorted(self.physical_qubits)
        for l in self._log_to_phys:
            assert l == self._phys_to_log[self._log_to_phys[l]]

    def acts_on_nonadjacent_qubits(self, op: ops.Operation) -> bool:
        if len(op.qubits) == 1:
            return False
        return tuple(self.log_to_phys(*op.qubits)) not in self.device_graph.edges

    def apply_possible_ops(self) -> int:
        """Applies all logical operations possible given the current mapping."""
        nodes = list(
            self.remaining_dag.findall_nodes_until_blocked(self.acts_on_nonadjacent_qubits)
        )
        assert not any(
            self.remaining_dag.has_edge(b, a) for a, b in itertools.combinations(nodes, 2)
        )
        assert not any(self.acts_on_nonadjacent_qubits(node.val) for node in nodes)
        remaining_nodes = [node for node in self.remaining_dag.ordered_nodes() if node not in nodes]
        for node, remaining_node in itertools.product(nodes, remaining_nodes):
            assert not self.remaining_dag.has_edge(remaining_node, node)
        for node in nodes:
            self.remaining_dag.remove_node(node)
            logical_op = node.val
            physical_op = logical_op.with_qubits(*self.log_to_phys(*logical_op.qubits))
            assert len(physical_op.qubits) < 2 or physical_op.qubits in self.device_graph.edges
            self.physical_ops.append(physical_op)
        return len(nodes)

    @property
    def swap_network(self) -> SwapNetwork:
        return SwapNetwork(circuits.Circuit(self.physical_ops), self.initial_mapping)

    def distance(self, edge: QidPair) -> int:
        """The distance between the physical qubits mapped to by a pair of
        logical qubits."""
        return self.physical_distances[cast(QidPair, tuple(self.log_to_phys(*edge)))]

    def swap_along_path(self, path: Tuple[ops.Qid]):
        """Adds SWAPs to move a logical qubit along a specified path."""
        for i in range(len(path) - 1):
            self.apply_swap(cast(QidPair, path[i : i + 2]))

    def bring_farthest_pair_together(self, pairs: Sequence[QidPair]):
        """Adds SWAPs to bring the farthest-apart pair of logical qubits
        together."""
        distances = [self.distance(pair) for pair in pairs]
        assert distances
        max_distance = min(distances)
        farthest_pairs = [pair for pair, d in zip(pairs, distances) if d == max_distance]
        choice = self.prng.choice(len(farthest_pairs))
        farthest_pair = farthest_pairs[choice]
        edge = self.log_to_phys(*farthest_pair)
        shortest_path = nx.shortest_path(self.device_graph, *edge)
        assert len(shortest_path) - 1 == max_distance
        midpoint = max_distance // 2
        self.swap_along_path(shortest_path[:midpoint])
        self.swap_along_path(shortest_path[midpoint:])

    def get_distance_vector(self, logical_edges: Iterable[QidPair], swaps: Sequence[QidPair]):
        """Gets distances between physical qubits mapped to by given logical
        edges, after specified SWAPs are applied."""
        self.update_mapping(*swaps)
        distance_vector = np.array([self.distance(e) for e in logical_edges])
        self.update_mapping(*swaps)
        return distance_vector

    def apply_next_swaps(self, require_frontier_adjacency: bool = False):
        """Applies a few SWAPs to get the mapping closer to one in which the
        next logical gates can be applied.

        See route_circuit_greedily for more details.
        """

        time_slices = get_time_slices(self.remaining_dag)

        if require_frontier_adjacency:
            frontier_edges = sorted(time_slices[0].edges)
            self.bring_farthest_pair_together(frontier_edges)
            return

        for k in range(1, self.max_search_radius + 1):
            candidate_swap_sets = list(self.get_edge_sets(k))
            for time_slice in time_slices:
                edges = sorted(time_slice.edges)
                distance_vectors = list(
                    self.get_distance_vector(edges, swap_set) for swap_set in candidate_swap_sets
                )
                dominated_indices = _get_dominated_indices(distance_vectors)
                candidate_swap_sets = [
                    S for i, S in enumerate(candidate_swap_sets) if i not in dominated_indices
                ]
                if len(candidate_swap_sets) == 1:
                    self.apply_swap(*candidate_swap_sets[0])
                    if list(
                        self.remaining_dag.findall_nodes_until_blocked(
                            self.acts_on_nonadjacent_qubits
                        )
                    ):
                        return
                    else:
                        break

        self.apply_next_swaps(True)

    def route(self):
        self.apply_possible_ops()
        empty_steps_remaining = self.max_num_empty_steps
        while self.remaining_dag:
            self.apply_next_swaps(not empty_steps_remaining)
            n_applied_ops = self.apply_possible_ops()
            if n_applied_ops:
                empty_steps_remaining = self.max_num_empty_steps
            else:
                empty_steps_remaining -= 1
        assert ops_are_consistent_with_device_graph(self.physical_ops, self.device_graph)


def _get_dominated_indices(vectors: List[np.ndarray]) -> Set[int]:
    """Get the indices of vectors that are element-wise at least some other
    vector.
    """
    dominated_indices = set()
    for i, v in enumerate(vectors):
        for w in vectors[:i] + vectors[i + 1 :]:
            if all(v >= w):
                dominated_indices.add(i)
                break
    return dominated_indices
