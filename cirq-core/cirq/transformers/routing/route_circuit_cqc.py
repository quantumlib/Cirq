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

"""Heuristic qubit routing algorithm based on arxiv:1902.08091."""

from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING
from itertools import combinations
import networkx as nx

from cirq import circuits, ops, protocols
from cirq.transformers import transformer_api, transformer_primitives
from cirq.transformers.routing import mapping_manager, line_initial_mapper

if TYPE_CHECKING:
    import cirq


def disjoint_pair_qubit_pair_combinations(
    qubit_pairs: List[Tuple['cirq.Qid', 'cirq.Qid']]
) -> List[Tuple[Tuple['cirq.Qid', 'cirq.Qid'], ...]]:
    """Gets disjoint pair combinations of qubits pairs.

    For example:

        >>> disjoint_pair_qubit_pair_combinations([(q(1), q(2)), (q(3), q(4)), (q(2), q(5))])
        >>> Returns [((q(1), q(2)), (q(3), q(4))), ((q(3), q(4)), (q(2), q(5)))].

    Args:
        qubit_pairs: list of qubit pairs to be combined.

    Returns:
        All 2-combinations between qubit pairs that are disjoint.

    """
    return [
        pair
        for pair in combinations(qubit_pairs, 2)
        if set(q for q in pair[0]).isdisjoint(set(q for q in pair[1]))
    ]


@transformer_api.transformer
class RouteCQC:
    """Transformer class that implements a circuit routing algorithm.

    The algorithm proceeds as follows:

    1. Computes the timesteps of the circuit: considering operations in the given circuit from
        beginning to end, the next timestep is a maximal set of 2-qubit operations that act on
        disjoint qubits. It is 'maximal' because any 2-qubit gate's qubits in the next timestep
        must intersect with the qubits that are acted on in the current timestep.

    2. Places the logical qubits in the input circuit onto some input device by using an
        initial mapper (`cirq.LineInitialMapper` by default).

    3. Insert necessary swaps to ensure all 2-qubit gates are executable on the device by
        traversing the timesteps from left to right and for each timestep:
            1. Remove any single qubit gate and executable 2-qubit gate in the current
                timestep and add it to the output routed circuit.
            2. If there aren't any gates left in the current timestep, move on to the next.
            3. If there are gates remaining in the current timesteps, consider a set of
                candidate swaps on them and rank them based on a heuristic cost function. Pick
                the swap that minimises the cost and use it to update our logical to physical
                mapping. Repeat from 3.1.

    For example:

        >>> import cirq_google as cg
        >>> circuit = cirq.testing.random_circuit(5, 10, 0.6)
        >>> router = cirq.RouteCQC(cg.Sycamore)

        >>> routed_circuit = router(circuit)
        or
        >>> routed_circuit, placement_map, swap_map = router.route_circuit(circuit)

    `circuit.transform_qubits(placement_map)` is unitarily equivalent to `routed_circuit` after
    permuting its qubits by `swap_map`.
    """

    def __init__(self, device_graph: nx.Graph):
        """Initializes the circuit routing transformer.

        Args:
            device_graph: The connectivity graph of physical qubits.

        Raises:
            ValueError: if `device_graph` is a directed graph.
        """

        if nx.is_directed(device_graph):
            raise ValueError("Device graph must be undirected.")
        self.device_graph = device_graph

    def __call__(
        self,
        circuit: 'cirq.AbstractCircuit',
        *,
        max_search_radius: int = 8,
        preserve_moment_structure: bool = False,
        tag_inserted_swaps: bool = False,
        initial_mapper: Optional['cirq.AbstractInitialMapper'] = None,
        context: Optional['cirq.TransformerContext'] = None,
    ) -> 'cirq.AbstractCircuit':
        """Transforms the given circuit to make it executable on the device.

        Since routing doesn't necessarily modify any specific operation and only adds swaps
        before / after operations to ensure the circuit can be executed, tagging operations with
        tags from context.tags_to_ignore will have no impact on the routing procedure.

        If `preserve_moment_structure` is True then the moments of `circuit` will be used as the
        timesteps of this circuit. This means that the moments in between inserted swap will retain
        their structure. If `preserve_moment_structure` is False no such guarantee is given but the
        algorithm generally performs better in this case.

        This transformer assumes that all multi-qubit operations have been decomposed into 2-qubit
        operations and will raise an error if `circuit` a n-qubit operation where n > 2. If
        `circuit` contains `cirq.CircuitOperation`s and `context.deep` is True then they are first
        unrolled before proceeding. If `context.deep` is False or `context` is None then any
        `cirq.CircuitOperation` that acts on more than 2-qubits will also raise an error.

        Args:
            circuit: the input circuit to be transformed.
            max_search_radius: the maximum number of times the cost function can be iterated for
                convergence. If the cost function is iterated `max_search_radius` number of times
                without converging then symmetry breaking is used.
            preserve_moment_structure: whether or not the transfomer should preserve the given
                moment structure of `circuit`.
            tag_inserted_swaps: whether or not a `cirq.RoutingSwapTag` should be attached to
                inserted swap operations.
            initial_mapper: an initial mapping strategy (placement) of logical qubits in the
                circuit onto physical qubits on the device. If not provided, defaults to an
                instance of `cirq.LineInitialMapper`.
            context: transformer context storing common configurable options for transformers.

        Returns:
            The routed circuit, which is equivalent to original circuit upto a final qubit
                permutation and where each 2-qubit operation is between adjacent qubits in the
                `device_graph`.

        Raises:
            ValueError: if circuit has operations that act on 3 or more qubits.
        """
        routed_circuit, _, _ = self.route_circuit(
            circuit=circuit,
            max_search_radius=max_search_radius,
            preserve_moment_structure=preserve_moment_structure,
            tag_inserted_swaps=tag_inserted_swaps,
            initial_mapper=initial_mapper,
            context=context,
        )
        return routed_circuit

    def route_circuit(
        self,
        circuit: 'cirq.AbstractCircuit',
        *,
        max_search_radius: int = 8,
        preserve_moment_structure: bool = False,
        tag_inserted_swaps: bool = False,
        initial_mapper: Optional['cirq.AbstractInitialMapper'] = None,
        context: Optional['cirq.TransformerContext'] = None,
    ) -> Tuple['cirq.AbstractCircuit', Dict['cirq.Qid', 'cirq.Qid'], Dict['cirq.Qid', 'cirq.Qid']]:
        """Transforms the given circuit to make it executable on the device.

        Since routing doesn't necessarily modify any specific operation and only adds swaps
        before / after operations to ensure the circuit can be executed, tagging operations with
        tags from context.tags_to_ignore will have no impact on the routing procedure.

        If `preserve_moment_structure` is True then the moments of `circuit` will be used as the
        timesteps of this circuit. This means that the moments in between inserted swap will retain
        their structure. If `preserve_moment_structure` is False no such guarantee is given but the
        algorithm generally performs better in this case.

        This transformer assumes that all multi-qubit operations have been decomposed into 2-qubit
        operations and will raise an error if `circuit` a n-qubit operation where n > 2. If
        `circuit` contains `cirq.CircuitOperation`s and `context.deep` is True then they are first
        unrolled before proceeding. If `context.deep` is False or `context` is None then any
        `cirq.CircuitOperation` that acts on more than 2-qubits will also raise an error.

        Args:
            circuit: the input circuit to be transformed.
            max_search_radius: the maximum number of times the cost function can be iterated for
                convergence. If the cost function is iterated `max_search_radius` number of times
                without converging then symmetry breaking is used.
            preserve_moment_structure: whether or not the transfomer should preserve the given
                moment structure of `circuit`.
            tag_inserted_swaps: whether or not a RoutingSwapTag should be attched to inserted swap
                operations.
            initial_mapper: an initial mapping strategy (placement) of logical qubits in the
                circuit onto physical qubits on the device.
            context: transformer context storing common configurable options for transformers.

        Returns:
            The routed circuit executable on the harware with the same unitary as `circuit`.
            The initial mapping from logical to physical qubits used as part of the routing
                procedure.
            The mapping from physical qubits before inserting swaps to physical qubits after
                inserting swaps.

        Raises:
            ValueError: if circuit has operations that act on 3 or more qubits.
        """

        # 0. Handle CircuitOperations by unrolling them.
        if context is not None and context.deep is True:
            circuit = transformer_primitives.unroll_circuit_op(circuit, deep=True)
        if not all(protocols.num_qubits(op) <= 2 for op in circuit.all_operations()):
            raise ValueError("Input circuit must only have ops that act on 1 or 2 qubits.")

        # 1. Do the initial mapping of logical to physical qubits.
        if initial_mapper is None:
            initial_mapper = line_initial_mapper.LineInitialMapper(self.device_graph)
        initial_mapping = initial_mapper.initial_mapping(circuit)

        # 2. Construct a mapping manager that implicitly keeps track of this mapping and provides
        # convinience methods over the image of the map on the device graph.
        self._mm = mapping_manager.MappingManager(self.device_graph, initial_mapping)

        # 3. Get timesteps and single-qubit operations.
        timesteps, single_qubit_ops = self._get_timesteps_and_single_qubit_ops(
            circuit, preserve_moment_structure
        )

        # 4. Do the routing and save the routed circuit as a list of moments.
        routed_ops = self._route(
            timesteps, single_qubit_ops, max_search_radius, tag_inserted_swaps=tag_inserted_swaps
        )

        # 5. Return the routed circuit by packing each inner list of ops as densely as posslbe and
        # preserving outer moment structure.
        return (
            circuits.Circuit(circuits.Circuit(m) for m in routed_ops),
            initial_mapping,
            {initial_mapping[k]: v for k, v in self._mm.map.items()},
        )

    def _get_timesteps_and_single_qubit_ops(
        self, circuit: 'cirq.AbstractCircuit', preserve_moment_structure: bool
    ) -> Tuple[List[List['cirq.Operation']], List[List['cirq.Operation']]]:
        """Gets the timesteps of the circuit and the the corresponding single-qubit operations."""
        if preserve_moment_structure:
            single_qubit_ops_preserve: List[List['cirq.Operation']] = [
                [] for i in range(len(circuit))
            ]

            def map_func_preserving(op: 'cirq.Operation', moment_index: int):
                if protocols.num_qubits(op) == 2:
                    return op
                single_qubit_ops_preserve[moment_index].append(op)
                return []

            reduced_circuit_preserve = transformer_primitives.map_operations(
                circuit, map_func=map_func_preserving
            )
            return (
                [list(moment.operations) for moment in reduced_circuit_preserve],
                single_qubit_ops_preserve,
            )

        reduced_circuit = circuits.Circuit()

        def map_func_not_preserving(op: 'cirq.Operation', _: int):
            timestep_index = reduced_circuit.earliest_available_moment(op)
            if protocols.num_qubits(op) == 2:
                reduced_circuit.append(op)
            return op.with_tags(timestep_index)

        circuit_with_tags = transformer_primitives.map_operations(
            circuit, map_func=map_func_not_preserving
        )
        single_qubit_operations: List[List['cirq.Operation']] = [
            [] for i in range(len(reduced_circuit) + 1)
        ]
        for op in circuit_with_tags.all_operations():
            if protocols.num_qubits(op) == 1 and isinstance(op.tags[-1], int):
                timestep_index = op.tags[-1]
                single_qubit_operations[timestep_index].append(op)
        return [list(moment.operations) for moment in reduced_circuit], single_qubit_operations

    def _route(
        self,
        timesteps: List[List['cirq.Operation']],
        single_qubit_ops: List[List['cirq.Operation']],
        max_search_radius: int,
        tag_inserted_swaps: bool = False,
    ) -> List[List['cirq.Operation']]:
        """Main routing procedure that creates the routed circuit, inserts the user's gates in it,
        and inserts the necessary swaps.

        Args:
          timesteps: the circuit's timesteps as defined by the paper
          max_search_radius: the maximum number of times the cost function can be iterated for
            convergence.

        Returns:
          a list of lists corresponding to moments of the routed circuit
        """

        def process_executable_ops(idx: int):
            unexecutable_ops = []
            for op in timesteps[idx]:
                if self._mm.can_execute(op):
                    routed_ops[idx].append(self._mm.mapped_op(op))
                else:
                    unexecutable_ops.append(op)
            timesteps[idx] = unexecutable_ops

        routed_ops: List[List['cirq.Operation']] = [[] for i in range(len(timesteps) + 1)]
        for idx in range(len(timesteps)):
            # add single qubit ops the current output moment.
            routed_ops[idx].extend([self._mm.mapped_op(op) for op in single_qubit_ops[idx]])

            process_executable_ops(idx)

            seen: Set[Tuple[Tuple['cirq.Qid', cirq.Qid], ...]] = set()
            while len(timesteps[idx]) != 0:
                sigma: List[Tuple[Tuple['cirq.Qid', 'cirq.Qid'], ...]] = [
                    (swap,) for swap in self._initial_candidate_swaps(timesteps[idx])
                ]
                for s in range(idx, min(max_search_radius + idx, len(timesteps))):
                    if len(sigma) <= 1:
                        break
                    sigma = self._next_candidate_swaps(sigma, timesteps[s])

                if len(sigma) > 1 and idx + max_search_radius <= len(timesteps):
                    chosen_swaps = self._symmetry_swap_pair(timesteps, idx, max_search_radius)
                else:
                    chosen_swaps = sigma[0]

                if chosen_swaps in seen:
                    chosen_swaps = self._symmetry_brute_force(timesteps, idx)
                else:
                    seen.add(chosen_swaps)

                for swap in chosen_swaps:
                    inserted_swap = self._mm.mapped_op(ops.SWAP(*swap))
                    if tag_inserted_swaps:
                        inserted_swap = inserted_swap.with_tags(ops.RoutingSwapTag())
                    routed_ops[idx].append(inserted_swap)
                    self._mm.apply_swap(*swap)
                process_executable_ops(idx)

        # edge case: single qubit gates may appear in moments after all 2-qubit gates
        for i in range(len(timesteps), len(single_qubit_ops)):
            routed_ops.append([self._mm.mapped_op(op) for op in single_qubit_ops[i]])

        return routed_ops

    def _symmetry_swap_pair(
        self, timesteps: List[List['cirq.Operation']], idx: int, max_search_radius: int
    ) -> Tuple[Tuple['cirq.Qid', 'cirq.Qid'], ...]:
        """Computes cost function with pairs of candidate swaps that act on disjoint qubits."""
        pair_sigma = disjoint_pair_qubit_pair_combinations(
            self._initial_candidate_swaps(timesteps[idx])
        )
        for s in range(idx, min(max_search_radius + idx, len(timesteps))):
            if len(pair_sigma) <= 1:
                break
            pair_sigma = self._next_candidate_swaps(pair_sigma, timesteps[s])

        if len(pair_sigma) > 1 and idx + max_search_radius <= len(timesteps):
            return self._symmetry_brute_force(timesteps, idx)
        chosen_swap_pair = pair_sigma[0]
        return (chosen_swap_pair[0], chosen_swap_pair[1])

    def _symmetry_brute_force(
        self, timesteps: List[List['cirq.Operation']], idx: int
    ) -> Tuple[Tuple['cirq.Qid', 'cirq.Qid'], ...]:
        """Inserts SWAPS along the shortest path of the qubits that are the farthest."""
        qubits = max(
            [(op.qubits, self._mm.dist_on_device(*op.qubits)) for op in timesteps[idx]],
            key=lambda x: x[1],
        )[0]
        path = self._mm.shortest_path(*qubits)
        q1 = path[0]
        return tuple([(q1, path[i + 1]) for i in range(len(path) - 2)])

    def _initial_candidate_swaps(
        self, timestep_ops: List['cirq.Operation']
    ) -> List[Tuple['cirq.Qid', 'cirq.Qid']]:
        """Finds all feasible SWAPs between qubits involved in 2-qubit operations."""
        physical_qubits = set(self._mm.map[op.qubits[i]] for op in timestep_ops for i in range(2))
        physical_swaps = self._mm.induced_subgraph.edges(nbunch=physical_qubits)
        return [(self._mm.inverse_map[q1], self._mm.inverse_map[q2]) for q1, q2 in physical_swaps]

    def _next_candidate_swaps(
        self,
        candidate_swaps: List[Tuple[Tuple['cirq.Qid', 'cirq.Qid'], ...]],
        timestep_ops: List['cirq.Operation'],
    ) -> List[Tuple[Tuple['cirq.Qid', 'cirq.Qid'], ...]]:
        """Iterates the heuristic function.

        Given a list of candidate swaps find a subset that leads to a minimal longest shortest path
        between any paired qubits in the curernt timestep.
        """
        costs = {}
        for swap in candidate_swaps:
            costs[swap] = self._cost(swap, timestep_ops)
        shortest_longest_paths = [
            swap for swap in costs if costs[swap][0] == min([v[0] for v in costs.values()])
        ]
        return [
            swap
            for swap in shortest_longest_paths
            if costs[swap][1] == min([costs[swap][1] for swap in shortest_longest_paths])
        ]

    def _cost(
        self, swaps: Tuple[Tuple['cirq.Qid', 'cirq.Qid'], ...], timestep_ops: List['cirq.Operation']
    ) -> Tuple[int, int]:
        """Computes the cost function for the given list of swaps over the current timestep ops."""
        for swap in swaps:
            self._mm.apply_swap(*swap)
        max_length, sum_length = 0, 0
        for op in timestep_ops:
            # need to add this if statement because future timesteps may still have 1-qubit ops
            q1, q2 = op.qubits[0], op.qubits[1]
            max_length = max(max_length, self._mm.dist_on_device(q1, q2))
            sum_length += self._mm.dist_on_device(q1, q2)
        for swap in swaps:
            self._mm.apply_swap(*swap)
        return (max_length, sum_length)

    def __eq__(self, other) -> bool:
        return nx.utils.graphs_equal(self.device_graph, other.device_graph)

    def __repr__(self) -> str:
        return f'cirq.RouteCQC(nx.Graph({dict(self.device_graph.adjacency())}))'
