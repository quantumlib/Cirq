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

from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
from itertools import combinations
import networkx as nx

from cirq import circuits, ops, protocols
from cirq.transformers import transformer_api, transformer_primitives
from cirq.transformers.routing import mapping_manager, line_initial_mapper

if TYPE_CHECKING:
    import cirq

QidPair = Tuple['cirq.Qid', 'cirq.Qid']


def disjoint_nc2_combinations(qubit_pairs: List[QidPair]) -> List[Tuple[QidPair, ...]]:
    """Gets disjoint pair combinations of qubits pairs.

    E.g. `disjoint_pair_qubit_pair_combinations([(q(1), q(2)), (q(3), q(4)), (q(2), q(5))])`
    should return `[((q(1), q(2)), (q(3), q(4))), ((q(3), q(4)), (q(2), q(5)))]`.

    Args:
        qubit_pairs: list of qubit pairs to be combined.

    Returns:
        All 2-combinations between qubit pairs that are disjoint.

    """
    return [pair for pair in combinations(qubit_pairs, 2) if set(pair[0]).isdisjoint(pair[1])]


@transformer_api.transformer
class RouteCQC:
    """Transformer class that implements a circuit routing algorithm.

    The algorithm proceeds as follows:

    1. Computes the timesteps (two_qubit_ops) of the circuit: considering operations in the given
        circuit from beginning to end, the next timestep is a maximal set of 2-qubit operations
        that act on disjoint qubits. It is 'maximal' because any 2-qubit gate's qubits in the next
        timestep must intersect with the qubits that are acted on in the current timestep.

    2. Places the logical qubits in the input circuit onto some input device by using an
        initial mapper (`cirq.LineInitialMapper` by default).

    3. Insert necessary swaps to ensure all 2-qubit gates are executable on the device by
        traversing the timesteps from left to right and for each timestep:
            1. Remove any single qubit gate and executable 2-qubit gate in the current
                timestep and add it to the output routed circuit.
            2. If there aren't any gates left in the current timestep, move on to the next.
            3. If there are gates remaining in the current timestep, consider a set of
                candidate swaps on them and rank them based on a heuristic cost function. Pick
                the swap that minimises the cost and use it to update our logical to physical
                mapping. Repeat from 3.1.

    For example:

        >>> import cirq_google as cg
        >>> circuit = cirq.testing.random_circuit(5, 10, 0.6)
        >>> device = cg.Sycamore
        >>> router = cirq.RouteCQC(device.metadata.nx_graph)
        >>> rcirc, initial_map, swap_map = router.route_circuit(circuit)
        >>> fcirc = cirq.optimize_for_target_gateset(rcirc, gateset = cg.SycamoreTargetGateset())
        >>> device.validate_circuit(fcirc)
        >>> cirq.testing.assert_circuits_have_same_unitary_given_final_permutation(
        ...     rcirc, circuit.transform_qubits(initial_map), swap_map
        ... )
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
        tag_inserted_swaps: bool = False,
        initial_mapper: Optional['cirq.AbstractInitialMapper'] = None,
        context: Optional['cirq.TransformerContext'] = None,
    ) -> 'cirq.AbstractCircuit':
        """Transforms the given circuit to make it executable on the device.

        Since routing doesn't necessarily modify any specific operation and only adds swaps
        before / after operations to ensure the circuit can be executed, tagging operations with
        tags from context.tags_to_ignore will have no impact on the routing procedure.

        If `preserve_moment_structure` is True then the moments of `circuit` will be used as the
        two_qubit_ops of this circuit. This means that the moments in between inserted swap will
        retain their structure. If `preserve_moment_structure` is False no such guarantee is given
        but the algorithm generally performs better in this case.

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
            ValueError: if circuit has operations that act on 3 or more qubits, except measurements.
        """
        routed_circuit, _, _ = self.route_circuit(
            circuit=circuit,
            max_search_radius=max_search_radius,
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
        tag_inserted_swaps: bool = False,
        initial_mapper: Optional['cirq.AbstractInitialMapper'] = None,
        context: Optional['cirq.TransformerContext'] = None,
    ) -> Tuple['cirq.AbstractCircuit', Dict['cirq.Qid', 'cirq.Qid'], Dict['cirq.Qid', 'cirq.Qid']]:
        """Transforms the given circuit to make it executable on the device.

        Since routing doesn't necessarily modify any specific operation and only adds swaps
        before / after operations to ensure the circuit can be executed, tagging operations with
        tags from context.tags_to_ignore will have no impact on the routing procedure.

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
            ValueError: if circuit has operations that act on 3 or more qubits, except measurements.
        """

        # 0. Handle CircuitOperations by unrolling them.
        if context is not None and context.deep is True:
            circuit = transformer_primitives.unroll_circuit_op(circuit, deep=True)
        if any(
            protocols.num_qubits(op) > 2 and not protocols.is_measurement(op)
            for op in circuit.all_operations()
        ):
            raise ValueError("Input circuit must only have ops that act on 1 or 2 qubits.")

        # 1. Do the initial mapping of logical to physical qubits.
        if initial_mapper is None:
            initial_mapper = line_initial_mapper.LineInitialMapper(self.device_graph)
        initial_mapping = initial_mapper.initial_mapping(circuit)

        # 2. Construct a mapping manager that implicitly keeps track of this mapping and provides
        # convinience methods over the image of the map on the device graph.
        mm = mapping_manager.MappingManager(self.device_graph, initial_mapping)

        # 3. Get two_qubit_ops and single-qubit operations.
        two_qubit_ops, single_qubit_ops = self._get_two_qubit_ops_and_single_qubit_ops(circuit)

        # 4. Do the routing and save the routed circuit as a list of moments.
        routed_ops = self._route(
            mm,
            two_qubit_ops,
            single_qubit_ops,
            max_search_radius,
            tag_inserted_swaps=tag_inserted_swaps,
        )

        # 5. Return the routed circuit by packing each inner list of ops as densely as posslbe and
        # preserving outer moment structure. Also return initial map and swap permutation map.
        return (
            circuits.Circuit(circuits.Circuit(m) for m in routed_ops),
            initial_mapping,
            {initial_mapping[k]: v for k, v in mm.map.items()},
        )

    @classmethod
    def _get_two_qubit_ops_and_single_qubit_ops(
        cls, circuit: 'cirq.AbstractCircuit'
    ) -> Tuple[List[List['cirq.Operation']], List[List['cirq.Operation']]]:
        """Gets the single and two qubit operations of the circuit.

        By construction, single-qubit operations are inserted after two-qubit operations when
        reinserting in the routed circuit.
        """
        two_qubit_circuit = circuits.Circuit()
        single_qubit_ops: List[List[cirq.Operation]] = []
        for moment in circuit:
            for op in moment:
                timestep = two_qubit_circuit.earliest_available_moment(op)
                single_qubit_ops.extend([] for _ in range(timestep + 1 - len(single_qubit_ops)))
                two_qubit_circuit.append(
                    circuits.Moment() for _ in range(timestep + 1 - len(two_qubit_circuit))
                )
                if protocols.num_qubits(op) == 2 and not protocols.is_measurement(op):
                    two_qubit_circuit[timestep] = two_qubit_circuit[timestep].with_operation(op)
                else:
                    single_qubit_ops[timestep].append(op)
        two_qubit_ops = [list(m) for m in two_qubit_circuit]
        return two_qubit_ops, single_qubit_ops

    @classmethod
    def _route(
        cls,
        mm: mapping_manager.MappingManager,
        two_qubit_ops: List[List['cirq.Operation']],
        single_qubit_ops: List[List['cirq.Operation']],
        max_search_radius: int,
        tag_inserted_swaps: bool = False,
    ) -> List[List['cirq.Operation']]:
        """Main routing procedure that creates the routed circuit, inserts the user's gates in it,
        and inserts the necessary swaps.

        Args:
          two_qubit_ops: the circuit's timesteps as defined by the paper
          max_search_radius: the maximum number of times the cost function can be iterated for
            convergence.
        tag_inserted_swaps: whether or not a RoutingSwapTag should be attched to inserted swap
            operations.

        Returns:
          a list of lists corresponding to moments of the routed circuit
        """

        def process_executable_two_qubit_ops(idx: int):
            unexecutable_ops = []
            for op in two_qubit_ops[idx]:
                if mm.can_execute(op):
                    routed_ops[idx].append(mm.mapped_op(op))
                else:
                    unexecutable_ops.append(op)
            two_qubit_ops[idx] = unexecutable_ops

        strats = [cls._iterate_single_swap_strategy, cls._iterate_swap_pair_strategy]
        routed_ops: List[List['cirq.Operation']] = list()

        for idx in range(len(two_qubit_ops)):
            # Add single-qubit ops with qubits given byh the current mapping
            routed_ops.append([mm.mapped_op(op) for op in single_qubit_ops[idx]])

            process_executable_two_qubit_ops(idx)

            # swaps applied in the current timestep thus far. This ensures the same swaps
            # don't get executed twice in the same timestep.
            seen: Set[Tuple[Tuple['cirq.Qid', cirq.Qid], ...]] = set()

            while len(two_qubit_ops[idx]) != 0:
                chosen_swaps: Optional[Tuple[QidPair, ...]] = None
                for strat in strats:
                    chosen_swaps = strat(mm, two_qubit_ops, idx, max_search_radius)
                    if chosen_swaps is not None:
                        break

                if chosen_swaps is None or chosen_swaps in seen:
                    chosen_swaps = cls._brute_force_strategy(mm, two_qubit_ops, idx)
                else:
                    seen.add(chosen_swaps)

                for swap in chosen_swaps:
                    inserted_swap = mm.mapped_op(ops.SWAP(*swap))
                    if tag_inserted_swaps:
                        inserted_swap = inserted_swap.with_tags(ops.RoutingSwapTag())
                    routed_ops[idx].append(inserted_swap)
                    mm.apply_swap(*swap)

                process_executable_two_qubit_ops(idx)

        return routed_ops

    @classmethod
    def _brute_force_strategy(
        cls,
        mm: mapping_manager.MappingManager,
        two_qubit_ops: List[List['cirq.Operation']],
        idx: int,
    ) -> Tuple[QidPair, ...]:
        """Inserts SWAPS along the shortest path of the qubits that are the farthest.

        Since swaps along the shortest path are being executed one after the other, in order
        to achieve the physical swaps (M[q1], M[q2]), (M[q2], M[q3]), ..., (M[q_{i-1}], M[q_i]),
        we must execute the logical swaps (q1, q2), (q1, q3), ..., (q_1, qi).
        """
        furthest_op = max(two_qubit_ops[idx], key=lambda op: mm.dist_on_device(*op.qubits))
        path = mm.shortest_path(*furthest_op.qubits)
        return tuple([(path[0], path[i + 1]) for i in range(len(path) - 2)])

    @classmethod
    def _iterate_swap_pair_strategy(
        cls,
        mm: mapping_manager.MappingManager,
        two_qubit_ops: List[List['cirq.Operation']],
        idx: int,
        max_search_radius: int,
    ) -> Optional[Tuple[QidPair, ...]]:
        """Computes cost function with pairs of candidate swaps that act on disjoint qubits."""
        pair_sigma = disjoint_nc2_combinations(cls._initial_candidate_swaps(mm, two_qubit_ops[idx]))
        return cls._iterate_swaps(mm, two_qubit_ops, idx, max_search_radius, pair_sigma)

    @classmethod
    def _iterate_single_swap_strategy(
        cls,
        mm: mapping_manager.MappingManager,
        two_qubit_ops: List[List['cirq.Operation']],
        idx: int,
        max_search_radius: int,
    ) -> Optional[Tuple[QidPair, ...]]:
        """Computes cost function with list of single candidate swaps."""
        sigma: List[Tuple[QidPair, ...]] = [
            (swap,) for swap in cls._initial_candidate_swaps(mm, two_qubit_ops[idx])
        ]
        return cls._iterate_swaps(mm, two_qubit_ops, idx, max_search_radius, sigma)

    @classmethod
    def _iterate_swaps(
        cls,
        mm: mapping_manager.MappingManager,
        two_qubit_ops: List[List['cirq.Operation']],
        idx: int,
        max_search_radius: int,
        sigma: List[Tuple[QidPair, ...]],
    ) -> Optional[Tuple[QidPair, ...]]:
        """Computes cost function of a list of n-tuple candidate swaps."""
        for s in range(idx, min(max_search_radius + idx, len(two_qubit_ops))):
            if len(sigma) <= 1:
                break
            sigma = cls._get_next_swaps_with_min_cost(mm, sigma, two_qubit_ops[s])

        return (
            None if len(sigma) > 1 and idx + max_search_radius <= len(two_qubit_ops) else sigma[0]
        )

    @classmethod
    def _initial_candidate_swaps(
        cls, mm: mapping_manager.MappingManager, two_qubit_ops: List['cirq.Operation']
    ) -> List[QidPair]:
        """Finds all feasible SWAPs between qubits involved in 2-qubit operations."""
        physical_qubits = (mm.map[op.qubits[i]] for op in two_qubit_ops for i in range(2))
        physical_swaps = mm.induced_subgraph.edges(nbunch=physical_qubits)
        return [(mm.inverse_map[q1], mm.inverse_map[q2]) for q1, q2 in physical_swaps]

    @classmethod
    def _get_next_swaps_with_min_cost(
        cls,
        mm: mapping_manager.MappingManager,
        candidate_swaps: List[Tuple[QidPair, ...]],
        two_qubit_ops: List['cirq.Operation'],
    ) -> List[Tuple[QidPair, ...]]:
        """Iterates the heuristic function.

        Given a list of candidate swaps find a subset that leads to a minimal longest shortest path
        between any paired qubits in the curernt timestep.
        """
        costs = {}
        for swaps in candidate_swaps:
            costs[swaps] = cls._cost(mm, swaps, two_qubit_ops)
        _, min_cost = min(costs.items(), key=lambda x: x[1])
        return [swaps for swaps, cost in costs.items() if cost == min_cost]

    @classmethod
    def _cost(
        cls,
        mm: mapping_manager.MappingManager,
        swaps: Tuple[QidPair, ...],
        two_qubit_ops: List['cirq.Operation'],
    ) -> Any:
        """Computes the cost function for the given list of swaps over the current timestep ops.

        To use this transformer with a different cost function, create a new subclass that derives
        from `RouteCQC` and override this method.
        """
        for swap in swaps:
            mm.apply_swap(*swap)
        max_length, sum_length = 0, 0
        for op in two_qubit_ops:
            q1, q2 = op.qubits
            dist = mm.dist_on_device(q1, q2)
            max_length = max(max_length, dist)
            sum_length += dist
        for swap in swaps:
            mm.apply_swap(*swap)
        return max_length, sum_length

    def __eq__(self, other) -> bool:
        return nx.utils.graphs_equal(self.device_graph, other.device_graph)

    def __repr__(self) -> str:
        return f'cirq.RouteCQC(nx.Graph({dict(self.device_graph.adjacency())}))'
