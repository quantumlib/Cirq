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

import itertools
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING

import networkx as nx

from cirq import circuits, ops, protocols
from cirq.transformers import transformer_api, transformer_primitives
from cirq.transformers.routing import line_initial_mapper, mapping_manager

if TYPE_CHECKING:
    import cirq

QidIntPair = Tuple[int, int]


def _disjoint_nc2_combinations(
    qubit_pairs: Sequence[QidIntPair],
) -> List[Tuple[QidIntPair, QidIntPair]]:
    """Gets disjoint pair combinations of qubits pairs.

    For example:

        >>> q = [*range(5)]
        >>> disjoint_swaps = cirq.transformers.routing.route_circuit_cqc._disjoint_nc2_combinations(
        ...     [(q[0], q[1]), (q[2], q[3]), (q[1], q[4])]
        ... )
        >>> disjoint_swaps == [((q[0], q[1]), (q[2], q[3])), ((q[2], q[3]), (q[1], q[4]))]
        True

    Args:
        qubit_pairs: list of qubit pairs to be combined.

    Returns:
        All 2-combinations between qubit pairs that are disjoint.
    """
    return [
        pair for pair in itertools.combinations(qubit_pairs, 2) if set(pair[0]).isdisjoint(pair[1])
    ]


@transformer_api.transformer
class RouteCQC:
    """Transformer class that implements a circuit routing algorithm.

    The algorithm proceeds as follows:

    1. Computes the timesteps (two_qubit_ops) of the circuit: considering operations in the given
        circuit from beginning to end, the next timestep is a maximal set of 2-qubit operations
        that act on disjoint qubits. It is 'maximal' because any 2-qubit gate's qubits in the next
        timestep must intersect with the qubits that are acted on in the current timestep.

    2. Places the logical qubits in the input circuit onto some input device graph by using an
        initial mapper (`cirq.LineInitialMapper` by default).

    3. Insert necessary swaps to ensure all 2-qubit gates are between adjacent qubits on the device
        graph by traversing the timesteps from left to right and for each timestep:
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
        lookahead_radius: int = 8,
        tag_inserted_swaps: bool = False,
        initial_mapper: Optional['cirq.AbstractInitialMapper'] = None,
        context: Optional['cirq.TransformerContext'] = None,
    ) -> 'cirq.AbstractCircuit':
        """Transforms the given circuit to make it executable on the device.

        This method calls self.route_circuit and returns the routed circuit. See docstring of
        `RouteCQC.route_circuit` for more details on routing.

        Args:
            circuit: the input circuit to be transformed.
            lookahead_radius: the maximum number of succeeding timesteps the algorithm will
                consider for ranking candidate swaps with the cost cost function.
            tag_inserted_swaps: whether or not a `cirq.RoutingSwapTag` should be attached to
                inserted swap operations.
            initial_mapper: an initial mapping strategy (placement) of logical qubits in the
                circuit onto physical qubits on the device. If not provided, defaults to an
                instance of `cirq.LineInitialMapper`.
            context: transformer context storing common configurable options for transformers.

        Returns:
            The routed circuit, which is equivalent to original circuit up to a final qubit
                permutation and where each 2-qubit operation is between adjacent qubits in the
                `device_graph`.

        Raises:
            ValueError: if circuit has operations that act on 3 or more qubits, except measurements.
        """
        routed_circuit, _, _ = self.route_circuit(
            circuit=circuit,
            lookahead_radius=lookahead_radius,
            tag_inserted_swaps=tag_inserted_swaps,
            initial_mapper=initial_mapper,
            context=context,
        )
        return routed_circuit

    def route_circuit(
        self,
        circuit: 'cirq.AbstractCircuit',
        *,
        lookahead_radius: int = 8,
        tag_inserted_swaps: bool = False,
        initial_mapper: Optional['cirq.AbstractInitialMapper'] = None,
        context: Optional['cirq.TransformerContext'] = None,
    ) -> Tuple['cirq.AbstractCircuit', Dict['cirq.Qid', 'cirq.Qid'], Dict['cirq.Qid', 'cirq.Qid']]:
        """Transforms the given circuit to make it executable on the device.

        This transformer assumes that all multi-qubit operations have been decomposed into 2-qubit
        operations and will raise an error if `circuit` a n-qubit operation where n > 2. If
        `circuit` contains `cirq.CircuitOperation`s and `context.deep` is True then they are first
        unrolled before proceeding. If `context.deep` is False or `context` is None then any
        `cirq.CircuitOperation` that acts on more than 2-qubits will also raise an error.

        The algorithm tries to find the best swap at each timestep by ranking a set of candidate
        swaps against operations starting from the current timestep (say s) to the timestep at index
        s + `lookahead_radius` to prune the set of candidate swaps. If it fails  to converge to a to
        a single swap because of highly symmetrical device or circuit connectivity, then symmetry
        breaking strategies are used.

        Since routing doesn't necessarily modify any specific operation and only adds swaps
        before / after operations to ensure the circuit can be executed, tagging operations with
        tags from context.tags_to_ignore will have no impact on the routing procedure.

        Args:
            circuit: the input circuit to be transformed.
            lookahead_radius: the maximum number of succeeding timesteps the algorithm will
                consider for ranking candidate swaps with the cost cost function.
            tag_inserted_swaps: whether or not a RoutingSwapTag should be attched to inserted swap
                operations.
            initial_mapper: an initial mapping strategy (placement) of logical qubits in the
                circuit onto physical qubits on the device.
            context: transformer context storing common configurable options for transformers.

        Returns:
            The routed circuit, which is equivalent to original circuit up to a final qubit
                permutation and where each 2-qubit operation is between adjacent qubits in the
                `device_graph`.
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
        two_qubit_ops, single_qubit_ops = self._get_one_and_two_qubit_ops_as_timesteps(circuit)

        # 4. Do the routing and save the routed circuit as a list of moments.
        routed_ops = self._route(
            mm,
            two_qubit_ops,
            single_qubit_ops,
            lookahead_radius,
            tag_inserted_swaps=tag_inserted_swaps,
        )

        # 5. Return the routed circuit by packing each inner list of ops as densely as possible and
        # preserving outer moment structure. Also return initial map and swap permutation map.
        return (
            circuits.Circuit(circuits.Circuit(m) for m in routed_ops),
            initial_mapping,
            {
                initial_mapping[mm.int_to_logical_qid[k]]: mm.int_to_physical_qid[v]
                for k, v in enumerate(mm.logical_to_physical)
            },
        )

    @classmethod
    def _get_one_and_two_qubit_ops_as_timesteps(
        cls, circuit: 'cirq.AbstractCircuit'
    ) -> Tuple[List[List['cirq.Operation']], List[List['cirq.Operation']]]:
        """Gets the single and two qubit operations of the circuit factored into timesteps.

        The i'th entry in the nested two-qubit and single-qubit ops correspond to the two-qubit
        gates and single-qubit gates of the i'th timesteps respectively. When constructing the
        output routed circuit, single-qubit operations are inserted before two-qubit operations.

        Raises:
            ValueError: if circuit has intermediate measurements that act on three or more
                        qubits with a custom key.
        """
        two_qubit_circuit = circuits.Circuit()
        single_qubit_ops: List[List[cirq.Operation]] = []

        for i, moment in enumerate(circuit):
            for op in moment:
                timestep = two_qubit_circuit.earliest_available_moment(op)
                single_qubit_ops.extend([] for _ in range(timestep + 1 - len(single_qubit_ops)))
                two_qubit_circuit.append(
                    circuits.Moment() for _ in range(timestep + 1 - len(two_qubit_circuit))
                )
                if protocols.num_qubits(op) > 2 and protocols.is_measurement(op):
                    key = op.gate.key  # type: ignore
                    default_key = ops.measure(op.qubits).gate.key  # type: ignore
                    if len(circuit.moments) == i + 1:
                        single_qubit_ops[timestep].append(op)
                    elif key in ('', default_key):
                        single_qubit_ops[timestep].extend(ops.measure(qubit) for qubit in op.qubits)
                    else:
                        raise ValueError(
                            'Intermediate measurements on three or more qubits '
                            'with a custom key are not supported'
                        )
                elif protocols.num_qubits(op) == 2:
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
        lookahead_radius: int,
        tag_inserted_swaps: bool = False,
    ) -> List[List['cirq.Operation']]:
        """Main routing procedure that inserts necessary swaps on the given timesteps.

        The i'th element of the returned list corresponds to the routed operatiosn in the i'th
        timestep.

        Args:
          two_qubit_ops: the circuit's two-qubit gates factored into timesteps as defined by the
            paper.
          single_qubit_ops: the circuit's single-qubit gates factored into timesteps as defined by
            the paper.
          lookahead_radius: the maximum number of times the cost function can be iterated for
            convergence.
        tag_inserted_swaps: whether or not a RoutingSwapTag should be attched to inserted swap
            operations.

        Returns:
            a list of lists corresponding to timesteps of the routed circuit.
        """
        two_qubit_ops_ints: List[List[QidIntPair]] = [
            [
                (mm.logical_qid_to_int[op.qubits[0]], mm.logical_qid_to_int[op.qubits[1]])
                for op in timestep_ops
            ]
            for timestep_ops in two_qubit_ops
        ]
        routed_ops: List[List['cirq.Operation']] = []

        def process_executable_two_qubit_ops(timestep: int) -> int:
            unexecutable_ops: List['cirq.Operation'] = []
            unexecutable_ops_ints: List[QidIntPair] = []
            for op, op_ints in zip(two_qubit_ops[timestep], two_qubit_ops_ints[timestep]):
                if mm.is_adjacent(*op_ints):
                    routed_ops[timestep].append(mm.mapped_op(op))
                else:
                    unexecutable_ops.append(op)
                    unexecutable_ops_ints.append(op_ints)
            two_qubit_ops[timestep] = unexecutable_ops
            two_qubit_ops_ints[timestep] = unexecutable_ops_ints
            return len(unexecutable_ops)

        strats = [cls._choose_single_swap, cls._choose_pair_of_swaps]

        for timestep in range(len(two_qubit_ops)):
            # Add single-qubit ops with qubits given by the current mapping.
            routed_ops.append([mm.mapped_op(op) for op in single_qubit_ops[timestep]])

            # swaps applied in the current timestep thus far. This ensures the same swaps
            # don't get executed twice in the same timestep.
            seen: Set[Tuple[QidIntPair, ...]] = set()

            while process_executable_two_qubit_ops(timestep):
                chosen_swaps: Optional[Tuple[QidIntPair, ...]] = None
                for strat in strats:
                    chosen_swaps = strat(mm, two_qubit_ops_ints, timestep, lookahead_radius)
                    if chosen_swaps is not None:
                        break

                if chosen_swaps is None or chosen_swaps in seen:
                    chosen_swaps = cls._brute_force_strategy(mm, two_qubit_ops_ints, timestep)
                else:
                    seen.add(chosen_swaps)

                for swap in chosen_swaps:
                    inserted_swap = mm.mapped_op(
                        ops.SWAP(mm.int_to_logical_qid[swap[0]], mm.int_to_logical_qid[swap[1]])
                    )
                    if tag_inserted_swaps:
                        inserted_swap = inserted_swap.with_tags(ops.RoutingSwapTag())
                    routed_ops[timestep].append(inserted_swap)
                    mm.apply_swap(*swap)

        return routed_ops

    @classmethod
    def _brute_force_strategy(
        cls,
        mm: mapping_manager.MappingManager,
        two_qubit_ops_ints: Sequence[Sequence[QidIntPair]],
        timestep: int,
    ) -> Tuple[QidIntPair, ...]:
        """Inserts SWAPS along the shortest path of the qubits that are the farthest.

        Since swaps along the shortest path are being executed one after the other, in order
        to achieve the physical swaps (M[q1], M[q2]), (M[q2], M[q3]), ..., (M[q_{i-1}], M[q_i]),
        we must execute the logical swaps (q1, q2), (q1, q3), ..., (q_1, qi).
        """
        furthest_op = max(two_qubit_ops_ints[timestep], key=lambda op: mm.dist_on_device(*op))
        path = mm.shortest_path(*furthest_op)
        return tuple([(path[0], path[i + 1]) for i in range(len(path) - 2)])

    @classmethod
    def _choose_pair_of_swaps(
        cls,
        mm: mapping_manager.MappingManager,
        two_qubit_ops_ints: Sequence[Sequence[QidIntPair]],
        timestep: int,
        lookahead_radius: int,
    ) -> Optional[Tuple[QidIntPair, ...]]:
        """Computes cost function with pairs of candidate swaps that act on disjoint qubits."""
        pair_sigma = _disjoint_nc2_combinations(
            cls._initial_candidate_swaps(mm, two_qubit_ops_ints[timestep])
        )
        return cls._choose_optimal_swap(
            mm, two_qubit_ops_ints, timestep, lookahead_radius, pair_sigma
        )

    @classmethod
    def _choose_single_swap(
        cls,
        mm: mapping_manager.MappingManager,
        two_qubit_ops_ints: Sequence[Sequence[QidIntPair]],
        timestep: int,
        lookahead_radius: int,
    ) -> Optional[Tuple[QidIntPair, ...]]:
        """Computes cost function with list of single candidate swaps."""
        sigma: List[Tuple[QidIntPair, ...]] = [
            (swap,) for swap in cls._initial_candidate_swaps(mm, two_qubit_ops_ints[timestep])
        ]
        return cls._choose_optimal_swap(mm, two_qubit_ops_ints, timestep, lookahead_radius, sigma)

    @classmethod
    def _choose_optimal_swap(
        cls,
        mm: mapping_manager.MappingManager,
        two_qubit_ops_ints: Sequence[Sequence[QidIntPair]],
        timestep: int,
        lookahead_radius: int,
        sigma: Sequence[Tuple[QidIntPair, ...]],
    ) -> Optional[Tuple[QidIntPair, ...]]:
        """Optionally returns the swap with minimum cost from a list of n-tuple candidate swaps.

        Computes a cost (as defined by the overridable function `_cost`) for each candidate swap
        in the current timestep. If there does not exist a unique list of swaps with minial cost,
        proceeds to the rank the subset of minimal swaps from the current timestep in the next
        timestep. Iterate this this looking ahead process up to the next `lookahead_radius`
        timesteps. If there still doesn't exist a unique swap with minial cost then returns None.
        """
        for s in range(timestep, min(lookahead_radius + timestep, len(two_qubit_ops_ints))):
            if len(sigma) <= 1:
                break

            costs = {}
            for swaps in sigma:
                costs[swaps] = cls._cost(mm, swaps, two_qubit_ops_ints[s])
            _, min_cost = min(costs.items(), key=lambda x: x[1])
            sigma = [swaps for swaps, cost in costs.items() if cost == min_cost]

        return (
            None
            if len(sigma) > 1 and timestep + lookahead_radius <= len(two_qubit_ops_ints)
            else sigma[0]
        )

    @classmethod
    def _initial_candidate_swaps(
        cls, mm: mapping_manager.MappingManager, two_qubit_ops: Sequence[QidIntPair]
    ) -> List[QidIntPair]:
        """Finds all feasible SWAPs between qubits involved in 2-qubit operations."""
        physical_qubits = (mm.logical_to_physical[lq[i]] for lq in two_qubit_ops for i in range(2))
        physical_swaps = mm.induced_subgraph_int.edges(nbunch=physical_qubits)
        return [
            (mm.physical_to_logical[q1], mm.physical_to_logical[q2]) for q1, q2 in physical_swaps
        ]

    @classmethod
    def _cost(
        cls,
        mm: mapping_manager.MappingManager,
        swaps: Tuple[QidIntPair, ...],
        two_qubit_ops: Sequence[QidIntPair],
    ) -> Any:
        """Computes the cost function for the given list of swaps over the current timestep ops.

        To use this transformer with a different cost function, create a new subclass that derives
        from `RouteCQC` and override this method.
        """
        for swap in swaps:
            mm.apply_swap(*swap)
        max_length, sum_length = 0, 0
        for lq in two_qubit_ops:
            dist = mm.dist_on_device(*lq)
            max_length = max(max_length, dist)
            sum_length += dist
        for swap in swaps:
            mm.apply_swap(*swap)
        return max_length, sum_length

    def __eq__(self, other) -> bool:
        return nx.utils.graphs_equal(self.device_graph, other.device_graph)

    def __repr__(self) -> str:
        return f'cirq.RouteCQC(nx.Graph({dict(self.device_graph.adjacency())}))'
