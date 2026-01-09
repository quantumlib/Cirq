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

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Sequence
from typing import Any, TYPE_CHECKING

import networkx as nx

from cirq import circuits, ops, protocols
from cirq.transformers import transformer_api, transformer_primitives
from cirq.transformers.routing import line_initial_mapper, mapping_manager

if TYPE_CHECKING:
    import cirq

QidIntPair = tuple[int, int]


@dataclasses.dataclass
class RoutingConfig:
    """Configuration parameters for circuit routing.

    Attributes:
        lookahead_radius: Maximum number of timesteps to look ahead for ranking swaps.
        tag_inserted_swaps: Whether to tag inserted swap operations with RoutingSwapTag.
    """

    lookahead_radius: int = 8
    tag_inserted_swaps: bool = False


@dataclasses.dataclass
class SwapSearchContext:
    """Context for swap search operations.

    Attributes:
        timestep: Current timestep index.
        lookahead_radius: Maximum lookahead for swap ranking.
    """

    timestep: int
    lookahead_radius: int


@dataclasses.dataclass
class CircuitOps:
    """Circuit operations organized by timesteps.

    Attributes:
        two_qubit: Two-qubit gates factored into timesteps.
        single_qubit: Single-qubit gates factored into timesteps.
    """

    two_qubit: list
    single_qubit: list


def _disjoint_nc2_combinations(
    qubit_pairs: Sequence[QidIntPair],
) -> list[tuple[QidIntPair, QidIntPair]]:
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
                Can be directed or undirected.
        """

        self.device_graph = device_graph

    def __call__(
        self,
        circuit: cirq.AbstractCircuit,
        *,
        config: RoutingConfig | None = None,
        initial_mapper: cirq.AbstractInitialMapper | None = None,
        context: cirq.TransformerContext | None = None,
    ) -> cirq.AbstractCircuit:
        """Transforms the given circuit to make it executable on the device.

        This method calls self.route_circuit and returns the routed circuit. See docstring of
        `RouteCQC.route_circuit` for more details on routing.

        Args:
            circuit: the input circuit to be transformed.
            config: routing configuration containing lookahead_radius and tag_inserted_swaps.
                If not provided, uses default RoutingConfig.
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
            circuit=circuit, config=config, initial_mapper=initial_mapper, context=context
        )
        return routed_circuit

    def route_circuit(
        self,
        circuit: cirq.AbstractCircuit,
        *,
        config: RoutingConfig | None = None,
        initial_mapper: cirq.AbstractInitialMapper | None = None,
        context: cirq.TransformerContext | None = None,
    ) -> tuple[cirq.AbstractCircuit, dict[cirq.Qid, cirq.Qid], dict[cirq.Qid, cirq.Qid]]:
        """Transforms the given circuit to make it executable on the device.

        This transformer assumes that all multi-qubit operations have been decomposed into 2-qubit
        operations and will raise an error if `circuit` a n-qubit operation where n > 2. If
        `circuit` contains `cirq.CircuitOperation`s and `context.deep` is True then they are first
        unrolled before proceeding. If `context.deep` is False or `context` is None then any
        `cirq.CircuitOperation` that acts on more than 2-qubits will also raise an error.

        The algorithm tries to find the best swap at each timestep by ranking a set of candidate
        swaps against operations starting from the current timestep (say s) to the timestep at index
        s + `config.lookahead_radius` to prune the set of candidate swaps. If it fails to converge
        to a single swap because of highly symmetrical device or circuit connectivity, then symmetry
        breaking strategies are used.

        Since routing doesn't necessarily modify any specific operation and only adds swaps
        before / after operations to ensure the circuit can be executed, tagging operations with
        tags from context.tags_to_ignore will have no impact on the routing procedure.

        Args:
            circuit: the input circuit to be transformed.
            config: routing configuration containing lookahead_radius and tag_inserted_swaps.
                If not provided, uses default RoutingConfig.
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
        if config is None:
            config = RoutingConfig()

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
        circuit_ops = CircuitOps(*self._get_one_and_two_qubit_ops_as_timesteps(circuit))

        # 4. Do the routing and save the routed circuit as a list of moments.
        routed_ops = self._route(mm, circuit_ops, config, self.device_graph)

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
        cls, circuit: cirq.AbstractCircuit
    ) -> tuple[list[list[cirq.Operation]], list[list[cirq.Operation]]]:
        """Gets the single and two qubit operations of the circuit factored into timesteps.

        The i'th entry in the nested two-qubit and single-qubit ops correspond to the two-qubit
        gates and single-qubit gates of the i'th timesteps respectively. When constructing the
        output routed circuit, single-qubit operations are inserted before two-qubit operations.

        Raises:
            ValueError: if circuit has intermediate measurements that act on three or more
                        qubits with a custom key.
        """
        two_qubit_circuit = circuits.Circuit()
        single_qubit_ops: list[list[cirq.Operation]] = []

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
                    elif key in ("", default_key):
                        single_qubit_ops[timestep].extend(ops.measure(qubit) for qubit in op.qubits)
                    else:
                        raise ValueError(
                            "Intermediate measurements on three or more qubits "
                            "with a custom key are not supported"
                        )
                elif protocols.num_qubits(op) == 2:
                    two_qubit_circuit[timestep] = two_qubit_circuit[timestep].with_operation(op)
                else:
                    single_qubit_ops[timestep].append(op)
        two_qubit_ops = [list(m) for m in two_qubit_circuit]
        return two_qubit_ops, single_qubit_ops

    @classmethod
    def emit_swap(
        cls,
        circuit_ops: list[cirq.Operation],
        qubit_pair: tuple[cirq.Qid, cirq.Qid],
        device_graph: nx.Graph,
        tag_inserted_swaps: bool = False,
    ) -> None:
        """Inserts a SWAP (or directed decomposition) between the qubit pair.

        For bidirectional edges, uses standard SWAP.
        For unidirectional edges, uses Hadamard-based decomposition.

        Args:
            circuit_ops: List of operations to append the swap to.
            qubit_pair: Tuple of (q1, q2) qubits to swap.
            device_graph: The device connectivity graph.
            tag_inserted_swaps: Whether to tag inserted swaps with RoutingSwapTag.
        """
        q1, q2 = qubit_pair

        # Helper to conditionally tag an operation
        def tag(op: cirq.Operation) -> cirq.Operation:
            return op.with_tags(ops.RoutingSwapTag()) if tag_inserted_swaps else op

        has_forward = device_graph.has_edge(q1, q2)
        has_reverse = device_graph.has_edge(q2, q1)

        if has_forward and has_reverse:
            # Bidirectional: use standard SWAP (decomposes to 3 CNOTs automatically)
            circuit_ops.append(tag(ops.SWAP(q1, q2)))

        elif has_forward or has_reverse:
            # Unidirectional: decompose SWAP using Hadamard trick
            # SWAP = CNOT - (H⊗H - CNOT - H⊗H) - CNOT
            ctrl, tgt = (q1, q2) if has_forward else (q2, q1)
            circuit_ops.append(tag(ops.CNOT(ctrl, tgt)))
            circuit_ops.extend([tag(ops.H(ctrl)), tag(ops.H(tgt))])
            circuit_ops.append(tag(ops.CNOT(ctrl, tgt)))
            circuit_ops.extend([tag(ops.H(ctrl)), tag(ops.H(tgt))])
            circuit_ops.append(tag(ops.CNOT(ctrl, tgt)))

        else:
            raise ValueError(f"No edge between {q1} and {q2} in device graph.")

    @classmethod
    def _route(
        cls,
        mm: mapping_manager.MappingManager,
        circuit_ops: CircuitOps,
        config: RoutingConfig,
        device_graph: nx.Graph,
    ) -> list[list[cirq.Operation]]:
        """Main routing procedure that inserts necessary swaps on the given timesteps.

        The i'th element of the returned list corresponds to the routed operations in the i'th
        timestep.

        Args:
            mm: Mapping manager for qubit mappings.
            circuit_ops: Circuit operations (two-qubit and single-qubit) factored into timesteps.
            config: routing configuration containing lookahead_radius and tag_inserted_swaps.
            device_graph: the device connectivity graph.

        Returns:
            a list of lists corresponding to timesteps of the routed circuit.
        """
        ops_ints: list[list[QidIntPair]] = [
            [
                (mm.logical_qid_to_int[op.qubits[0]], mm.logical_qid_to_int[op.qubits[1]])
                for op in timestep_ops
            ]
            for timestep_ops in circuit_ops.two_qubit
        ]
        routed: list[list[cirq.Operation]] = []

        def process_executable(t: int) -> int:
            unexec, unexec_ints = [], []
            for op, ints in zip(circuit_ops.two_qubit[t], ops_ints[t]):
                if mm.is_adjacent(*ints):
                    routed[t].append(mm.mapped_op(op))
                else:
                    unexec.append(op)
                    unexec_ints.append(ints)
            circuit_ops.two_qubit[t], ops_ints[t] = unexec, unexec_ints
            return len(unexec)

        for t in range(len(circuit_ops.two_qubit)):
            routed.append([mm.mapped_op(op) for op in circuit_ops.single_qubit[t]])
            seen: set[tuple[QidIntPair, ...]] = set()

            while process_executable(t):
                swaps: tuple[QidIntPair, ...] | None = None
                ctx = SwapSearchContext(t, config.lookahead_radius)
                for strat in (cls._choose_single_swap, cls._choose_pair_of_swaps):
                    swaps = strat(mm, ops_ints, ctx)
                    if swaps is not None:
                        break

                if swaps is None or swaps in seen:
                    swaps = cls._brute_force_strategy(mm, ops_ints, t)
                else:
                    seen.add(swaps)

                for swap in swaps:
                    cls.emit_swap(
                        routed[t],
                        (
                            mm.int_to_physical_qid[mm.logical_to_physical[swap[0]]],
                            mm.int_to_physical_qid[mm.logical_to_physical[swap[1]]],
                        ),
                        device_graph,
                        config.tag_inserted_swaps,
                    )
                    mm.apply_swap(*swap)

        return routed

    @classmethod
    def _brute_force_strategy(
        cls,
        mm: mapping_manager.MappingManager,
        two_qubit_ops_ints: Sequence[Sequence[QidIntPair]],
        timestep: int,
    ) -> tuple[QidIntPair, ...]:
        """Inserts SWAPS along the shortest path of the qubits that are the farthest.

        Since swaps along the shortest path are being executed one after the other, in order
        to achieve the physical swaps (M[q1], M[q2]), (M[q2], M[q3]), ..., (M[q_{i-1}], M[q_i]),
        we must execute the logical swaps (q1, q2), (q1, q3), ..., (q_1, qi).
        """
        furthest_op = max(two_qubit_ops_ints[timestep], key=lambda op: mm.dist_on_device(*op))
        path = mm.shortest_path(*furthest_op)
        return tuple((path[0], path[i + 1]) for i in range(len(path) - 2))

    @classmethod
    def _choose_pair_of_swaps(
        cls,
        mm: mapping_manager.MappingManager,
        two_qubit_ops_ints: Sequence[Sequence[QidIntPair]],
        ctx: SwapSearchContext,
    ) -> tuple[QidIntPair, ...] | None:
        """Computes cost function with pairs of candidate swaps that act on disjoint qubits."""
        pair_sigma = _disjoint_nc2_combinations(
            cls._initial_candidate_swaps(mm, two_qubit_ops_ints[ctx.timestep])
        )
        return cls._choose_optimal_swap(mm, two_qubit_ops_ints, ctx, pair_sigma)

    @classmethod
    def _choose_single_swap(
        cls,
        mm: mapping_manager.MappingManager,
        two_qubit_ops_ints: Sequence[Sequence[QidIntPair]],
        ctx: SwapSearchContext,
    ) -> tuple[QidIntPair, ...] | None:
        """Computes cost function with list of single candidate swaps."""
        sigma: list[tuple[QidIntPair, ...]] = [
            (swap,) for swap in cls._initial_candidate_swaps(mm, two_qubit_ops_ints[ctx.timestep])
        ]
        return cls._choose_optimal_swap(mm, two_qubit_ops_ints, ctx, sigma)

    @classmethod
    def _choose_optimal_swap(
        cls,
        mm: mapping_manager.MappingManager,
        two_qubit_ops_ints: Sequence[Sequence[QidIntPair]],
        ctx: SwapSearchContext,
        sigma: Sequence[tuple[QidIntPair, ...]],
    ) -> tuple[QidIntPair, ...] | None:
        """Optionally returns the swap with minimum cost from a list of n-tuple candidate swaps.

        Computes a cost (as defined by the overridable function `_cost`) for each candidate swap
        in the current timestep. If there does not exist a unique list of swaps with minimal cost,
        proceeds to rank the subset of minimal swaps from the current timestep in the next
        timestep. Iterates this looking ahead process up to the next `lookahead_radius`
        timesteps. If there still doesn't exist a unique swap with minimal cost then returns None.
        """
        end = min(ctx.lookahead_radius + ctx.timestep, len(two_qubit_ops_ints))
        for s in range(ctx.timestep, end):
            if len(sigma) <= 1:
                break

            costs = {}
            for swaps in sigma:
                costs[swaps] = cls._cost(mm, swaps, two_qubit_ops_ints[s])
            _, min_cost = min(costs.items(), key=lambda x: x[1])
            sigma = [swaps for swaps, cost in costs.items() if cost == min_cost]

        return (
            None
            if len(sigma) > 1 and ctx.timestep + ctx.lookahead_radius <= len(two_qubit_ops_ints)
            else sigma[0]
        )

    @classmethod
    def _initial_candidate_swaps(
        cls, mm: mapping_manager.MappingManager, two_qubit_ops: Sequence[QidIntPair]
    ) -> list[QidIntPair]:
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
        swaps: tuple[QidIntPair, ...],
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
        return f"cirq.RouteCQC(nx.Graph({dict(self.device_graph.adjacency())}))"
