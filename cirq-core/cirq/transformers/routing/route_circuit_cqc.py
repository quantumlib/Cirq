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

"""Routing transformer implementation based on arxiv:1902.08091."""

import networkx as nx

from typing import List, Tuple, Dict, Optional, TYPE_CHECKING
from itertools import combinations
from cirq import AbstractInitialMapper, MappingManager

if TYPE_CHECKING:
    import cirq


@cirq.transformer
class RouteCQC: 
    #TODO: add docstring explaining how algorithm works here.
    """Initializes the circuit routing transformer."""
    def __init__(self, device: cirq.Device):
        if nx.is_directed(device.metadata.nx_graph):
           raise ValueError("Device graph must be undirected")
        self.device = device
        self.device_graph = device.metadata.nx_graph
        self.circuit_graph = None
        self.initial_mapper = None
        self.mm = None


    def __call__(
      self,
      circuit: cirq.AbstractCircuit,
      *,
      max_search_radius: int = 10,
      preserve_moment_strucutre: bool = False,
      initial_mapper: Optional[AbstractInitialMapper] = None,
      context: Optional[cirq.TransformerContext] = None,
      ) -> cirq.AbstractCircuit:

        # 0. Handle CircuitOps by unrolling them
        if context is not None and context.deep == True:
            circuit = cirq.unroll_circuit_op(circuit, deep=True)
        if not all(len(op.qubits) <= 2 for op in circuit.all_operations()):
            raise ValueError("Input circuit must only have ops that act on 1 or 2 qubits")

        # 1. Do the initial mapping of logical to physical qubits
        #TODO: change this to default to LineInitialMapper once it is implemented
        assert initial_mapper is not None
        self.initial_mapper = initial_mapper

        # 2. Construct a mapping manager that implicitly keeps track of this mapping and provides
        # convinience methods over the image of the map on the device graph
        self.mm = MappingManager(self.device_graph, self.initial_mapper.initial_mapping())
        
        # 3. Get timesteps and single-qubit operations
        timesteps, single_qubit_ops = self._get_timesteps_and_single_qubit_ops(
            circuit, preserve_moment_strucutre
            )
        
        # 4. Do the routing and save the routed circuit as a list of moments
        routed_ops = self._route(timesteps, single_qubit_ops, max_search_radius)

        # 5. Return the routed circuit by packing each inner list of ops as densely as posslbe and 
        # preserving outer moment structure
        return cirq.Circuit(cirq.Circuit(m) for m in routed_ops)


    def _get_timesteps_and_single_qubit_ops(
        self,
        circuit: cirq.AbstractCircuit,
        preserve_moment_structure: bool,
        ) -> Tuple[List[List[cirq.Operation]], List[List[cirq.Operation]]]:
        """Returns the timesteps of the circuit and the single-qubit operations tagged with the
        timestep they are to be inserting in."""

        if preserve_moment_structure:
            single_qubit_operations = [[] for i in range(len(circuit))]

            def map_func_preserving(op: cirq.Operations, moment_index: int):
                if cirq.num_qubits(op) == 2:
                    return op 
                single_qubit_operations[moment_index].append(op)
                return []

            reduced_circuit = cirq.map_operations(circuit, map_func=map_func_preserving)
            return [list(moment.operations) for moment in reduced_circuit], single_qubit_operations

        reduced_circuit = cirq.Circuit()

        def map_func_not_preserving(op: cirq.Operation, moment_index: int):
            timestep_index = reduced_circuit.earliest_available_moment(op)
            if cirq.num_qubits(op) == 2:
                reduced_circuit.append(op)
            return op.with_tags(timestep_index)

        circuit_with_tags = cirq.map_operations(circuit, map_func=map_func_not_preserving)
        single_qubit_operations = [[] for i in range(len(reduced_circuit)+1)]
        for op in circuit_with_tags.all_operations():
            if len(op.qubits) == 1:
                timestep_index = op.tags[-1]
                single_qubit_operations[timestep_index].append(op)
        return [list(moment.operations) for moment in reduced_circuit], single_qubit_operations


    def _route(
        self,
        timesteps: List[List[cirq.Operation]],
        single_qubit_ops: List[List[cirq.Operation]],
        max_search_radius: int,
        ) ->  List[List[cirq.Operation]]:
        """Main routing procedure that creates the routed circuit, inserts the user's gates in it,
        and inserts the necessary swaps.
        
        Args:
          timesteps: the circuit's timesteps as defined by the paper
          max_search_radius: the maximum number of times we are allowed to iterate the cost
            function for convergence
            
        Returns:
          a list of lists corresponding to moments of the routed circuit
        """
        def process_executable_ops(idx: int):
            unexecutable_ops = []
            for op in timesteps[idx]:
                if self.mm.can_execute(op):
                    routed_ops[idx].append(self.mm.mapped_op(op))
                else:
                    unexecutable_ops.append(op)
            timesteps[idx] = unexecutable_ops

        routed_ops: List[List[cirq.Operation]] = [[] for i in range(len(timesteps)+1)]
        for idx in range(len(timesteps)):
            # add single qubit ops
            routed_ops[idx].extend([self.mm.mapped_op(op) for op in single_qubit_ops[idx]])

            process_executable_ops(idx)
            while len(timesteps[idx]) != 0:
                sigma = self._initial_candidate_swaps(timesteps[idx])
                for s in range(idx, min(max_search_radius + idx, len(timesteps))):
                    if len(sigma) <= 1:
                        break
                    sigma = self._next_candidate_swaps(sigma, timesteps[s])

                if (len(sigma) > 1 and idx + max_search_radius <= len(timesteps)):
                    chosen_swaps = self._symmetry_swap_pair(timesteps, idx, max_search_radius)
                else:
                    chosen_swaps = [sigma[0][0]]

                for swap in chosen_swaps:
                    routed_ops[idx].append(self.mm.mapped_op(cirq.SWAP(*swap).with_tags('s')))
                    self.mm.apply_swap(*swap)
                process_executable_ops(idx)

        # edge case: there may be a single qubit gate that act on the same qubit as a 2-qubit gate
        # in the last moment of the circuit
        routed_ops[len(timesteps)].extend(
            self.mm.mapped_op(op) for op in single_qubit_ops[len(timesteps)]
            )

        return routed_ops

    def _disjoint_qubit_combinations(
        self,
        single_candidates: List[Tuple[Tuple['cirq.Qid', 'cirq.Qid'],...]],
        ) ->List[Tuple[Tuple['cirq.Qid', 'cirq.Qid'],...]]:
        single_candidates = [x[0] for x in single_candidates]
        return [
            pair for pair in combinations(single_candidates, 2)
            if set(q for q in pair[0]).isdisjoint(set(q for q in pair[1]))
            ]


    def _symmetry_swap_pair(
        self,
        timesteps: List[List[cirq.Operation]],
        idx: int,
        max_search_radius: int,
        ) -> List[Tuple[cirq.Qid, cirq.Qid]]:
        """Computes cost function with pairs of candidate swaps that act on disjoint qubits."""
        pair_sigma = self._disjoint_qubit_combinations(
            self._initial_candidate_swaps(timesteps[idx])
            )
        for s in range(idx, min(max_search_radius + idx, len(timesteps))):
            if len(pair_sigma) <= 1:
                break
            pair_sigma = self._next_candidate_swaps(pair_sigma, timesteps[s])

        if (len(pair_sigma) > 1 and idx + max_search_radius <= len(timesteps)):
            return self._symmetry_brute_force(timesteps, idx)
        chosen_swap_pair = pair_sigma[0]
        return [chosen_swap_pair[0], chosen_swap_pair[1]]
        


    def _symmetry_brute_force(
        self,
        timesteps: List[List[cirq.Operation]],
        idx: int,
        ) -> List[Tuple[cirq.Qid, cirq.Qid]]:
        """Inserts SWAPS along the shortest path of the qubits that are the farthest."""
        qubits = max(
            [(op.qubits, self.mm.dist_on_device(*op.qubits)) for op in timesteps[idx]],
            key=lambda x: x[1]
            )[0]
        path = self.mm.shortest_path(*qubits)        
        q1 = self.mm.inverse_map[path[0]]
        return [(q1, path[i+1]) for i in range(len(path)-2)]


    def _initial_candidate_swaps(
        self,
        timestep_ops: List[cirq.Operation],
        ) -> List[Tuple[Tuple['cirq.Qid', 'cirq.Qid'],...]]:
        """Finds all feasible SWAPs between qubits involved in 2-qubit operations."""
        physical_qubits = set(
            self.mm.map[op.qubits[i]]
            for op in timestep_ops for i in range(2)
            ) 
        physical_swaps = self.mm.induced_subgraph.edges(nbunch=physical_qubits)
        return  [
            ((self.mm.inverse_map[q1], self.mm.inverse_map[q2]),)
            for q1, q2 in physical_swaps
            ]


    def _next_candidate_swaps(
        self,
        candidate_swaps: List[Tuple[Tuple['cirq.Qid', 'cirq.Qid'],...]],
        timestep_ops: List[cirq.Operation],
        ) -> List[Tuple[Tuple['cirq.Qid', 'cirq.Qid'],...]]:
        """Iterates the heuristic function.
        
        Given a list of candidate swaps find a subset that leads to a minimal longest shortest path
        between any paired qubits in the curernt timestep. 
        """ 
        costs = {swap: self._cost(swap, timestep_ops) for swap in candidate_swaps}
        return [swap for swap in costs.keys() if costs[swap] == min(costs.values())]


    def _cost(
        self,
        swaps: Tuple[Tuple['cirq.Qid', 'cirq.Qid'],...],
        timestep_ops: List[cirq.Operation],
        ) -> int:
        """Computes the cost function for the given list of swaps over the current timestep ops."""
        for swap in swaps:
            self.mm.apply_swap(*swap)
        shortest_path_lengths = [self.mm.dist_on_device(*op.qubits) for op in timestep_ops]
        for swap in swaps:
            self.mm.apply_swap(*swap)
        return max(shortest_path_lengths)


    @property
    def final_mapping(self) -> Dict['cirq.Qid', 'cirq.Qid']:
        """The final mapping from qubits in given circuit to qubits in the routed circuit"""
        if self.mm is None:
            raise ValueError("Cannot get final_mapping before calling the transformer.")
        return self.mm.map


    @property
    def initial_mapping(self) -> Dict['cirq.Qid', 'cirq.Qid']:
        """The initial mapping from qubits in given circuit to qubits in the routed circuit"""
        return self.initial_mapper.initial_mapping()
