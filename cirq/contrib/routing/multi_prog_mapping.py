# Copyright 2020 The Cirq Developers
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
"""
This module implements a multi-program mapping algorithm that is introduced in https://arxiv.org/abs/2004.12854.
To improve overall throughput and resource utilization, we can map muliple programs to a specific quantum chip at the same time. 
This paper proposes a new approach to map concurrent quantum programs. This approachhas some critical components. 
The first one is the Community Detection Assisted Partition (CDAP) algorithm, which partitions physical qubits for 
concurrent quantum programs by considering both physical typology (device graph) and the error rates (calibration data), 
avoiding the waste of robust resources. The second one is the X-SWAP scheme that enables inter-program SWAP operations besides 
intra-program SWAPs to reduce the SWAP overheads.
 
"""

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

import networkx as nx
import numpy as np

import cirq
import cirq.contrib.routing as ccr
from cirq import circuits, ops, value

SWAPTypeLogical = Tuple[Tuple[ops.Qid, int], Tuple[ops.Qid, int]]


class HierarchyTree:
    """
    Create a dendrogram tree using coupling graph and calibration data.
    Each node of this tree shows which physical qubits are better to be in same group.

    Args:
        device_graph: coupling graph of physical device
        single_er: a dictionary that shows operation error of single-qubit gates
        two_er: a dictionary that shows operation error of two-qubits gates
    """

    def __init__(
        self,
        device_graph: nx.Graph,
        single_er: Dict[
            Tuple[
                ops.Qid,
            ],
            List[float],
        ],
        two_er: Dict[Tuple[ops.Qid, ops.Qid], List[float]],
    ):
        self.device_graph = device_graph
        self.single_er = single_er
        self.two_er = two_er

    def compute_F(
        self,
        community1: List[ops.Qid],
        community2: List[ops.Qid],
        q1: float,
        q2: float,
        fidelity: float,
    ) -> Tuple[float, float]:
        """
        Computes the reward function F, F = Qmerged - Qorigin + (omega)*E*V
        Qorigin, Qmerged denote the modularity of the original partition and the new partition after merging the two communities.
        w is a weight parameter.
        E denotes the average fidelity of two-qubits gates (e.g. CNOTs).
        V denotes the average fidelity of readout operations on the qubits connecting the two communities.

        Args:
            community1: first community
            community2: second community
            q1: original value of Qorigin for first community
            q2: original value of Qorigin for second community
            fidelity: total value of E*V

        Return:
            F_value: the value of reward function
            q_merged: the modularity value of 2 combined communities

        """

        edges_count = len(self.device_graph.edges)
        omega = 1.0
        inside_edges = 0
        outside_edges = 0
        total_edges = 0
        # Compute inside edges
        for c1 in community1:
            for c2 in community2:
                if self.device_graph.has_edge(c1, c2):
                    inside_edges += 1
        for i in range(len(community1) - 1):
            for j in range(i + 1, len(community1), 1):
                if self.device_graph.has_edge(community1[i], community1[j]):
                    inside_edges += 1
        for i in range(len(community2) - 1):
            for j in range(i + 1, len(community2), 1):
                if self.device_graph.has_edge(community2[i], community2[j]):
                    inside_edges += 1
        # Compute total edges
        for c1 in community1:
            total_edges += len(c1.neighbors())
        for c2 in community2:
            total_edges += len(c2.neighbors())
        # Compute outside edges
        outside_edges = total_edges - 2 * inside_edges

        q_merged = float(inside_edges / edges_count) - pow(float(outside_edges / edges_count), 2)
        delta_q = q_merged - q1 - q2
        F_value = delta_q + omega * fidelity

        return F_value, q_merged

    def compute_fidelity_of2merged_communities(
        self, community1: List[ops.Qid], community2: List[ops.Qid]
    ) -> float:
        """
        Computes fidelity of two communities after merging.
        This is due to check merging these communities is a good choice or not.

        Args:
            community1: first community
            community2: second community

        Return:
            fidelity: the value of computed fidelity.
        """

        fidelity = 0.0
        for c1 in community1:
            for c2 in community2:
                if (c1, c2) in self.two_er:
                    fidelity_two = 1 - self.two_er[(c1, c2)][0]
                    fidelity_single = 0.5 * (
                        1 - self.single_er[(c1,)][0] + 1 - self.single_er[(c2,)][0]
                    )
                    fidelity += fidelity_two * fidelity_single
                elif (c2, c1) in self.two_er:
                    fidelity_two = 1 - self.two_er[(c2, c1)][0]
                    fidelity_single = 0.5 * (
                        1 - self.single_er[(c1,)][0] + 1 - self.single_er[(c2,)][0]
                    )
                    fidelity += fidelity_two * fidelity_single

        return fidelity

    def compute_new_node(
        self, communities: List[List[ops.Qid]], q_values: List[float]
    ) -> Tuple[int, int, float]:
        """
        Find the best 2 communities (high fidelity) to merge and create new node.

        Args:
            communities: list of all communities
            q_values: list of modularity values correspond to communities

        Return:
            idx1: index of the first selected community
            idx2: index of the second selected community
            q_merged: the modularity value (Qorigin) of 2 selected communities after merge
        """

        idx1 = 0
        idx2 = 1
        q_merged = 0.0
        Fmax = -np.inf

        for i in range(len(communities) - 1):
            for j in range(i + 1, len(communities), 1):
                fidelity = self.compute_fidelity_of2merged_communities(
                    communities[i], communities[j]
                )
                if fidelity != 0.0:
                    F, q_merged_temp = self.compute_F(
                        communities[i], communities[j], q_values[i], q_values[j], fidelity
                    )
                    if F > Fmax:
                        Fmax = F
                        idx1 = i
                        idx2 = j
                        q_merged = q_merged_temp

        if idx1 > idx2:
            return idx1, idx2, q_merged
        else:
            return idx2, idx1, q_merged

    def tree_construction(self) -> nx.DiGraph:
        """
        Construct a dendrogram tree of physical qubits.
        This function uses device graph and calibration data togeteher
        to specify wich physical qubits can be in same community to assign
        them to programs (qunatum circuits).

        Return:
            tree: constructed dendrogram tree
        """

        tree = nx.DiGraph()
        label = 0
        communities = []

        communities = [[i] for i in sorted(list(self.device_graph.nodes))]
        q_values = [0.0] * len(communities)

        while len(communities) > 1:
            idx1, idx2, q_merged = self.compute_new_node(communities, q_values)
            new_node_list = sorted(communities[idx1] + communities[idx2])
            new_node = tuple(new_node_list)
            tree.add_edge(new_node, tuple(communities[idx1]))
            tree.add_edge(new_node, tuple(communities[idx2]))

            communities.pop(idx1)
            communities.pop(idx2)
            q_values.pop(idx1)
            q_values.pop(idx2)
            communities.append(new_node_list)
            q_values.append(q_merged)

        return tree


class QubitsPartitioning:
    """
    Partition physical qubits to some groups using constrcuted heirarcical tree
    and assign them to different programs.

    Args:
        tree: constrcuted heirarchical tree of physical qubits
        program_circuits: a list of multiple program to run at the same time
        single_er: readout error corresponding to each physical qubit
        two_er: gate error on connected link between 2 physical qubits
        twoQ_gate_type: type of 2-qubits gate (CZ or CNOT)
    """

    def __init__(
        self,
        tree: nx.DiGraph,
        program_circuits: List[circuits.Circuit],
        single_er: Dict[
            Tuple[
                ops.Qid,
            ],
            List[float],
        ],
        two_er: Dict[Tuple[ops.Qid, ops.Qid], List[float]],
        twoQ_gate_type: ops.Gate,
    ):
        self.tree = tree
        self.program_circuits = program_circuits
        self.single_er = single_er
        self.two_er = two_er
        self.twoQ_gate_type = twoQ_gate_type

    def circuits_descending(self) -> List[circuits.Circuit]:
        """
        Sort programs with their density (#CNOTs/#qubits)

        Return:
            circuits_desc: list of circuits in descending order
        """

        cnot_density = []
        for circuit in self.program_circuits:
            density = len(
                list(circuit.findall_operations(lambda op: op.gate == self.twoQ_gate_type))
            ) / float(len(circuit.all_qubits()))
            cnot_density.append(density)
        # Computing indices regarding descending order of cnot densities
        idxs = np.argsort(cnot_density)
        idxs_descending = np.flip(idxs)
        # Reorder list of program_circuits
        circuits_desc = []
        for id in idxs_descending:
            circuits_desc.append(self.program_circuits[id])

        return circuits_desc

    def compute_EPST(self, partition: List[ops.Qid], cir: circuits.Circuit) -> float:
        """
        Compute Estimated Probability of a Successful Trial (EPST).
        EPST estimates the fidelity of the executtion of a quantum program on a specific quantum chip.
        EPST = (f_2q^#CNOTs) * (f_1q^#1q_gates) * (f_ro^#qubits)
        f_2q: the average fidelity of CNOTs
        f_1q: the average fidelity of 1-qubit gates
        f_ro: the average fidelity of readout operations

        Args:
            partition: a group of physical qubits
            cir: program circuit
        """

        twoQ_gates = len(list(cir.findall_operations(lambda op: op.gate == self.twoQ_gate_type)))
        twoQ_gates = len(list(cir.all_operations())) - twoQ_gates
        qubits = len(cir.all_qubits())

        err_2 = 0.0
        count_2 = 0
        err_1 = 0.0
        for i in range(len(partition) - 1):
            for j in range(i + 1, len(partition)):
                if (partition[i], partition[j]) in self.two_er:
                    err_2 = err_2 + self.two_er[(partition[i], partition[j])][0]
                    count_2 = count_2 + 1
                elif (partition[j], partition[i]) in self.two_er:
                    err_2 = err_2 + self.two_er[(partition[j], partition[i])][0]
                    count_2 = count_2 + 1
        for i in range(len(partition)):
            err_1 = err_1 + self.single_er[(partition[i],)][0]

        avgF_2 = 1 - float(err_2 / count_2)
        avgF_1 = 1 - float(err_1 / len(partition))
        # to do : readout error
        epst = pow(avgF_2, twoQ_gates) * pow(avgF_1, twoQ_gates) * pow(avgF_1, qubits)

        return epst

    def find_best_partition(
        self, cands: List[List[ops.Qid]], cir: circuits.Circuit
    ) -> List[ops.Qid]:
        """
        Find best partition for current circuit based on average fidelity.

        Args:
            cands: list of diifferent partitions
            cir: program circuit
        """

        best_cand = cands[0]
        max_f = -np.inf

        for cand in cands:
            epst = self.compute_EPST(cand, cir)
            if epst > max_f:
                max_f = epst
                best_cand = cand

        return best_cand

    def qubits_allocation(self, desc_prog_circuits: List[circuits.Circuit]) -> List[List[ops.Qid]]:
        """
        Allocate different groups of physical qubits to different programs.

        Args:
            desc_prog_circuits: list of different programs

        Return:
            partitions: list of different partitions corresponding to programs
        """

        partitions = []

        for cir in desc_prog_circuits:
            candidates: List[List[ops.Qid]] = []
            leaves = [
                x
                for x in self.tree.nodes()
                if self.tree.out_degree(x) == 0 and self.tree.in_degree(x) == 1
            ]
            for leaf in leaves:
                while leaf is not None:
                    if len(cir.all_qubits()) <= len(list(leaf)):
                        exist = 0
                        for c in candidates:
                            if c == leaf:
                                exist = 1
                                break
                            elif set(list(c)).issubset(leaf):
                                # Keep independent candidates
                                exist = 1
                                break
                        if not exist:
                            candidates.append(leaf)
                        break
                    else:
                        leaf = tuple(self.tree.predecessors(leaf))[0]

            if len(candidates) == 0:
                print("fail -- run programs seperately")

            best_cand = self.find_best_partition(candidates, cir)
            partitions.append(best_cand)
            # Remove nodes from tree & relabel remaining nodes
            successors = list(self.tree.successors(best_cand))
            self.tree.remove_nodes_from(successors)
            self.tree.remove_node(best_cand)
            label_mapping = {}
            for n in list(self.tree.nodes()):
                label_mapping[n] = tuple(x for x in n if x not in best_cand)

            nx.relabel_nodes(self.tree, label_mapping)

        return partitions


class XSWAP:
    """
    This class apply XSWAP algorithm to generate the final schedule
    for running multiple programs at the same time.
    XSWAP uses both inter-program and intra-program swaps to satisfy hardware constraints.

    Args:
        device_graph: physical coupling graph
        desc_program_circuits: list of different programs in descending order based on CNOT density
        partitions: groups of physical qubits corresponding to programs
        twoQ_gate_type: type of 2-qubits gate (CZ or CNOT)
        l_to_ph: mapping logical qubits and their program id to physical qubits
        ph_to_l: mapping physical qubits to logical qubits and their program id
    """

    def __init__(
        self,
        device_graph: nx.Graph,
        desc_program_circuits: List[circuits.Circuit],
        partitions: List[List[ops.Qid]],
        twoQ_gate_type: ops.Gate,
        l_to_ph={},
        ph_to_l={},
    ):
        self.device_graph = device_graph
        self.desc_program_circuits = desc_program_circuits
        self.partitions = partitions
        self.twoQ_gate_type = twoQ_gate_type
        self.l_to_ph = l_to_ph
        self.ph_to_l = ph_to_l

    def generate_2qGates_dags(self) -> List[cirq.CircuitDag]:
        """
        Generate the dag representation of program circuits only for 2-qubits gates.

        Return:
            cir_dags: list of generated dags
        """

        cir_dags = []
        for c in self.desc_program_circuits:
            # Remove single qubit gates before creating dag
            singleq_gates = list(c.findall_operations_with_gate_type(self.twoQ_gate_type))
            c.batch_remove(singleq_gates)
            # Create dag
            cir_dags.append(cirq.CircuitDag.from_circuit(c))

        return cir_dags

    def generate_dags(self) -> List[cirq.CircuitDag]:
        """
        Generate the dag representation of program circuits consisting of all gate types.

        Return:
            cir_dags: list of generated dags
        """

        cir_dags = []
        for c in self.desc_program_circuits:
            # Create dag
            cir_dags.append(cirq.CircuitDag.from_circuit(c))

        return cir_dags

    def generate_2qGates_front_layers(
        self, cir_dags: List[cirq.CircuitDag]
    ) -> List[List[ops.Operation]]:
        """
        Generate front layers for all programs by considering only 2-qubits gates. Front layer is consisting of all gates
        that could run and doesn't have any predecessors.

        Args:
            cir_dags: dag representation of all programs
        Return:
            flayers: list of front layers corresponding to programs
        """

        flayers = []
        for dag in cir_dags:
            fl = []
            nodes = list(dag.ordered_nodes())
            ops = list(dag.all_operations())
            for i in range(len(nodes)):
                if len(list(dag.predecessors(nodes[i]))) == 0:
                    fl.append(ops[i])
            flayers.append(fl)

        return flayers

    def generate_front_layers(self, cir_dags: List[cirq.CircuitDag]) -> List[List[ops.Operation]]:
        """
        Generate front layers for all programs. Front layer is consisting of all gates
        that could run and doesn't have any predecessors.

        Args:
            cir_dags: dag representation of all programs
        Return:
            flayers: list of front layers corresponding to programs
        """

        # Check all dags are empty
        counter = 0
        for dag in cir_dags:
            nodes = list(dag.ordered_nodes())
            if len(nodes) == 0:
                counter = counter + 1
        if counter == len(cir_dags):
            return None
        # Set of nodes
        flayers = []
        for dag in cir_dags:
            fl = []
            nodes = list(dag.ordered_nodes())
            # ops = list(dag.all_operations()) ??
            for i in range(len(nodes)):
                if len(list(dag.predecessors(nodes[i]))) == 0:
                    fl.append(nodes[i])  # .val  # fl.append(ops[i]) ??
                else:
                    counter = 0
                    for p in list(dag.predecessors(nodes[i])):
                        if p.val.gate != self.twoQ_gate_type:
                            counter = counter + 1
                    if counter == len(list(dag.predecessors(nodes[i]))):
                        fl.append(nodes[i])  # .val  # fl.append(ops[i]) ??

            flayers.append(fl)

        return flayers

    def generate_second_layer(self, cir_dag):
        secondl = []
        ## to do

        return secondl

    def initial_mapping(self) -> None:
        """
        Initialize 2 mapping dictionaries l_to_ph and ph_to_l.
        """

        total_l_qubits = 0
        total_ph_qubits = len(self.device_graph.nodes)
        for i in range(len(self.desc_program_circuits)):
            logical_qubits = sorted(list(self.desc_program_circuits[i].all_qubits()))
            total_l_qubits = total_l_qubits + len(logical_qubits)
            for j in range(len(logical_qubits)):
                self.l_to_ph[(logical_qubits[j], i)] = self.partitions[i][j]
                self.ph_to_l[self.partitions[i][j]] = (logical_qubits[j], i)
        # Manage unused qubits and map them to a unreal program with id = -1
        l_qubits = cirq.LineQubit.range(total_ph_qubits - total_l_qubits)
        i = 0
        for ph_q in list(self.device_graph.nodes):
            if ph_q not in self.ph_to_l:
                self.ph_to_l[ph_q] = (l_qubits[i], -1)
                self.l_to_ph[(l_qubits[i], -1)] = ph_q
                i = i + 1
        print("logical to physical")
        print(self.l_to_ph)
        print("physical to logical")
        print(self.ph_to_l)

        return

    def log_to_phy_edge(
        self, log_edge: Tuple[ops.Qid, ops.Qid], pid: int
    ) -> Tuple[ops.Qid, ops.Qid]:
        """
        Map logical qubits and their program id of an edge to physical qubits and return it as an edge.

        Args:
            log_edge: 2 logical qubits connected by a link
            pid: program id

        Return:
            phy_edge: 2 physical qubits corresponding to logical edge
        """

        phy0 = self.l_to_ph[(log_edge[0], pid)]
        phy1 = self.l_to_ph[(log_edge[1], pid)]
        phy_edge = (phy0, phy1)

        return phy_edge

    def phy_to_log_edge(self, phy_edge: Tuple[ops.Qid, ops.Qid]) -> SWAPTypeLogical:
        """
        Map physical qubits of an edge to logical qubits and their program id and return it as an edge.

        Args:
            phy_edge: 2 physical qubits connected by a link

        Return:
            log_edge: 2 logical qubits and their program id corresponding to physical edge
        """

        log_pid0 = self.ph_to_l[phy_edge[0]]
        log_pid1 = self.ph_to_l[phy_edge[1]]
        log_edge = (log_pid0, log_pid1)

        return log_edge

    def obtain_swaps(self, node_gates: cirq.CircuitDag.nodes, pid: int) -> List[SWAPTypeLogical]:
        """
        Find SWAPs to enable gate execution.

        Args:
            node_gates: nodes of circuit dag
            pid: program id
        Returns:
            swaps: all possible SWAPs of gates corresponding to dag nodes for all programs
        """

        swaps = []
        for n in node_gates:
            g = n.val
            phy_qs = self.log_to_phy_edge(g.qubits, pid)
            neighbors0 = list(self.device_graph.neighbors(phy_qs[0]))
            neighbors1 = list(self.device_graph.neighbors(phy_qs[1]))
            for ne in neighbors0:
                swaps.append(self.phy_to_log_edge((phy_qs[0], ne)))
            for ne in neighbors1:
                swaps.append(self.phy_to_log_edge((phy_qs[1], ne)))
        return swaps

    def update_mapping(
        self, swap: SWAPTypeLogical
    ) -> Tuple[Dict[Tuple[ops.Qid, int], ops.Qid], Dict[ops.Qid, Tuple[ops.Qid, int]]]:
        """
        Update 2 mapping dictionaries after applying a SWAP.

        Args:
            swap: swaping of 2 logical qubits

        Returns:
            new_l_ph: mapping dictionary from logical to physical qubits
            new_ph_l: mapping dictionary from physical to logical qubits
        """

        new_ph_l = self.ph_to_l.copy()
        new_l_ph = self.l_to_ph.copy()

        ph0 = self.l_to_ph[swap[0]]
        ph1 = self.l_to_ph[swap[1]]
        new_l_ph[swap[0]] = ph1
        new_l_ph[swap[1]] = ph0
        new_ph_l[ph0] = self.ph_to_l[ph1]
        new_ph_l[ph1] = self.ph_to_l[ph0]

        return new_l_ph, new_ph_l

    def compute_H(
        self,
        flayers: List[List[ops.Gate]],
        new_l_ph: Dict[Tuple[ops.Qid, int], ops.Qid],
        new_ph_l: Dict[ops.Qid, Tuple[ops.Qid, int]],
    ) -> float:
        """
        Compute heuristic cost after inserting SWAP based on Nearest Neighbor Cost (NNC).
        Comes from https://arxiv.org/pdf/1809.02573.pdf

        Args:
            flayers: front layers
            new_l_ph: mapping dictionary from logical to physical qubits after applying SWAP
            new_ph_l: mapping dictionary from physical to logical qubits after applying SWAP

        Return:
            H_cost: cost value
        """

        H_cost = 0
        for i in range(len(flayers)):
            if len(flayers[i]) == 0:
                continue
            for n in flayers[i]:
                lq = n.val.qubits
                phy_edge = (new_l_ph[(lq[0], i)], new_l_ph[(lq[1], i)])
                H_cost = H_cost + len(
                    list(nx.all_shortest_paths(self.device_graph, phy_edge[0], phy_edge[1]))[0]
                )
        return H_cost - 1

    def compute_path_distance_in_sameP(self, edge: Tuple[ops.Qid, ops.Qid], pid: int) -> int:
        """
        Compute path distance of two qubits by considering qubits only in same program.

        Args:
            edge: 2 physical qubits that we need to be connected
            pid: program id

        Return:
            distance: distance value in device graph
        """

        paths = list(
            nx.all_simple_paths(self.device_graph, edge[0], edge[1], cutoff=None)
        )  # cutoff=len(self.partitions[pid]))

        distance = np.inf
        for p in paths:
            if set(p).issubset(set(self.partitions[pid])):
                if len(p) < distance:
                    distance = len(p)
        return distance

    def compute_gainCost(
        self,
        flayers: List[List[ops.Gate]],
        new_l_ph: Dict[Tuple[ops.Qid, int], ops.Qid],
        new_ph_l: Dict[ops.Qid, Tuple[ops.Qid, int]],
        swap: SWAPTypeLogical,
    ) -> float:
        """
        Compute Gain cost. It shows that inter-program SWAPs outperform intra-program SWAPs or not.

        Args:
            flayers: front layers
            new_l_ph: mapping dictionary from logical to physical qubits after applying SWAP
            new_ph_l: mapping dictionary from physical to logical qubits after applying SWAP
            swap: considered swap gate

        Return:
            gain_cost: gain cost value
        """

        gain_cost = 0.0

        for i in range(len(flayers)):
            if len(flayers[i]) == 0:
                continue
            cost_i = 0.0
            for n in flayers[i]:
                lq = n.val.qubits
                new_phy_edge = (new_l_ph[(lq[0], i)], new_l_ph[(lq[1], i)])
                new_paths = nx.all_shortest_paths(
                    self.device_graph, new_phy_edge[0], new_phy_edge[1]
                )

                phy_edge = (self.l_to_ph[(lq[0], i)], self.l_to_ph[(lq[1], i)])
                paths = nx.all_shortest_paths(self.device_graph, phy_edge[0], phy_edge[1])
                path_len = len(list(paths)[0])
                new_path_len = len(list(new_paths)[0])

                in_path = 0

                if new_path_len < path_len:
                    in_path = 1
                D_allp = new_path_len
                D_singlep = self.compute_path_distance_in_sameP(phy_edge, i)
                cost_i = cost_i + (D_allp - D_singlep) * in_path
            gain_cost = gain_cost + float(1.0 / len(flayers[i])) * cost_i
        return gain_cost

    def find_best_swap(
        self, swap_candidate_lists: List[List[SWAPTypeLogical]], flayers: List[List[ops.Gate]]
    ) -> SWAPTypeLogical:
        """
        Find best SWAP among candidates.

        Args:
            swap_candidate_lists: SWAP candidates
            flayers: front layers

        Return:
            best_swap: best SWAP gate
        """

        min_cost = np.inf
        best_swap = None
        for swaps in swap_candidate_lists:
            for s in swaps:
                new_l_ph, new_ph_l = self.update_mapping(s)
                H_cost = self.compute_H(flayers, new_l_ph, new_ph_l)
                gain_cost = self.compute_gainCost(flayers, new_l_ph, new_ph_l, s)
                cost = H_cost + gain_cost
                if cost < min_cost:
                    min_cost = cost
                    best_swap = s

        return best_swap

    def insert_SWAP_and_generate_schedule(self) -> circuits.Circuit:
        """
        Insert the required SWAP gates and generate the final schedule on physical qubits for all program at the same time.

        Return:
            schedule: final quantum circuit consisting of all programs
        """

        schedule = cirq.Circuit()
        dags = self.generate_dags()
        self.initial_mapping()

        flayers = self.generate_front_layers(dags)
        while flayers != None:
            # Solve hardware-compliant gates
            # i specify program index
            require_swap = 0

            for i in range(len(flayers)):
                if len(flayers[i]) == 0:
                    continue
                gate_nodes = flayers[i].copy()
                for n in gate_nodes:
                    if len(n.val.qubits) == 1:
                        g = n.val.gate
                        lq = n.val.qubit
                        phq = self.l_to_ph[(lq, i)]

                        schedule.append(g(phq))
                        dags[i].remove_node(n)
                        # Update front layer
                        flayers[i].remove(n)
                    else:
                        ph0 = self.l_to_ph[(n.val.qubits[0], i)]
                        ph1 = self.l_to_ph[(n.val.qubits[1], i)]
                        if (ph0, ph1) in self.device_graph.edges or (
                            ph1,
                            ph0,
                        ) in self.device_graph.edges:
                            g = n.val.gate
                            # lqs = n.val.qubits
                            # phq0 = self.l_to_ph[(lqs[0], i)]
                            # phq1 = self.l_to_ph[(lqs[1], i)]
                            schedule.append(g(ph0, ph1))

                            dags[i].remove_node(n)
                            # Update front layer
                            flayers[i].remove(n)
                        else:
                            # Rrequire SWAP
                            require_swap = 1
            # Solve hardware-incompliant gates by inserting SWAPs
            if require_swap:
                swap_candidate_lists: List[List[SWAPTypeLogical]] = []
                for i in range(len(flayers)):
                    if len(flayers[i]) == 0:
                        swap_candidate_lists.append([])
                        continue
                    critical_node_gates = flayers[i].copy()  # to do ??
                    swap_candidates = self.obtain_swaps(critical_node_gates, i)
                    swap_candidate_lists.append(swap_candidates)
                # Find best SWAP
                # List of 2 pairs: show each qubit belongs to which program id
                print(f"number of swaps: {len(swap_candidate_lists)}")
                best_swap = self.find_best_swap(swap_candidate_lists, flayers)
                print(f"best swap: {best_swap}")

                ph0 = self.l_to_ph[best_swap[0]]
                ph1 = self.l_to_ph[best_swap[1]]
                schedule.append(cirq.SWAP(ph0, ph1))
                # Update mapping
                ph0 = self.l_to_ph[best_swap[0]]
                ph1 = self.l_to_ph[best_swap[1]]
                self.l_to_ph[best_swap[0]] = ph1
                self.l_to_ph[best_swap[1]] = ph0
                self.ph_to_l[ph0] = self.ph_to_l[ph1]
                self.ph_to_l[ph1] = self.ph_to_l[ph0]

            flayers = self.generate_front_layers(dags)

        return schedule


############################################################


def multi_prog_map(
    device_graph: nx.Graph,
    single_er: Dict[
        Tuple[
            ops.Qid,
        ],
        List[float],
    ],
    two_er: Dict[Tuple[ops.Qid, ops.Qid], List[float]],
    prog_circuits: List[circuits.Circuit],
) -> None:
    """
    Initialize and run the procedure of mapping multiple programs to a same physical device step by step.

    Args:
        device_graph: graph of physical device
        single_er: a dictionary that shows operation error of single-qubit gates
        two_er: a dictionary that shows operation error of two-qubits gates
        prog_circuits: circuits of all programs
    """

    twoQ_gate_type = cirq.CZ
    tree_obj = HierarchyTree(device_graph, single_er, two_er)
    tree = tree_obj.tree_construction()
    print("tree")
    print(tree.nodes)

    par_obj = QubitsPartitioning(tree, prog_circuits, single_er, two_er, twoQ_gate_type)
    desc_cirs = par_obj.circuits_descending()

    partitions = par_obj.qubits_allocation(desc_cirs)

    print("partitions:")
    print(partitions)

    xswap = XSWAP(device_graph, desc_cirs, partitions, twoQ_gate_type)
    schedule = xswap.insert_SWAP_and_generate_schedule()
    print("schedule:")
    print(schedule)

    return partitions, schedule


def prepare_couplingGraph_errorValues(device_graph):
    # single_er = load_calibrations()['single_qubit_p00_error'] # to do ??
    # two_er = load_calibrations()['two_qubit_sycamore_gate_xeb_cycle_purity_error'] # to do ??

    single_er = {
        (cirq.GridQubit(1, 0),): [0.028600441075128205],
        (cirq.GridQubit(0, 0),): [0.01138359559038841],
        (cirq.GridQubit(1, 1),): [0.05313138858345922],
        (cirq.GridQubit(0, 1),): [0.0005880214404983153],
        (cirq.GridQubit(1, 2),): [0.0018232495924263727],
        (cirq.GridQubit(0, 2),): [0.039571298178797366],
    }
    two_er = {
        (cirq.GridQubit(1, 0), cirq.GridQubit(0, 0)): [0.018600441075128205],
        (cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)): [0.01938359559038841],
        (cirq.GridQubit(1, 1), cirq.GridQubit(0, 1)): [0.01313138858345922],
        (cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)): [0.005880214404983153],
        (cirq.GridQubit(1, 1), cirq.GridQubit(1, 2)): [0.008232495924263727],
        (cirq.GridQubit(0, 2), cirq.GridQubit(1, 2)): [0.03571298178797366],
    }

    # coupling graph
    dgraph = nx.Graph()

    for q0, q1 in two_er:
        dgraph.add_edge(q0, q1)

    # list of program circuits
    qubits = cirq.LineQubit.range(3)
    circuit1 = cirq.Circuit(
        cirq.X(qubits[0]),
        cirq.Y(qubits[1]),
        cirq.CZ(qubits[0], qubits[1]),
        cirq.CZ(qubits[0], qubits[2]),
    )
    qubits = cirq.LineQubit.range(2)
    circuit2 = cirq.Circuit(cirq.X(qubits[0]), cirq.Y(qubits[1]), cirq.CZ(qubits[0], qubits[1]))
    program_circuits = []
    program_circuits.append(circuit2)
    program_circuits.append(circuit1)

    print(circuit2)
    print(circuit1)

    multi_prog_map(dgraph, single_er, two_er, program_circuits)


if __name__ == "__main__":
    # print( load_calibrations()['single_qubit_p00_error'] )
    device_graph1 = ccr.get_grid_device_graph(3, 2)
    device_graph = cirq.google.Sycamore
    prepare_couplingGraph_errorValues(device_graph)
