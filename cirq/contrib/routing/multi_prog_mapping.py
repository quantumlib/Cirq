""" Author: Fereshte Mozafari """

import networkx as nx
import numpy as np
import math
from numpy import inf

import cirq
import cirq.contrib.routing as ccr


class Hierarchy_tree:

    def __init__(self, device_graph, single_er, two_er):
        self.device_graph = device_graph
        self.single_er = single_er
        self.two_er = two_er

    def compute_F(self, com1, com2, Q1, Q2, fidelity):
        edges_count = len(list(self.device_graph.edges))
        omega = 1.0
        inside_edges = 0
        outside_edges = 0
        total_edges = 0
        """ compute inside edges """
        for c1 in com1:
            for c2 in com2:
                if (self.device_graph.has_edge(c1, c2)):
                    inside_edges += 1
        for i in range(len(com1) - 1):
            for j in range(i + 1, len(com1), 1):
                if (self.device_graph.has_edge(com1[i], com1[j])):
                    inside_edges += 1
        for i in range(len(com2) - 1):
            for j in range(i + 1, len(com2), 1):
                if (self.device_graph.has_edge(com2[i], com2[j])):
                    inside_edges += 1
        """ compute total edges """
        for c1 in com1:
            total_edges += len(list(c1.neighbors()))
        for c2 in com2:
            total_edges += len(list(c2.neighbors()))
        """ compute outside edges """
        outside_edges = total_edges - 2 * inside_edges

        Qmerged = float(inside_edges / edges_count) - pow(
            float(outside_edges / edges_count), 2)
        deltaQ = Qmerged - Q1 - Q2
        F_value = deltaQ + omega * fidelity

        return F_value, Qmerged

    def find_edge_among_coms(self, com1, com2):
        fidelity = 0.0
        for c1 in com1:
            for c2 in com2:
                if (c1, c2) in self.two_er:
                    fidelity_two = 1 - self.two_er[(c1, c2)][0]
                    fidelity_single = 0.5 * (1 - self.single_er[(c1,)][0] + 1 -
                                             self.single_er[(c2,)][0])
                    fidelity += (fidelity_two * fidelity_single)
                elif (c2, c1) in self.two_er:
                    fidelity_two = 1 - self.two_er[(c2, c1)][0]
                    fidelity_single = 0.5 * (1 - self.single_er[(c1,)][0] + 1 -
                                             self.single_er[(c2,)][0])
                    fidelity += (fidelity_two * fidelity_single)

        return fidelity

    def compute_new_node(self, communities, Qvalues):
        idx1 = 0
        idx2 = 1
        Qmerged = 0.0
        Fmax = -math.inf

        for i in range(len(communities) - 1):
            for j in range(i + 1, len(communities), 1):
                fidelity = self.find_edge_among_coms(communities[i],
                                                     communities[j])
                if (fidelity != 0.0):
                    F, qmerged = self.compute_F(communities[i], communities[j],
                                                Qvalues[i], Qvalues[j],
                                                fidelity)
                    if (F > Fmax):
                        Fmax = F
                        idx1 = i
                        idx2 = j
                        Qmerged = qmerged

        if (idx1 > idx2):
            return idx1, idx2, Qmerged
        else:
            return idx2, idx1, Qmerged

    def tree_construction(self):
        tree = nx.DiGraph()
        label = 0
        communities = []
        #for n in list(self.device_graph.nodes):
        #tree.add_node(n)   #(label, data = [n])
        #label = label + 1

        communities = [[i] for i in list(self.device_graph.nodes)]
        Qvalues = [0.0] * len(communities)

        while len(communities) > 1:
            idx1, idx2, Qmerged = self.compute_new_node(communities, Qvalues)
            # new_node = []
            # new_node.append(communities[idx1])
            # new_node.append(communities[idx2])
            new_node_list = communities[idx1] + communities[idx2]
            #print(new_node_list)
            #tree[tuple(new_node_idx)] = new_node
            new_node = tuple(new_node_list)
            #tree.add_node(new_node) #(label , data = new_node_idx)
            tree.add_edge(new_node, tuple(communities[idx1]))
            tree.add_edge(new_node, tuple(communities[idx2]))

            communities.pop(idx1)
            communities.pop(idx2)
            Qvalues.pop(idx1)
            Qvalues.pop(idx2)
            communities.append(new_node_list)
            Qvalues.append(Qmerged)

        return tree


class Qubits_partitioning:

    def __init__(self, tree, program_circuits):
        self.tree = tree
        self.program_circuits = program_circuits

    def circuits_descending(self):
        cnot_density = []
        for circuit in self.program_circuits:
            density = len(
                list(circuit.findall_operations(lambda op: op.gate == cirq.CZ))
            ) / float(len(circuit.all_qubits()))
            cnot_density.append(density)
        """ computing indices regarding descending order of cnot densities """
        idxs = np.argsort(cnot_density)
        idxs_descending = np.flip(idxs)

        print(cnot_density)
        print(idxs_descending)
        """ reorder list of program_circuits """
        circuits_temp = []  #copy.deepcopy(self.program_circuits)
        #self.program_circuits.clear()
        for id in idxs_descending:
            circuits_temp.append(self.program_circuits[id])

        return circuits_temp

    def find_best_candidate(self, cands):
        print("cands:")
        print(cands)
        best_cand = list(self.tree.nodes())[0]  # to do

        return tuple(best_cand)

    def qubits_allocation(self, desc_prog_circuits):
        #self.circuits_descending()
        partition = []

        for cir in desc_prog_circuits:
            candidates = []
            leaves = [
                x for x in self.tree.nodes()
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
                        if not exist:
                            candidates.append(leaf)
                        break
                    else:
                        leaf = tuple(self.tree.predecessors(leaf))[0]

            if len(candidates) == 0:
                print("fail -- run programs seperately")
            #print(candidates)
            best_cand = self.find_best_candidate(candidates)
            partition.append(best_cand)
            """ remove nodes from tree & relabel remaining nodes"""
            successors = list(self.tree.successors(best_cand))
            self.tree.remove_nodes_from(successors)
            self.tree.remove_node(best_cand)
            label_mapping = {}
            for n in list(self.tree.nodes()):
                label_mapping[n] = tuple(x for x in n if x not in best_cand)

            nx.relabel_nodes(self.tree, label_mapping)

        return partition


class X_SWAP:

    def __init__(self, device_graph, desc_program_circuits, partitions):
        self.device_graph = device_graph
        self.desc_program_circuits = desc_program_circuits
        self.partitions = partitions

    def generate_2qGates_dags(self, twoq_gateType):
        cir_dags = []
        for c in self.desc_prog_circuits:
            """ remove single qubit gates before creating dag """
            singleq_gates = list(
                c.findall_operations(lambda op: op.gate != twoq_gateType))
            c.batch_remove(singleq_gates)
            """ create dag """
            cir_dags.append(cirq.CircuitDag.from_circuit(c))

        return cir_dags

    def generate_dags(self):
        cir_dags = []
        for c in self.desc_program_circuits:
            """ create dag """
            cir_dags.append(cirq.CircuitDag.from_circuit(c))

        return cir_dags

    def generate_2qGates_front_layers(self, cir_dags):
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

    def generate_front_layers(self, cir_dags, twoq_gate_type):
        """ check all dags are empty """
        counter = 0
        for dag in cir_dags:
            nodes = list(dag.ordered_nodes())
            if len(nodes)==0:
                counter = counter + 1
        if counter == len(cir_dags):
            return None

        """ set of nodes """
        flayers = [] 
        for dag in cir_dags:
            fl = []
            nodes = list(dag.ordered_nodes())
            # ops = list(dag.all_operations()) ??
            for i in range(len(nodes)):
                if len(list(dag.predecessors(nodes[i]))) == 0:
                    fl.append(nodes[i]) # .val  # fl.append(ops[i]) ??
                else:
                    counter = 0
                    for p in list(dag.predecessors(nodes[i])):
                        if p.val.gate != twoq_gate_type:
                            counter = counter + 1
                    if counter == len(list(dag.predecessors(nodes[i]))):
                        fl.append(nodes[i]) # .val  # fl.append(ops[i]) ??

            flayers.append(fl)

        return flayers

    def generate_second_layer(self, cir_dag):
        secondl = []
        ## to do

        return secondl

    def initial_mapping(self):
        """ return a list of dictionaries that shows mapping 
        between logical qubits (key) and physical qubits (value) """
        mappings = []
        for i in range(len(self.desc_program_circuits)):
            map = {}
            logical_qubits = list(self.desc_program_circuits[i].all_qubits())
            print(self.desc_program_circuits[i])
            for j in range(len(logical_qubits)):
                print("yes")
                print(len(logical_qubits))
                print(len(self.partitions[i]))
                print(self.partitions[i][j])
                print(logical_qubits[j])
                map[logical_qubits[j]] = self.partitions[i][j]
            mappings.append(map)
        return mappings
    
    def log_to_phy_edge (self, log_edge, mappings):
        phy0 = None
        phy1 = None
        for map in mappings:
            for log, phy in map.items():
                if log_edge[0] == log:
                    phy0 = phy
                elif log_edge[1] == log:
                    phy1 = phy

        return (phy0, phy1)

    def phy_to_log_edge (self, phy_edge, mappings):
        log0 = None
        log1 = None
        for map in mappings:
            for log, phy in map.items():
                if phy_edge[0] == phy:
                    log0 = log
                elif phy_edge[1] == phy:
                    log1 = log
        return (log0, log1)


    def obtain_swaps (self, node_gates, mappings):
        swaps = []
        for n in node_gates:
            g = n.val
            phy_qs = self.log_to_phy_edge(g.qubits, mappings)
            neighbors0 = list(self.device_graph.neighbors(phy_qs[0]))
            neighbors1 = list(self.device_graph.neighbors(phy_qs[1]))
            for ne in neighbors0:
                swaps.append(self.phy_to_log_edge((phy_qs[0], ne),mappings))
            for ne in neighbors1:
                swaps.append(self.phy_to_log_edge((phy_qs[1], ne),mappings))
        return swaps

    def update_mapping (self, mappings, swap):
        new_maps = mappings.copy()
        pidx0 = None
        pidx1 = None
        for i in range(len(mappings)):
            for l, p in mappings[i].items():
                if l == swap[0]:
                    pidx0 = i
                elif l == swap[1]:
                    pidx1 = i
        new_maps[pidx0][swap[0]] = mappings[pidx1][swap[1]]
        new_maps[pidx1][swap[1]] = mappings[pidx0][swap[0]]

        return new_maps

    def compute_H(self, flayers, maps):
        H_cost = 0
        for i in range(len(flayers)):
            if len(flayers[i]) == 0:
                continue
            for n in flayers[i]:
                phy_edge = self.log_to_phy_edge(n.val.qubits, maps)
                H_cost = H_cost + len(list(nx.all_shortest_paths(self.device_graph, phy_edge[0], phy_edge[1]))[0])
        return H_cost
    
    def compute_path_in_sameP(self, edge, pidx):
        paths = list(nx.all_simple_paths(self.device_graph, edge[0], edge[1], cutoff=len(self.partitions[pidx])))
        distance = inf
        for p in paths:
            if self.partitions[pidx] in p:
                if len(p) < distance:
                    distance = len(p)
        return distance

    def compute_gainCost(self, flayers, maps, swap):
        gain_cost = 0
        
        for i in range(len(flayers)):
            if len(flayers[i]) == 0:
                continue
            cost_i = 0
            for n in flayers[i]:
                phy_edge = self.log_to_phy_edge(n.val.qubits, maps)
                paths = nx.all_shortest_paths(self.device_graph, phy_edge[0], phy_edge[1])
                in_path = 0
                for p in paths:
                    if swap in p:
                        in_path = 1
                D_allp = len(list(nx.all_shortest_paths(self.device_graph, phy_edge[0], phy_edge[1]))[0])
                D_singlep = self.compute_path_in_sameP(phy_edge, i)
                cost_i = cost_i + (D_allp-D_singlep) * in_path
            gain_cost = gain_cost + float(1/len(flayers[i])) * cost_i
        return gain_cost

    def find_best_swap (self, swap_candidate_lists, mappings, flayers):
        min_cost = inf
        best_swap = None
        for swaps in swap_candidate_lists:
            for s in swaps:
                new_maps = self.update_mapping(mappings, s)
                H_cost = self.compute_H(flayers, new_maps)
                gain_cost = self.compute_gainCost(flayers, new_maps, s)
                cost = H_cost + gain_cost
                if(cost<min_score):
                    min_cost = cost
                    best_swap = s
        return best_swap
                




    def insert_SWAP_and_generate_schedule(self, twoq_gateType):
        schedule = cirq.Circuit()
        #twoq_dags = self.generate_2qGates_dags(twoq_gateType)

        initial_maps = self.initial_mapping()
        dags = self.generate_dags()
        
        mappings = self.initial_mapping()

        flayers = self.generate_front_layers(dags, twoq_gateType)
        while flayers != None:
            """ solve hardware-compliant gates 
            i specify program index """
            require_swap = 0
            for i in range(len(flayers)):
                if len(flayers[i]) == 0:
                        continue
                gate_nodes = flayers[i].copy()
                for n in gate_nodes:
                    if len(n.val.qubits) == 1:
                        schedule.append(n.val)
                        dags[i].remove_node(n)
                        # update front layer
                        flayers[i].remove(n)
                    else:
                        if self.log_to_phy_edge(n.val.qubits, mappings) in self.device_graph.edges:
                            schedule.append(n.val)
                            dags[i].remove_node(n)
                            # update front layer
                            flayers[i].remove(n)
                        else:
                            """ require SWAP """
                            require_swap = 1

            """ solve hardware-incompliant gates by inserting SWAPs """
            if(require_swap):
                swap_candidate_lists = []
                for i in range(len(flayers)):
                    if len(flayers[i]) == 0:
                        continue
                    critical_node_gates = flayers[i].copy() # to do ??
                    swap_candidates = self.obtain_swaps(critical_node_gates, mappings)
                    swap_candidate_lists.append(swap_candidates)
                
                """ find best SWAP """
                best_swap = self.find_best_swap(swap_candidate_lists, mappings, flayers)
                schedule.append(cirq.SWAP(best_swap[0], best_swap[1]))

                """ update mapping """
                mappings = self.update_mapping(mappings, best_swap)

            flayers = self.generate_front_layers(dags, twoq_gateType)

        return initial_maps, schedule


############################################################


def multi_prog_map(device_graph, single_er, two_er, prog_circuits):
    treeObj = Hierarchy_tree(device_graph, single_er, two_er)
    tree = treeObj.tree_construction()

    #print(len(tree.nodes))
    #print(tree.nodes)

    #print((list(tree.nodes)[1]))
    parObj = Qubits_partitioning(tree, prog_circuits)
    desc_cirs = parObj.circuits_descending()

    partitions = parObj.qubits_allocation(desc_cirs)
    partitions.reverse()
    print("partitions:")
    print(partitions)

    xswap = X_SWAP(device_graph, desc_cirs, partitions)
    xswap.insert_SWAP_and_generate_schedule(cirq.CZ)

    #parObj.reorder_program_circuits()


def prepare_couplingGraph_errorValues(device_graph):
    #single_er = load_calibrations()['single_qubit_p00_error'] # to do ??
    #two_er = load_calibrations()['two_qubit_sycamore_gate_xeb_cycle_purity_error'] # to do ??

    single_er = {
        (cirq.GridQubit(1, 0),): [0.028600441075128205],
        (cirq.GridQubit(0, 0),): [0.01138359559038841],
        (cirq.GridQubit(1, 1),): [0.05313138858345922],
        (cirq.GridQubit(0, 1),): [0.0005880214404983153],
        (cirq.GridQubit(1, 2),): [0.0018232495924263727],
        (cirq.GridQubit(0, 2),): [0.039571298178797366]
    }
    two_er = {
        (cirq.GridQubit(1, 0), cirq.GridQubit(0, 0)): [0.018600441075128205],
        (cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)): [0.01938359559038841],
        (cirq.GridQubit(1, 1), cirq.GridQubit(0, 1)): [0.01313138858345922],
        (cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)): [0.005880214404983153],
        (cirq.GridQubit(1, 1), cirq.GridQubit(1, 2)): [0.008232495924263727],
        (cirq.GridQubit(0, 2), cirq.GridQubit(1, 2)): [0.03571298178797366]
    }

    # coupling graph
    dgraph = nx.Graph()

    for q0, q1 in two_er:  # .items()
        dgraph.add_edge(q0, q1)

    # list of program circuits
    qubits = cirq.LineQubit.range(3)
    circuit1 = cirq.Circuit(cirq.X(qubits[0]), cirq.Y(qubits[1]),
                            cirq.CZ(qubits[0], qubits[1]),
                            cirq.CZ(qubits[1], qubits[2]),
                            cirq.measure(*qubits))
    qubits = cirq.LineQubit.range(2)
    circuit2 = cirq.Circuit(cirq.X(qubits[0]), cirq.Y(qubits[1]),
                            cirq.CZ(qubits[0], qubits[1]))
    program_circuits = []
    program_circuits.append(circuit2)
    program_circuits.append(circuit1)

    print(circuit2)
    print(circuit1)

    multi_prog_map(dgraph, single_er, two_er, program_circuits)


if __name__ == "__main__":
    #print( load_calibrations()['single_qubit_p00_error'] )
    device_graph1 = ccr.get_grid_device_graph(3, 2)
    device_graph = cirq.google.Sycamore
    prepare_couplingGraph_errorValues(device_graph)

    #multi_prog_map(device_graph)
