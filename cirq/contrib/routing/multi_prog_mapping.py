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

    def __init__(self, tree, program_circuits, single_er, two_er, twoQ_gate_type):
        self.tree = tree
        self.program_circuits = program_circuits
        self.single_er = single_er
        self.two_er = two_er
        self.twoQ_gate_type = twoQ_gate_type

    def circuits_descending(self):
        cnot_density = []
        for circuit in self.program_circuits:
            density = len(
                list(circuit.findall_operations(lambda op: op.gate == self.twoQ_gate_type))
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

    def compute_EPST(self, partition, cir):
        print(partition)
        twoQ_gs = len(list(cir.findall_operations(lambda op: op.gate == self.twoQ_gate_type)))
        oneQ_gs = len(list(cir.all_operations())) - twoQ_gs
        qubits = len(cir.all_qubits())

        err_2 = 0
        count_2 = 0
        err_1 = 0
        for i in range(len(partition)-1):
            for j in range(i+1,len(partition)):
                if (partition[i], partition[j]) in self.two_er: 
                    err_2 = err_2 + self.two_er[(partition[i], partition[j])][0]
                    count_2 = count_2 + 1
                elif (partition[j], partition[i]) in self.two_er:
                    err_2 = err_2 + self.two_er[(partition[j], partition[i])][0]
                    count_2 = count_2 + 1
        for i in range(len(partition)):
            err_1 = err_1 + self.single_er[(partition[i],)][0]

        avgF_2 = 1 - float(err_2/count_2)
        avgF_1 = 1 - float(err_1/len(partition))
        # to do : readout error

        return pow(avgF_2,twoQ_gs) * pow(avgF_1, oneQ_gs) * pow(avgF_1, qubits)



    def find_best_candidate(self, cands, cir):
        """ 
        find best candidate based on average fidelity
        """
        print("cands:")
        print(cands)
        best_cand = cands[0]
        max_f = -inf
        # if len(cands)==1:
        #     return tuple(best_cand)

        for cand in cands:
            epst = self.compute_EPST(cand, cir)
            print("epst:")
            print(epst)
            if epst > max_f:
                max_f = epst
                best_cand = cand

        return best_cand

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
                            elif set(c).issubset(leaf): 
                                """ keep independent candidates"""
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
            best_cand = self.find_best_candidate(candidates, cir)
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

    def __init__(self, device_graph, desc_program_circuits, partitions, twoQ_gate_type, l_to_ph = {}, ph_to_l = {}):
        self.device_graph = device_graph
        self.desc_program_circuits = desc_program_circuits
        self.partitions = partitions
        self.twoQ_gate_type = twoQ_gate_type
        self.l_to_ph = l_to_ph
        self.ph_to_l = ph_to_l

    def generate_2qGates_dags(self):
        cir_dags = []
        for c in self.desc_prog_circuits:
            """ remove single qubit gates before creating dag """
            singleq_gates = list(
                c.findall_operations(lambda op: op.gate != self.twoQ_gate_type))
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

    def generate_front_layers(self, cir_dags):
        """ check all dags are empty """
        counter = 0
        for dag in cir_dags:
            nodes = list(dag.ordered_nodes())
            if len(nodes) == 0:
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

    def initial_mapping(self):
        """ initialize 2 dictionaries l_to_ph and ph_to_l in order """
        total_l_qubits = 0
        total_ph_qubits = len(self.device_graph.nodes)
        for i in range(len(self.desc_program_circuits)):
            logical_qubits = list(self.desc_program_circuits[i].all_qubits())
            total_l_qubits = total_l_qubits + len(logical_qubits)
            for j in range(len(logical_qubits)):
                self.l_to_ph[(logical_qubits[j],i)] = self.partitions[i][j]
                self.ph_to_l[self.partitions[i][j]] = (logical_qubits[j], i)

        """ manage unused qubits and map them to a unreal program with id = -1 """
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

    def log_to_phy_edge(self, log_edge, pid):
        phy0 = self.l_to_ph[(log_edge[0], pid)]
        phy1 = self.l_to_ph[(log_edge[1], pid)]
        return (phy0, phy1)

    def phy_to_log_edge(self, phy_edge):
        log_pid0 = self.ph_to_l[phy_edge[0]]
        log_pid1 = self.ph_to_l[phy_edge[1]]
        return [log_pid0, log_pid1]

    def obtain_swaps(self, node_gates, pid):
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

    def update_mapping(self, swap):
        new_ph_l = self.ph_to_l.copy()
        new_l_ph = self.l_to_ph.copy()

        ph0 = self.l_to_ph[swap[0]]
        ph1 = self.l_to_ph[swap[1]]
        new_l_ph[swap[0]] = ph1
        new_l_ph[swap[1]] = ph0
        new_ph_l[ph0] = self.ph_to_l[ph1]
        new_ph_l[ph1] = self.ph_to_l[ph0]

        return new_l_ph, new_ph_l

    def compute_H(self, flayers, new_l_ph, new_ph_l):
        H_cost = 0
        for i in range(len(flayers)):
            if len(flayers[i]) == 0:
                continue
            for n in flayers[i]:
                lq = n.val.qubits
                phy_edge = (new_l_ph[(lq[0], i)], new_l_ph[(lq[1], i)]) #self.log_to_phy_edge(n.val.qubits, maps)
                H_cost = H_cost + len(
                    list(
                        nx.all_shortest_paths(self.device_graph, phy_edge[0],
                                              phy_edge[1]))[0])
        return H_cost -1

    def compute_path_in_sameP(self, edge, pidx):
        paths = list(
            nx.all_simple_paths(self.device_graph,
                                edge[0],
                                edge[1],
                                cutoff=len(self.partitions[pidx])))
        distance = inf
        for p in paths:
            if self.partitions[pidx] in p:
                if len(p) < distance:
                    distance = len(p)
        return distance

    def compute_gainCost(self, flayers, new_l_ph, new_ph_l, swap):
        gain_cost = 0

        for i in range(len(flayers)):
            if len(flayers[i]) == 0:
                continue
            cost_i = 0
            for n in flayers[i]:
                lq = n.val.qubits
                new_phy_edge = (new_l_ph[(lq[0], i)], new_l_ph[(lq[1], i)])
                new_paths = nx.all_shortest_paths(self.device_graph, new_phy_edge[0],
                                              new_phy_edge[1])

                phy_edge = (self.l_to_ph[(lq[0], i)], self.l_to_ph[(lq[1], i)])
                paths = nx.all_shortest_paths(self.device_graph, phy_edge[0],
                                              phy_edge[1])
                path_len = len(list(paths)[0])
                new_path_len = len(list(new_paths)[0])

                print("path")
                print(list(paths))
                in_path = 0
                
                if new_path_len<path_len:
                    in_path = 1
                D_allp = new_path_len
                D_singlep = self.compute_path_in_sameP(phy_edge, i)
                cost_i = cost_i + (D_allp - D_singlep) * in_path
                print("in path")
                print(in_path)
                print(cost_i)
                print("next i")
            gain_cost = gain_cost + float(1 / len(flayers[i])) * cost_i
        return gain_cost

    def find_best_swap(self, swap_candidate_lists, flayers):
        min_cost = inf
        best_swap = None
        for swaps in swap_candidate_lists:
            for s in swaps:
                new_l_ph, new_ph_l = self.update_mapping(s)
                H_cost = self.compute_H(flayers, new_l_ph, new_ph_l)
                gain_cost = self.compute_gainCost(flayers, new_l_ph, new_ph_l, s)
                cost = H_cost + gain_cost
                print("cost")
                print(H_cost)
                print(gain_cost)
                print(cost)
                if (cost < min_cost):
                    min_cost = cost
                    best_swap = s
        return best_swap

    def insert_SWAP_and_generate_schedule(self):
        schedule = cirq.Circuit()
        dags = self.generate_dags()
        self.initial_mapping()

        sig = 0
        flayers = self.generate_front_layers(dags)
        while flayers != None:
            if sig == 3:
                break
            sig = sig +1
            """ solve hardware-compliant gates 
            i specify program index """
            require_swap = 0
            print("flayers")
            print(flayers)
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
                        if self.log_to_phy_edge(
                                n.val.qubits, i) in self.device_graph.edges:
                            schedule.append(n.val)
                            dags[i].remove_node(n)
                            # update front layer
                            flayers[i].remove(n)
                        else:
                            """ require SWAP """
                            require_swap = 1
            """ solve hardware-incompliant gates by inserting SWAPs """
            if (require_swap):
                swap_candidate_lists = []
                for i in range(len(flayers)):
                    if len(flayers[i]) == 0:
                        swap_candidate_lists.append([])
                        continue
                    critical_node_gates = flayers[i].copy()  # to do ??
                    swap_candidates = self.obtain_swaps(critical_node_gates, i)
                    swap_candidate_lists.append(swap_candidates)
                """ find best SWAP """
                best_swap = self.find_best_swap(swap_candidate_lists,
                                                flayers)

                print("best swap")
                print(swap_candidate_lists)
                print(best_swap)
                #schedule.append(cirq.SWAP(best_swap[0], best_swap[1]))
                """ update mapping """
                mappings = self.update_mapping(best_swap)

            flayers = self.generate_front_layers(dags)

        return  schedule


############################################################


def multi_prog_map(device_graph, single_er, two_er, prog_circuits):
    twoQ_gate_type = cirq.CZ
    treeObj = Hierarchy_tree(device_graph, single_er, two_er)
    tree = treeObj.tree_construction()
    print("tree")

    #print(len(tree.nodes))
    print(tree.nodes)

    #print((list(tree.nodes)[1]))
    parObj = Qubits_partitioning(tree, prog_circuits, single_er, two_er, twoQ_gate_type)
    desc_cirs = parObj.circuits_descending()

    partitions = parObj.qubits_allocation(desc_cirs)
    
    print("partitions:")
    print(partitions)

    xswap = X_SWAP(device_graph, desc_cirs, partitions, twoQ_gate_type)
    xswap.insert_SWAP_and_generate_schedule()

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
