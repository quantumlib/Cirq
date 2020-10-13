""" Author: Fereshte Mozafari """

import networkx as nx
import numpy as np
import math
import copy

import cirq
import cirq.contrib.routing as ccr
from cirq import circuits, ops, value
from util import load_calibrations


class Hierarchy_tree:

  def __init__(self, device_graph, single_er, two_er):
    self.device_graph = device_graph
    self.single_er = single_er
    self.two_er = two_er

  def compute_F(self, com1, com2, Q1, Q2, fidelity):
    edges_count = len( list(self.device_graph.edges) )
    omega = 1.0
    inside_edges = 0
    outside_edges = 0
    total_edges = 0

    """ compute inside edges """
    for c1 in com1:
      for c2 in com2:
        if(self.device_graph.has_edge(c1,c2)):
          inside_edges += 1
    for i in range( len(com1)-1 ):
      for j in range( i+1, len(com1), 1 ):
        if(self.device_graph.has_edge(com1[i],com1[j])):
          inside_edges += 1
    for i in range( len(com2)-1 ):
      for j in range( i+1, len(com2), 1 ):
        if(self.device_graph.has_edge(com2[i],com2[j])):
          inside_edges += 1

    """ compute total edges """
    for c1 in com1:
      total_edges += len( list (c1.neighbors()) )
    for c2 in com2:
      total_edges += len ( list (c2.neighbors() ) )

    """ compute outside edges """
    outside_edges = total_edges - 2 * inside_edges

    Qmerged = float(inside_edges/edges_count) - pow( float(outside_edges/edges_count), 2 )
    deltaQ = Qmerged - Q1 - Q2
    F_value = deltaQ + omega * fidelity

    return F_value, Qmerged

  def find_edge_among_coms( self, com1, com2 ):
    fidelity = 0.0
    for c1 in com1:
      for c2 in com2: 
        if (c1, c2) in self.two_er:
          fidelity_two = 1 - self.two_er[(c1, c2)][0]
          fidelity_single = 0.5 * (1-self.single_er[(c1, )][0] + 1-self.single_er[(c2, )][0])
          fidelity += (fidelity_two * fidelity_single)
        elif (c2, c1) in self.two_er:
          fidelity_two = 1 - self.two_er[(c2, c1)][0]
          fidelity_single = 0.5 * (1-self.single_er[(c1, )][0] + 1-self.single_er[(c2, )][0])
          fidelity += (fidelity_two * fidelity_single)

    return fidelity

  def compute_new_node(self, communities, Qvalues):
    idx1 = 0
    idx2 = 1
    Qmerged = 0.0
    Fmax = -math.inf

    for i in range( len(communities)-1 ):
      for j in range( i+1, len(communities), 1 ):
        fidelity = self.find_edge_among_coms( communities[i], communities[j] )
        if(fidelity != 0.0):
          F , qmerged = self.compute_F( communities[i], communities[j], Qvalues[i], Qvalues[j], fidelity )
          if(F > Fmax):
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

    while len(communities)>1:
      idx1, idx2, Qmerged = self.compute_new_node(communities, Qvalues)
      # new_node = []
      # new_node.append(communities[idx1])
      # new_node.append(communities[idx2])
      new_node_list = communities[idx1] + communities[idx2]
      print(new_node_list)
      #tree[tuple(new_node_idx)] = new_node
      new_node = tuple(new_node_list)
      #tree.add_node(new_node) #(label , data = new_node_idx)
      tree.add_edge( new_node, tuple(communities[idx1]) )
      tree.add_edge( new_node, tuple(communities[idx2]) )
      
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
      density = len(list(circuit.findall_operations(lambda op: op.gate == cirq.CZ))) / float(len(circuit.all_qubits()))
      cnot_density.append(density)
    """ computing indices regarding descending order of cnot densities """
    idxs = np.argsort(cnot_density)
    idxs_descending = np.flip(idxs)
    
    """ reorder list of program_circuits """
    circuits_temp = copy.deepcopy(self.program_circuits)
    self.program_circuits.clear()
    for id in idxs_descending:
      self.program_circuits.append(circuits_temp[id])

  def find_best_candidate(self, cands):
    best_cand = list(self.tree.nodes())[0] # to do

    return tuple(best_cand)

  def qubits_allocation(self):
    self.circuits_descending()
    partition = []

    for cir in self.program_circuits:
      candidates = []
      leaves = [x for x in self.tree.nodes() if self.tree.out_degree(x)==0 and self.tree.in_degree(x)==1]
      for leaf in leaves:
        while leaf is not None:
          if len(cir.all_qubits()) <= len(list(leaf)):
            exist = 0
            for c in candidates:
              if c == leaf:
                exist =1
                break
            if not exist:
              candidates.append(leaf)
            break
          else:
            leaf = tuple(self.tree.predecessors(leaf))[0]

      if len(candidates) == 0:
        print("fail -- run programs seperately")
      print(candidates)   
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
  def __init__(self, device_graph, program_circuits, partitions):
    self.device_graph = device_graph
    self.program_circuits = program_circuits
    self.partitions = partitions






############################################################    

def multi_prog_map( device_graph, single_er, two_er, prog_circuits ):
  treeObj = Hierarchy_tree( device_graph, single_er, two_er )
  tree = treeObj.tree_construction()

  print(len(tree.nodes))
  print(tree.nodes)

  print((list(tree.nodes)[1]))
  parObj = Qubits_partitioning(tree, prog_circuits)
  parObj.qubits_allocation()
  #parObj.reorder_program_circuits()

def prepare_couplingGraph_errorValues( device_graph ):
  #single_er = load_calibrations()['single_qubit_p00_error'] # to do ??
  #two_er = load_calibrations()['two_qubit_sycamore_gate_xeb_cycle_purity_error'] # to do ??

  single_er = {
    (cirq.GridQubit(1,0), ): [0.028600441075128205], 
    (cirq.GridQubit(0,0), ): [0.01138359559038841], 
    (cirq.GridQubit(1,1), ): [0.05313138858345922], 
    (cirq.GridQubit(0,1), ): [0.0005880214404983153], 
    (cirq.GridQubit(1,2), ): [0.0018232495924263727], 
    (cirq.GridQubit(0,2), ): [0.039571298178797366]
  }
  two_er = {
    (cirq.GridQubit(1,0), cirq.GridQubit(0,0)): [0.018600441075128205], 
    (cirq.GridQubit(0,0), cirq.GridQubit(0,1)): [0.01938359559038841], 
    (cirq.GridQubit(1,1), cirq.GridQubit(0,1)): [0.01313138858345922], 
    (cirq.GridQubit(0,1), cirq.GridQubit(0,2)): [0.005880214404983153], 
    (cirq.GridQubit(1,1), cirq.GridQubit(1,2)): [0.008232495924263727], 
    (cirq.GridQubit(0,2), cirq.GridQubit(1,2)): [0.03571298178797366]
  }
 
  # coupling graph
  dgraph = nx.Graph()

  for q0, q1 in two_er: # .items() 
    dgraph.add_edge(q0, q1)

  # list of program circuits
  qubits1 = cirq.LineQubit.range(3)
  circuit1 = cirq.Circuit(cirq.X(qubits1[0]), cirq.Y(qubits1[1]), cirq.CZ(qubits1[0], qubits1[1]), cirq.CZ(qubits1[1], qubits1[2]), cirq.measure(*qubits1))
  qubits2 = cirq.LineQubit.range(3)
  circuit2 = cirq.Circuit(cirq.X(qubits2[0]), cirq.Y(qubits2[1]), cirq.CZ(qubits2[0], qubits2[1]) )
  program_circuits = []
  program_circuits.append(circuit2)
  program_circuits.append(circuit1)

  multi_prog_map(dgraph, single_er, two_er, program_circuits)
  





if __name__ == "__main__":
  #print( load_calibrations()['single_qubit_p00_error'] )
  device_graph1 = ccr.get_grid_device_graph(3, 2)
  device_graph = cirq.google.Sycamore
  prepare_couplingGraph_errorValues(device_graph)

  #multi_prog_map(device_graph)