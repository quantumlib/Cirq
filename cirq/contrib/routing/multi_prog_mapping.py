## Author: Fereshte Mozafari

import networkx as nx
import numpy as np
import math

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

    # compute inside edges
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

    # compute total edges 
    for c1 in com1:
      total_edges += len( list (c1.neighbors()) )
    for c2 in com2:
      total_edges += len ( list (c2.neighbors() ) )

    # compute outside edges
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
    tree = nx.Graph()
    label = 0
    communities = []
    for n in list(self.device_graph.nodes):
      tree.add_node((n))   #(label, data = [n])
      label = label + 1

    communities = [[i] for i in list(self.device_graph.nodes)]
    Qvalues = [0.0] * len(communities)

    while len(communities)>1:
      idx1, idx2, Qmerged = self.compute_new_node(communities, Qvalues)
      # new_node = []
      # new_node.append(communities[idx1])
      # new_node.append(communities[idx2])
      new_node_list = communities[idx1] + communities[idx2]
      #tree[tuple(new_node_idx)] = new_node
      new_node = tuple(new_node_list)
      tree.add_node(new_node) #(label , data = new_node_idx)
      tree.add_edge(tuple(communities[idx1]), new_node)
      tree.add_edge(tuple(communities[idx2]), new_node)
      label = label + 1

      communities.pop(idx1)
      communities.pop(idx2)
      Qvalues.pop(idx1)
      Qvalues.pop(idx2)
      communities.append(new_node_list)
      Qvalues.append(Qmerged)
  
    return tree


############################################################    

def multi_prog_map( device_graph, single_er, two_er ):
  treeObj = Hierarchy_tree( device_graph, single_er, two_er )
  tree = treeObj.tree_construction()
  #parObj = Qubit_partitioning(tree, [circuits.Circuit])
  #parObj.reorder_program_circuits()

def prepare_couplingGraph_errorValues( device_graph ):
  single_er = load_calibrations()['single_qubit_p00_error'] # to do ??
  two_er = load_calibrations()['two_qubit_sycamore_gate_xeb_cycle_purity_error'] # to do ??
  # print(two_er)

  # qubits = list(device_graph.qubits)
  # qubits_count = len(qubits)
  # cgMatrix = np.zeros( (qubits_count, qubits_count) )

  dgraph = nx.Graph()

  for q0, q1 in two_er: # .items() 
    dgraph.add_edge(q0, q1)

  multi_prog_map(dgraph, single_er, two_er)
  





if __name__ == "__main__":
  #print( load_calibrations()['single_qubit_p00_error'] )
  device_graph1 = ccr.get_grid_device_graph(3, 2)
  device_graph = cirq.google.Sycamore
  prepare_couplingGraph_errorValues(device_graph)

  #multi_prog_map(device_graph)