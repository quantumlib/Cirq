## Author: Fereshte Mozafari

import networkx as nx

import cirq
import cirq.contrib.routing as ccr
from cirq import circuits, ops, value


class Hierarchy_tree:

  def __init__(self, device_graph, calibration_data):
    self.device_graph = device_graph

  def compute_F(self):
    """
    To Do
    """
    return

  def compute_new_node(self, communities):
    """
    To Do
    """
    id1 = 0
    id2 = 1
    # sort idx in decending order 1 , 0
    if id1<id2:
      idx1 = id2
      idx2 = id1
    else:
      idx1 = id1
      idx2 = id2

    return idx1, idx2


  def tree_construction(self):
    tree = {}
    communities = [[i] for i in list(self.device_graph.nodes)]
    while len(communities)>1:
      idx1, idx2 = self.compute_new_node(communities)
      new_node = []
      new_node.append(communities[idx1])
      new_node.append(communities[idx2])
      new_node_idx = communities[idx1] + communities[idx2]
      tree[tuple(new_node_idx)] = new_node
      communities.pop(idx1)
      communities.pop(idx2)
      communities.append(new_node_idx)
    print("yes")
    return tree

class Qubit_partitioning:
  def __init__(self, tree, programs_circuits):
    self.tree = tree
    self.programs_circuits = programs_circuits

  def cnot_density(self, pcircuit):
    # #cnots/#qubits ??
    """
    To Do
    """
    d = 0
    return d

  def reorder_program_circuits(self):
    cnot_d = []
    for p in self.programs_circuits:
      cnot_d.append(self.cnot_density(p))
    idxs = sorted(range(len(cnot_d)), key=lambda k: cnot_d[k])
    temp_p_circuit = self.programs_circuits.copy()
    for i in range(len(idxs)):
      self.programs_circuits[i] = temp_p_circuit[idxs[i]]


  def find_best_candidate(self, candidates):
    """
    To Do 
    """
    cand = []
    return cand

  def find_partitions(self ):
    partition = []
    for pcrct in self.programs_circuits:
      candidates = []
      nodes_count = 1#??
      for key, val in self.tree:
        if len(list(key)) == nodes_count:
          candidates.append(list(key))

        else:
          a=0
          #??
      if not candidates:
        print("fail")
      else:
        cand = self.find_best_candidate(candidates)
        partition.append(cand)
        self.tree.pop(tuple(cand))
        # to complete

    return partition

class X_SWAP:
  def __init__(self, device_graph, programs_circuits)






############################################################    

def multi_prog_map(device_graph):
  calibration_data= []
  treeObj = Hierarchy_tree(device_graph, calibration_data)
  tree = treeObj.tree_construction()
  parObj = Qubit_partitioning(tree, [circuits.Circuit])
  parObj.reorder_program_circuits()

if __name__ == "__main__":
  device_graph = ccr.get_grid_device_graph(3, 2)
  multi_prog_map(device_graph)