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
  def __init__(self, tree, programs_circuit):
    self.tree = tree
    self.programs_circuit = programs_circuit

  def cnot_density(self, program_circuit):
    # #cnots/#qubits ??
    """
    To Do
    """
    d = 0
    return d

  def reorder_program_circuits(self):
    cnot_d = []
    for p in self.programs_circuit:
      cnot_d.append(self.cnot_density(p))
    idxs = sorted(range(len(cnot_d)), key=lambda k: cnot_d[k])
    temp_p_circuit = self.programs_circuit.copy()
    for i in range(len(idxs)):
      self.programs_circuit[i] = temp_p_circuit[idxs[i]]


  def find_candidates(self ):
    return


    

def multi_prog_map(device_graph):
  calibration_data= []
  treeObj = Hierarchy_tree(device_graph, calibration_data)
  tree = treeObj.tree_construction()
  parObj = Qubit_partitioning(tree, [circuits.Circuit])
  parObj.reorder_program_circuits()

if __name__ == "__main__":
  device_graph = ccr.get_grid_device_graph(3, 2)
  multi_prog_map(device_graph)