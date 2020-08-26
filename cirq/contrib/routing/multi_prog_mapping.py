import networkx as nx

import cirq
import cirq.contrib.routing as ccr


class Hierarchy_tree:

  def __init__(self, device_graph, calibration_data, tree ):
    self.device_graph = device_graph
    self.tree = tree

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
    communities = [[i] for i in list(self.device_graph.nodes)]
    while len(communities)>1:
      idx1, idx2 = self.compute_new_node(communities)
      new_node = []
      new_node.append(communities[idx1])
      new_node.append(communities[idx2])
      new_node_idx = communities[idx1] + communities[idx2]
      self.tree[tuple(new_node_idx)] = new_node
      communities.pop(idx1)
      communities.pop(idx2)
      communities.append(new_node_idx)
    

def multi_prog_map(device_graph):
  calibration_data= []
  tree = {}
  obj = Hierarchy_tree(device_graph, calibration_data, tree)
  obj.tree_construction()

if __name__ == "__main__":
  device_graph = ccr.get_grid_device_graph(3, 2)
  multi_prog_map(device_graph)