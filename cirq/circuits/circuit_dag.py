# Copyright 2018 The ops Developers
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

from typing import Any, Callable, Generic, Iterator, Type, TypeVar

import networkx

from cirq import ops, devices
from cirq.circuits import circuit


T = TypeVar('T')

class Unique(Generic[T]):
    """A wrapper for a value that doesn't compare equal to other instances.

    For example: 5 == 5 but Unique(5) != Unique(5).

    Unique is used by CircuitDag to wrap operations because nodes in a graph
    are considered the same node if they compare equal to each other.  X(q0)
    in one moment of a Circuit and X(q0) in another moment of the Circuit are
    wrapped by Unique(X(q0)) so they are distinct nodes in the graph.
    """
    def __init__(self, val: T) -> None:
        self.val = val

    def __repr__(self):
        return 'Unique({}, {!r})'.format(id(self), self.val)


TSelf = TypeVar('TSelf', bound='CircuitDag')


def _default_can_reorder(op1: ops.Operation, op2: ops.Operation) -> bool:
    """Returns true only if the operations have any qubits in common."""
    return not set(op1.qubits) & set(op2.qubits)


class CircuitDag(networkx.DiGraph):
    """A representation of a Circuit as a directed acyclic graph.

    Nodes of the graph are instances of Unique containing each operation of a
    circuit.

    Edges of the graph are tuples of nodes.  Each edge specifies a required
    application order between two operations.  The first must be applied before
    the second.
    """

    default_can_reorder = staticmethod(_default_can_reorder)

    def __init__(self,
                 can_reorder: Callable[[ops.Operation, ops.Operation], bool] =
                    _default_can_reorder,
                 incoming_graph_data: Any = None,
                 device: devices.Device = devices.UnconstrainedDevice
                 ) -> None:
        """Initializes a CircuitDag.

        Args:
            can_reorder: A predicate that determines if two operations may be
                reordered.  Graph edges are created for pairs of operations
                where this returns False.

                The default predicate allows reordering only when the operations
                don't share common qubits.
            incoming_graph_data: Data in initialize the graph.  This can be any
                value supported by networkx.DiGraph() e.g. an edge list or
                another graph.
            device: Hardware that the circuit should be able to run on.
        """
        super().__init__(incoming_graph_data)
        self.can_reorder = can_reorder
        self.device = device

    @staticmethod
    def make_node(op: ops.Operation) -> Unique:
        return Unique(op)

    @classmethod
    def from_circuit(cls: Type[TSelf],
                     circuit: circuit.Circuit,
                     can_reorder: Callable[[ops.Operation, ops.Operation],
                                           bool] = _default_can_reorder
                     ) -> TSelf:
        return cls.from_ops(circuit.all_operations(),
                            can_reorder=can_reorder,
                            device=circuit.device)

    @classmethod
    def from_ops(cls: Type[TSelf],
                 *operations: ops.OP_TREE,
                 can_reorder: Callable[[ops.Operation, ops.Operation], bool] =
                    _default_can_reorder,
                 device: devices.Device = devices.UnconstrainedDevice
                 ) -> TSelf:
        dag = cls(can_reorder=can_reorder, device=device)
        for op in ops.flatten_op_tree(operations):
            dag.append(op)
        return dag

    def append(self, op: ops.Operation) -> None:
        new_node = self.make_node(op)
        self.add_edges_from([(node, new_node)
                             for node in self.nodes
                             if not self.can_reorder(node.val, new_node.val)])
        self.add_node(new_node)

    def all_operations(self) -> Iterator[ops.Operation]:
        if not self.nodes:
            return
        g = self.copy()

        def get_root_node(some_node: Unique[ops.Operation]
                          ) -> Unique[ops.Operation]:
            pred = g.pred
            while pred[some_node]:
                some_node = next(iter(pred[some_node]))
            return some_node

        def get_first_node() -> Unique[ops.Operation]:
            return get_root_node(next(iter(g.nodes)))

        def get_next_node(succ: networkx.classes.coreviews.AtlasView
                          ) -> Unique[ops.Operation]:
            if succ:
                return get_root_node(next(iter(succ)))
            else:
                return get_first_node()

        node = get_first_node()
        while True:
            yield node.val
            succ = g.succ[node]
            g.remove_node(node)

            if not g.nodes:
                return

            node = get_next_node(succ)

    def to_circuit(self) -> circuit.Circuit:
        return circuit.Circuit.from_ops(
                    self.all_operations(),
                    strategy=circuit.InsertStrategy.EARLIEST,
                    device=self.device)
