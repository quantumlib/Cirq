# Copyright 2018 The Cirq Developers
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

from typing import Any, Callable, Dict, Generic, Iterator, TypeVar, cast, TYPE_CHECKING

import functools
import networkx

from cirq import ops, devices
from cirq.circuits import circuit

if TYPE_CHECKING:
    import cirq

T = TypeVar('T')


@functools.total_ordering
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

    def __repr__(self) -> str:
        return f'cirq.Unique({id(self)}, {self.val!r})'

    def __lt__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return id(self) < id(other)


def _disjoint_qubits(op1: 'cirq.Operation', op2: 'cirq.Operation') -> bool:
    """Returns true only if the operations have qubits in common."""
    return not set(op1.qubits) & set(op2.qubits)


class CircuitDag(networkx.DiGraph):
    """A representation of a Circuit as a directed acyclic graph.

    Nodes of the graph are instances of Unique containing each operation of a
    circuit.

    Edges of the graph are tuples of nodes.  Each edge specifies a required
    application order between two operations.  The first must be applied before
    the second.

    The graph is maximalist (transitive completion).
    """

    disjoint_qubits = staticmethod(_disjoint_qubits)

    def __init__(
        self,
        can_reorder: Callable[['cirq.Operation', 'cirq.Operation'], bool] = _disjoint_qubits,
        incoming_graph_data: Any = None,
        device: devices.Device = devices.UNCONSTRAINED_DEVICE,
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
    def make_node(op: 'cirq.Operation') -> Unique:
        return Unique(op)

    @staticmethod
    def from_circuit(
        circuit: circuit.Circuit,
        can_reorder: Callable[['cirq.Operation', 'cirq.Operation'], bool] = _disjoint_qubits,
    ) -> 'CircuitDag':
        return CircuitDag.from_ops(
            circuit.all_operations(), can_reorder=can_reorder, device=circuit.device
        )

    @staticmethod
    def from_ops(
        *operations: 'cirq.OP_TREE',
        can_reorder: Callable[['cirq.Operation', 'cirq.Operation'], bool] = _disjoint_qubits,
        device: devices.Device = devices.UNCONSTRAINED_DEVICE,
    ) -> 'CircuitDag':
        dag = CircuitDag(can_reorder=can_reorder, device=device)
        for op in ops.flatten_op_tree(operations):
            dag.append(cast(ops.Operation, op))
        return dag

    def append(self, op: 'cirq.Operation') -> None:
        new_node = self.make_node(op)
        for node in list(self.nodes()):
            if not self.can_reorder(node.val, op):
                self.add_edge(node, new_node)
                for pred in self.pred[node]:
                    self.add_edge(pred, new_node)
        self.add_node(new_node)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        g1 = self.copy()
        g2 = other.copy()
        for node, attr in g1.nodes(data=True):
            attr['val'] = node.val
        for node, attr in g2.nodes(data=True):
            attr['val'] = node.val

        def node_match(attr1: Dict[Any, Any], attr2: Dict[Any, Any]) -> bool:
            return attr1['val'] == attr2['val']

        return networkx.is_isomorphic(g1, g2, node_match=node_match)

    def __ne__(self, other):
        return not self == other

    __hash__ = None  # type: ignore

    def ordered_nodes(self) -> Iterator[Unique[ops.Operation]]:
        if not self.nodes():
            return
        g = self.copy()

        def get_root_node(some_node: Unique[ops.Operation]) -> Unique[ops.Operation]:
            pred = g.pred
            while pred[some_node]:
                some_node = next(iter(pred[some_node]))
            return some_node

        def get_first_node() -> Unique[ops.Operation]:
            return get_root_node(next(iter(g.nodes())))

        def get_next_node(succ: networkx.classes.coreviews.AtlasView) -> Unique[ops.Operation]:
            if succ:
                return get_root_node(next(iter(succ)))

            return get_first_node()

        node = get_first_node()
        while True:
            yield node
            succ = g.succ[node]
            g.remove_node(node)

            if not g.nodes():
                return

            node = get_next_node(succ)

    def all_operations(self) -> Iterator[ops.Operation]:
        return (node.val for node in self.ordered_nodes())

    def all_qubits(self):
        return frozenset(q for node in self.nodes for q in node.val.qubits)

    def to_circuit(self) -> circuit.Circuit:
        return circuit.Circuit(
            self.all_operations(), strategy=circuit.InsertStrategy.EARLIEST, device=self.device
        )

    def findall_nodes_until_blocked(
        self, is_blocker: Callable[[ops.Operation], bool]
    ) -> Iterator[Unique[ops.Operation]]:
        """Finds all nodes before blocking ones.

        Args:
            is_blocker: The predicate that indicates whether or not an
            operation is blocking.
        """
        remaining_dag = self.copy()

        for node in self.ordered_nodes():
            if node not in remaining_dag:
                continue
            if is_blocker(node.val):
                successors = list(remaining_dag.succ[node])
                remaining_dag.remove_nodes_from(successors)
                remaining_dag.remove_node(node)
                continue
            yield node
