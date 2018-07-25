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

from typing import Any, Generic, Iterator, Type, TypeVar

import networkx

from cirq import ops, devices, abc
from cirq.circuits import circuit


T = TypeVar('T')

class Unique(Generic[T]):
    def __init__(self, val: T) -> None:
        self.val = val

    def __repr__(self):
        return 'Unique({}, {!r})'.format(id(self), self.val)


TSelf = TypeVar('TSelf', bound='CircuitDagAbc')

class CircuitDagAbc(networkx.DiGraph, metaclass=abc.ABCMeta):
    """A representation of a Circuit as a directed acyclic graph.  Subclass
    CircuitDagAbc and implement can_reorder()."""

    def __init__(self,
                 incoming_graph_data: Any = None,
                 device: devices.Device = devices.UnconstrainedDevice
                 ) -> None:
        super().__init__(incoming_graph_data)
        self.device = device

    @abc.abstractmethod
    def can_reorder(self,
                    op1: ops.Operation,
                    op2: ops.Operation) -> bool:
        """Returns true if the order that op1 and op2 are applied in a circuit
        does not matter."""

    @staticmethod
    def make_node(op: ops.Operation) -> Unique:
        return Unique(op)

    @classmethod
    def from_circuit(cls: Type[TSelf], circuit: circuit.Circuit) -> TSelf:
        return cls.from_ops(circuit.all_operations(), device=circuit.device)

    @classmethod
    def from_ops(cls: Type[TSelf],
                 *operations: ops.OP_TREE,
                 device: devices.Device = devices.UnconstrainedDevice
                 ) -> TSelf:
        dag = cls(device=device)
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
        if len(self.nodes) <= 0:
            return
        g = self.copy()

        def get_root_node(some_node):
            pred = g.pred
            while len(pred[some_node]) > 0:
                some_node = next(iter(pred[some_node]))
            return some_node

        def get_first_node():
            return get_root_node(next(iter(g.nodes)))

        def get_next_node(succ):
            if len(succ) > 0:
                return get_root_node(next(iter(succ)))
            else:
                return get_first_node()

        node = get_first_node()
        while True:
            yield node.val
            succ = g.succ[node]
            g.remove_node(node)

            if len(g.nodes) <= 0:
                return

            node = get_next_node(succ)

    def to_circuit(self) -> circuit.Circuit:
        return circuit.Circuit.from_ops(
                    self.all_operations(),
                    strategy=circuit.InsertStrategy.EARLIEST,
                    device=self.device)


class CircuitDag(CircuitDagAbc):
    """A representation of a Circuit as a directed acyclic graph.  Gate can
    reorder only if they don't share qubits."""
    def can_reorder(self,
                    op1: ops.Operation,
                    op2: ops.Operation) -> bool:
        return len(set(op1.qubits) & set(op2.qubits)) == 0
