# Copyright 2021 The Cirq Developers
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

import typing

import networkx

from cirq.circuits.circuit import Circuit
from cirq.ops.moment import Moment
from cirq.ops.raw_types import Qid, Operation


class CircuitDagRepresentation(networkx.DiGraph):
    """A Directed Acyclic Graph representation of the Circuit
    The DAG is built on the operations, two operations are connected if
    they act on the same qubit, and the moments they belong to are labelled.
    """

    def __init__(self, circuit: Circuit = None):
        """Initializes a Circuit Dag representation.

        Args:
            circuit: The circuit for which we need to get the representation
        """
        super().__init__()
        self.num_moments = 0
        self.device = None
        if circuit is not None:
            self.from_circuit(circuit)

    def from_circuit(self, circuit: Circuit) -> None:
        """Loads the circuit into a DAG representation, clears out whatever
        graph existed before it.

        Args:
            circuit: The circuit for which we need to get the representation
        """
        self.clear()
        frontier: typing.Dict[Qid, int] = dict()
        operation_id = 0
        for idx, moment in enumerate(circuit.moments):
            for op in moment.operations:
                operation_id += 1
                self.add_node(operation_id, op=op, moment=idx)
                for qubit_operands in op.qubits:
                    if qubit_operands in frontier:
                        self.add_edge(operation_id, frontier[qubit_operands])
                    frontier[qubit_operands] = operation_id
        self.num_moments = max(self.num_moments, len(circuit.moments))

    def to_circuit(self) -> Circuit:
        """Initializes a Circuit Dag representation.

        Returns:
            The representation converted back to the circuit
        """
        operations_list: typing.List[typing.List[Operation]] = [[] for _ in range(self.num_moments)]
        for _node_idx, node in self.nodes(data=True):
            operations_list[node['moment']].append(node['op'])
        moments_list: typing.List[Moment] = list(map(Moment, operations_list))
        circuit = Circuit(moments_list)
        return circuit

    def __repr__(self):
        return f'cirq.{self.__class__.__name__}({repr(self.to_circuit())})'

    def _json_dict_(self):
        return {'cirq_type': self.__class__.__name__, 'circuit': self.to_circuit()}

    @classmethod
    def _from_json_dict_(cls, circuit, **kwargs):
        return cls(circuit=circuit)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.to_circuit() == other.to_circuit()
        elif isinstance(other, Circuit):
            return self.to_circuit() == other
        else:
            return False
