# Copyright 2020 The Cirq Developers
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
"""A structure for encapsulating entire circuits in a gate.

A CircuitGate is a Gate object that wraps a FrozenCircuit. When applied as part
of a larger circuit, a CircuitGate will execute all component gates in order,
including any nested CircuitGates.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Union

from cirq import circuits, devices, ops, protocols
from cirq.circuits.insert_strategy import InsertStrategy
from cirq.type_workarounds import NotImplementedType

if TYPE_CHECKING:
    import cirq


class CircuitGate(ops.Gate):
    """A single gate that encapsulates a Circuit."""

    def __init__(self,
                 *contents: 'cirq.OP_TREE',
                 strategy: 'cirq.InsertStrategy' = InsertStrategy.EARLIEST,
                 device: 'cirq.Device' = devices.UNCONSTRAINED_DEVICE,
                 name: Optional[str] = None,
                 exp_modulus: Optional[int] = None) -> None:
        """Constructs a gate that wraps a circuit.

        Args:
            contents: The initial list of moments and operations defining the
                circuit. This accepts the same set of objects as the Circuit
                constructor.
            strategy: When initializing the circuit with operations and moments
                from `contents`, this determines how the operations are packed
                together.
            device: Hardware that the circuit should be able to run on.
            name: (Optional) A name for this gate. This will appear alongside
                the gate's hash in its serialized form.
            exp_modulus: (Optional) The root of identity that this gate
                represents. When an operation using this gate is looped over,
                it will consult this value to minimize the total operations
                performed. CircuitGates containing measurements cannot have
                an exp_modulus value.
        """
        self._circuit = circuits.FrozenCircuit(contents,
                                               strategy=strategy,
                                               device=device)
        self._name = name
        if exp_modulus is not None and protocols.is_measurement(self._circuit):
            raise ValueError(
                'CircuitGates with measurement cannot have exp_modulus.')
        self._exp_modulus = exp_modulus

    @property
    def circuit(self):
        return self._circuit

    @property
    def name(self):
        return self._name

    @property
    def exp_modulus(self):
        return self._exp_modulus

    def _num_qubits_(self):
        return protocols.num_qubits(self.circuit)

    def _qid_shape_(self):
        return protocols.qid_shape(self.circuit)

    def on(self, *qubits: 'cirq.Qid') -> 'CircuitOperation':
        return CircuitOperation(self, qubits)

    def _with_measurement_key_mapping_(self, key_map: Dict[str, str]):
        new_gate = CircuitGate()
        new_gate._circuit = protocols.with_measurement_key_mapping(
            self.circuit, key_map)
        new_gate._name = self.name
        new_gate._exp_modulus = self.exp_modulus
        return new_gate

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        return (self.circuit == other.circuit and self.name == other.name and
                self.exp_modulus == other.exp_modulus)

    def __hash__(self):
        return hash((self.circuit, self.name, self.exp_modulus))

    def __pow__(self, power):
        try:
            return CircuitGate(self.circuit**power, device=self.circuit.device)
        except:
            return NotImplemented

    def __repr__(self):
        base_repr = repr(self.circuit)
        if self.name is not None:
            base_repr += f', name={self.name}'
        if self.exp_modulus is not None:
            base_repr += f', exp_modulus={self.exp_modulus}'
        return f'cirq.CircuitGate({base_repr})'

    def __str__(self):
        header = 'CircuitGate:' if self.name is None else f'{self.name}:'
        msg_lines = str(self.circuit).split('\n')
        msg_width = max([len(header) - 4] + [len(line) for line in msg_lines])
        full_msg = '\n'.join([
            '[ {line:<{width}} ]'.format(line=line, width=msg_width)
            for line in msg_lines
        ])
        return header + '\n' + full_msg

    def _commutes_(self, other: Any,
                   atol: float) -> Union[None, NotImplementedType, bool]:
        return protocols.commutes(self.circuit, other, atol=atol)

    def _has_unitary_(self):
        return protocols.has_unitary(self.circuit)

    def _decompose_(self, qubits=None):
        applied_circuit = self.circuit.unfreeze()
        if qubits is not None and qubits != self.ordered_qubits():
            qmap = {old: new for old, new in zip(self.ordered_qubits(), qubits)}
            applied_circuit = applied_circuit.transform_qubits(lambda q: qmap[q]
                                                              )
        return protocols.decompose(applied_circuit)

    def ordered_qubits(
            self,
            qubit_order: 'cirq.QubitOrderOrList' = ops.QubitOrder.DEFAULT,
    ):
        return ops.QubitOrder.as_qubit_order(qubit_order).order_for(
            self.circuit.all_qubits())

    def _json_dict_(self):
        return protocols.obj_to_dict_helper(self,
                                            ['circuit', 'name', 'exp_modulus'])

    @classmethod
    def _from_json_dict_(cls, circuit, name, exp_modulus, **kwargs):
        return cls(circuit,
                   strategy=InsertStrategy.EARLIEST,
                   device=circuit.device,
                   name=name,
                   exp_modulus=exp_modulus)


class CircuitOperation(ops.GateOperation):
    """An operation that encapsulates a circuit.

    This class captures modifications to the contained circuit, such as tags
    and loops, to support more condensed serialization.
    """

    # TODO: Fully implement this class.

    def __init__(self, gate: 'CircuitGate', qubits: Sequence['cirq.Qid']):
        if not isinstance(gate, CircuitGate):
            raise TypeError('CircuitOperations must contain CircuitGates.')
        super().__init__(gate, qubits)
