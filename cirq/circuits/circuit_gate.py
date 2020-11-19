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

from typing import (TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple,
                    Union)

import numpy as np
import sympy

from cirq import circuits, devices, linalg, ops, protocols
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
        self._exp_modulus = exp_modulus
        if exp_modulus is not None:
            if protocols.is_measurement(self.circuit):
                raise ValueError(
                    'CircuitGates with measurement cannot have exp_modulus.')
            if not linalg.allclose_up_to_global_phase(
                    protocols.unitary(self.circuit * exp_modulus),
                    np.eye(2 ** protocols.num_qubits(self.circuit))):
                raise ValueError('Raising the gate contents to exp_modulus '
                                 'must produce the identity.')

    @property
    def circuit(self) -> 'cirq.FrozenCircuit':
        """The circuit represented by this gate."""
        return self._circuit

    @property
    def name(self) -> Optional[str]:
        """The name used to represent this gate in serialization."""
        return self._name

    @property
    def exp_modulus(self) -> Optional[int]:
        """The exponential modulus of this gate.

        Operations which wrap this gate will have their exponent mapped to the
        range [-exp_modulus / 2, exp_modulus / 2) if invertible, or
        [0, exp_modulus) otherwise.
        """
        return self._exp_modulus

    def _num_qubits_(self) -> int:
        return protocols.num_qubits(self.circuit)

    def _qid_shape_(self) -> Tuple[int, ...]:
        return protocols.qid_shape(self.circuit)

    def on(self, *qubits: 'cirq.Qid') -> 'CircuitOperation':
        return CircuitOperation(self, qubits)

    def _with_measurement_key_mapping_(self, key_map: Dict[str, str]
                                      ) -> 'CircuitGate':
        new_gate = CircuitGate()
        new_gate._circuit = protocols.with_measurement_key_mapping(
            self.circuit, key_map)
        new_gate._name = self.name
        new_gate._exp_modulus = self.exp_modulus
        return new_gate

    def __eq__(self, other) -> bool:
        if type(other) != type(self):
            return False
        return (self.circuit == other.circuit and self.name == other.name and
                self.exp_modulus == other.exp_modulus)

    def __hash__(self) -> int:
        return hash((self.circuit, self.name, self.exp_modulus))

    def __pow__(self, power) -> 'CircuitGate':
        try:
            return CircuitGate(self.circuit**power, device=self.circuit.device)
        except:
            return NotImplemented

    def __repr__(self) -> str:
        base_repr = repr(self.circuit)
        if self.name is not None:
            base_repr += f""", name='{self.name}'"""
        if self.exp_modulus is not None:
            base_repr += f', exp_modulus={self.exp_modulus}'
        return f'cirq.CircuitGate({base_repr})'

    def __str__(self) -> str:
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
        # Until Circuit supports _commutes_, this will return NotImplemented.
        return protocols.commutes(self.circuit, other, atol=atol)

    def _has_unitary_(self) -> bool:
        return protocols.has_unitary(self.circuit)

    def _decompose_(self, qubits=None) -> 'cirq.OP_TREE':
        applied_circuit = self.circuit.unfreeze()
        if qubits is not None and qubits != self.ordered_qubits():
            qmap = {old: new for old, new in zip(self.ordered_qubits(), qubits)}
            applied_circuit = applied_circuit.transform_qubits(lambda q: qmap[q]
                                                              )
        return protocols.decompose(applied_circuit)

    def ordered_qubits(
            self,
            qubit_order: 'cirq.QubitOrderOrList' = ops.QubitOrder.DEFAULT,
    ) -> Sequence['cirq.Qid']:
        """Returns the qubits in the contained circuit.

        Args:
            qubit_order: The order in which to return the circuit's qubits.
        """
        return ops.QubitOrder.as_qubit_order(qubit_order).order_for(
            self.circuit.all_qubits())

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self,
                                            ['circuit', 'name', 'exp_modulus'])

    @classmethod
    def _from_json_dict_(cls, circuit, name, exp_modulus,
                         **kwargs) -> 'CircuitGate':
        return cls(circuit,
                   strategy=InsertStrategy.EARLIEST,
                   device=circuit.device,
                   name=name,
                   exp_modulus=exp_modulus)


class CircuitOperation(ops.GateOperation):
    """An operation that encapsulates a circuit.

    This class captures modifications to the contained circuit, such as tags
    and loops, to support more condensed serialization.

    Similar to GateOperation, this type is immutable.
    """

    def __init__(self, gate: 'CircuitGate', qubits: Sequence['cirq.Qid']):
        super().__init__(gate, qubits)
        self._gate: 'CircuitGate'
        self._measurement_key_map: Dict[str, str] = {}
        self._param_values: Dict[sympy.Symbol, Any] = {}
        self._repetitions: int = 1

    @property
    def gate(self) -> 'CircuitGate':
        return self._gate

    @property
    def circuit(self):
        return self.gate.circuit

    @property
    def measurement_key_map(self):
        return self._measurement_key_map

    @property
    def param_values(self):
        return self._param_values

    @property
    def repetitions(self):
        return self._repetitions

    def __repr__(self):
        base_repr = super().__repr__()
        if self._measurement_key_map:
            base_repr += (
                f'.with_measurement_key_mapping({self._measurement_key_map})')
        if self._param_values:
            base_repr += f'.with_params({self._param_values})'
        if self._repetitions:
            base_repr += f'.repeat({self._repetitions})'
        return base_repr

    def base_operation(self):
        return CircuitOperation(self._gate, self._qubits)

    def copy(self):
        new_op = self.base_operation()
        new_op._measurement_key_map = self._measurement_key_map
        new_op._param_values = self._param_values
        new_op._repetitions = self._repetitions
        return new_op

    def with_gate(self, new_gate: 'cirq.Gate'):
        if not isinstance(new_gate, CircuitGate):
            raise TypeError('CircuitOperations may only contain CircuitGates.')
        if protocols.is_measurement(new_gate) and self._repetitions != 1:
            raise NotImplementedError(
                'Measurements cannot be added to looped circuits.')
        new_op = self.copy()
        new_op._gate = new_gate
        return new_op

    def with_measurement_key_mapping(self, key_map: Dict[str, str]):
        new_op = self.copy()
        new_op._measurement_key_map = {
            k: (key_map[v] if v in key_map else v)
            for k, v in self._measurement_key_map.items()
        }
        new_op._measurement_key_map.update({
            key: val
            for key, val in key_map.items()
            if key not in self._measurement_key_map.values()
        })
        return new_op

    def _with_measurement_key_mapping_(self, key_map: Dict[str, str]):
        return self.with_measurement_key_mapping(key_map)

    def with_params(self, param_values: Dict[sympy.Symbol, Any]):
        new_op = self.copy()
        new_op._param_values = {
            key: (val if not protocols.is_parameterized(val) else
                  protocols.resolve_parameters(val, param_values))
            for key, val in self._param_values.items()
        }
        new_op._param_values.update({
            key: val
            for key, val in param_values.items()
            if key not in self._param_values.values()
        })
        return new_op

    def repeat(self, repetitions: int):
        if protocols.is_measurement(self.circuit):
            raise NotImplementedError(
                'Loops over measurements are not supported.')
        new_op = self.copy()
        new_op._repetitions *= repetitions
        if self.gate.exp_modulus is not None:
            # Map repetitions to [-exp_modulus / 2, exp_modulus / 2)
            new_op._repetitions %= self.gate.exp_modulus
            if new_op._repetitions >= self.gate.exp_modulus / 2:
                new_op._repetitions -= self.gate.exp_modulus
        return new_op

    def __pow__(self, power: int):
        if not isinstance(power, int):
            return NotImplemented
        return self.repeat(power)

    def _measurement_keys_(self):
        base_keys = protocols.measurement_keys(self.gate)
        return set(self._measurement_key_map[key] if key in
                   self._measurement_key_map else key for key in base_keys)

    def _decompose_(self) -> 'cirq.OP_TREE':
        result = self.circuit.unfreeze()
        if (self.qubits is not None and
                self.qubits != self.gate.ordered_qubits()):
            qmap = {
                old: new
                for old, new in zip(self.gate.ordered_qubits(), self.qubits)
            }
            result = result.transform_qubits(lambda q: qmap[q])

        if self._repetitions < 0:
            result = result**-1

        if self._measurement_key_map:
            result = protocols.with_measurement_key_mapping(
                result, self._measurement_key_map)
        if self._param_values:
            result = protocols.resolve_parameters(result, self._param_values)

        return list(result.all_operations()) * abs(self._repetitions)
