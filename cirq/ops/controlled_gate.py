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

from typing import Any, List, Union

import numpy as np

import cirq
from cirq import linalg, protocols, value
from cirq.ops import (raw_types, controlled_operation as cop, gate_operation)
from cirq.type_workarounds import NotImplementedType


@value.value_equality
class ControlledGate(raw_types.Gate):
    """Augments existing gates with a control qubit."""

    def __init__(self, sub_gate: raw_types.Gate,
                 control_qubits: List[raw_types.QubitId] = None,
                 num_controls: int = 1) -> None:
        """Initializes the controlled gate.

        Args:
            sub_gate: The gate to add a control qubit to.
            control_qubits: The qubits that would act as controls.
            num_control: Total number of control qubits.
        """
        if control_qubits is None:
            control_qubits = []
        if num_controls < len(control_qubits):
            raise ValueError('More specified control qubits than num_controls')

        # Leave unspecified controls as Nones.
        self.control_qubits = ([None]*(num_controls - # type: ignore
                                       len(control_qubits)) +
                               control_qubits)

        self.num_controls = num_controls

        # Flatten nested ControlledGates.
        if isinstance(sub_gate, ControlledGate):
            self.sub_gate = sub_gate.sub_gate # type: ignore
            self.control_qubits += sub_gate.control_qubits
            self.num_controls += sub_gate.num_controls
        else:
            self.sub_gate = sub_gate

        self.num_unspecified_controls = self.control_qubits.count(None)

    def num_qubits(self) -> int:
        return self.sub_gate.num_qubits() + self.num_controls

    def _decompose_(self, qubits):
        result = protocols.decompose_once_with_qubits(self.sub_gate,
                                            qubits[self.num_controls:],
                                            NotImplemented)

        if result is NotImplemented:
            return NotImplemented

        decomposed = []
        for op in result:
            controlled_op = op
            for qubit in qubits[:self.num_controls]:
                controlled_op = cop.ControlledOperation(qubit, controlled_op)
            decomposed.append(controlled_op)
        return decomposed

    def validate_args(self, qubits) -> None:
        if len(qubits) < self.num_unspecified_controls:
            raise ValueError('Not all control qubits specified.')
        self.sub_gate.validate_args(qubits[self.num_unspecified_controls:])

    def on(self, *qubits: raw_types.QubitId) -> gate_operation.GateOperation:
        if len(qubits) == 0:
            raise ValueError(
                "Applied a gate to an empty set of qubits. Gate: {}".format(
                    repr(self)))
        self.validate_args(qubits)

        # Merge specified controls and new controls.
        j=0
        merged_controls = []
        for control in self.control_qubits:
            if control is None:
                merged_controls.append(qubits[j])
                j+=1
            else:
                merged_controls.append(control)

        # Convert to ControlledGate without control qubits that can be reused.
        self.control_qubits = [None]*self.num_controls
        self.num_unspecified_controls = self.num_controls

        return gate_operation.GateOperation(self,
                                            merged_controls+list(qubits[j:]))

    def _value_equality_values_(self):
        return self.sub_gate, self.num_controls

    def _apply_unitary_(self, args: protocols.ApplyUnitaryArgs) -> np.ndarray:
        qubits = cirq.LineQubit.range(self.num_controls +
                                      self.sub_gate.num_qubits())
        c_op = self.sub_gate.on(*qubits[self.num_controls:])
        for control in qubits[:self.num_controls]:
            c_op = cop.ControlledOperation(control, c_op)

        return protocols.apply_unitary(c_op, args, default=NotImplemented)

    def _has_unitary_(self) -> bool:
        return protocols.has_unitary(self.sub_gate)

    def _unitary_(self) -> Union[np.ndarray, NotImplementedType]:
        sub_matrix = protocols.unitary(self.sub_gate, None)
        if sub_matrix is None:
            return NotImplemented
        for _ in range(self.num_controls):
            sub_matrix = linalg.block_diag(np.eye(sub_matrix.shape[0]),
                                           sub_matrix)
        return sub_matrix

    def __pow__(self, exponent: Any) -> 'ControlledGate':
        new_sub_gate = protocols.pow(self.sub_gate,
                                     exponent,
                                     NotImplemented)
        if new_sub_gate is NotImplemented:
            return NotImplemented
        return ControlledGate(new_sub_gate, self.control_qubits, # type: ignore
                              self.num_controls)

    def _is_parameterized_(self):
        return protocols.is_parameterized(self.sub_gate)

    def _resolve_parameters_(self, param_resolver):
        new_sub_gate = protocols.resolve_parameters(self.sub_gate,
                                                    param_resolver)
        return ControlledGate(new_sub_gate, self.control_qubits,
                              self.num_controls)

    def _trace_distance_bound_(self):
        return protocols.trace_distance_bound(self.sub_gate)

    def _circuit_diagram_info_(self,
                               args: protocols.CircuitDiagramInfoArgs
                               ) -> protocols.CircuitDiagramInfo:
        sub_args = protocols.CircuitDiagramInfoArgs(
            known_qubit_count=(args.known_qubit_count - self.num_controls
                               if args.known_qubit_count is not None else None),
            known_qubits=(args.known_qubits[self.num_controls:]
                          if args.known_qubits is not None else None),
            use_unicode_characters=args.use_unicode_characters,
            precision=args.precision,
            qubit_map=args.qubit_map
        )
        sub_info = protocols.circuit_diagram_info(self.sub_gate,
                                                  sub_args,
                                                  None)
        if sub_info is None:
            return NotImplemented
        return protocols.CircuitDiagramInfo(
            wire_symbols=('@',)*self.num_controls + sub_info.wire_symbols,
            exponent=sub_info.exponent)

    def __str__(self):
        return 'C'*self.num_controls + str(self.sub_gate)

    def __repr__(self):
        if (self.num_controls is 1 and self.num_unspecified_controls is 1):
            return 'cirq.ControlledGate(sub_gate={!r})'.format(self.sub_gate)

        return ('cirq.ControlledGate(sub_gate={!r}, control_qubits={!r}, '
                    'num_controls={!r})'.
                    format(self.sub_gate, self.control_qubits,
                           self.num_controls))
