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

from cirq import linalg, protocols, value
from cirq.ops import raw_types, controlled_operation as cop
from cirq.type_workarounds import NotImplementedType


@value.value_equality
class ControlledGate(raw_types.Gate):
    """Augments existing gates with a control qubit."""

    def __init__(self, sub_gate: raw_types.Gate,
                 control_qubits: List[raw_types.QubitId] = None,
                 num_unspecified_control_qubits: int = None) -> None:
        """Initializes the controlled gate.

        Args:
            sub_gate: The gate to add a control qubit to.
            control_qubits: The qubits that would act as controls.
            num_unspecified_control_qubits: Additional control qubits that
                                            haven't been specified yet.
        """
        if control_qubits is None:
            control_qubits = []
        self.control_qubits = control_qubits

        if num_unspecified_control_qubits is None:
            if len(control_qubits) is 0:
                self.num_unspecified_control_qubits = 1
            else:
                self.num_unspecified_control_qubits = 0
        else:
            self.num_unspecified_control_qubits = \
                 num_unspecified_control_qubits

        # Flatten nested ControlledGates.
        if isinstance(sub_gate, ControlledGate):
            self.sub_gate = sub_gate.sub_gate # type: ignore
            self.control_qubits += sub_gate.control_qubits
            self.num_unspecified_control_qubits += \
                 sub_gate.num_unspecified_control_qubits
        else:
            self.sub_gate = sub_gate

        self.total_control_qubits = (len(self.control_qubits) +
                                     self.num_unspecified_control_qubits)

    def num_qubits(self) -> int:
        return self.sub_gate.num_qubits() + self.num_unspecified_control_qubits

    def _decompose_(self, qubits):
        result = protocols.decompose_once_with_qubits(self.sub_gate,
                                            qubits[self.total_control_qubits:],
                                            NotImplemented)

        if result is NotImplemented:
            return NotImplemented

        decomposed = []
        for op in result:
            controlled_op = op
            for qubit in qubits[:self.total_control_qubits]:
                controlled_op = cop.ControlledOperation(qubit, controlled_op)
            decomposed.append(controlled_op)
        return decomposed

    def validate_args(self, qubits) -> None:
        if len(qubits) < self.num_unspecified_control_qubits:
            raise ValueError('No control qubit specified.')
        self.sub_gate.validate_args(
                          qubits[self.num_unspecified_control_qubits:])

    def _value_equality_values_(self):
        return (self.sub_gate, self.total_control_qubits)

    def _apply_unitary_(self, args: protocols.ApplyUnitaryArgs) -> np.ndarray:
        control = args.axes[0]
        rest = args.axes[1:]
        active = linalg.slice_for_qubits_equal_to([control], 1)
        sub_axes = [r - int(r > control) for r in rest]
        target_view = args.target_tensor[active]
        buffer_view = args.available_buffer[active]
        if self.total_control_qubits is 1:
            result = protocols.apply_unitary(
                self.sub_gate,
                protocols.ApplyUnitaryArgs(
                    target_view,
                    buffer_view,
                    sub_axes),
                default=NotImplemented)
        else:
            result = protocols.apply_unitary(
                ControlledGate(self.sub_gate, num_unspecified_control_qubits =
                                                  self.total_control_qubits-1),
                protocols.ApplyUnitaryArgs(
                    target_view,
                    buffer_view,
                    sub_axes),
                default=NotImplemented)

        if result is NotImplemented:
            return NotImplemented

        if result is target_view:
            return args.target_tensor

        if result is buffer_view:
            inactive = linalg.slice_for_qubits_equal_to([control], 0)
            args.available_buffer[inactive] = args.target_tensor[inactive]
            return args.available_buffer

        # HACK: assume they didn't somehow escape the slice view and edit the
        # rest of target_tensor.
        args.target_tensor[active] = result
        return args.target_tensor

    def _has_unitary_(self) -> bool:
        return protocols.has_unitary(self.sub_gate)

    def _unitary_(self) -> Union[np.ndarray, NotImplementedType]:
        sub_matrix = protocols.unitary(self.sub_gate, None)
        if sub_matrix is None:
            return NotImplemented
        for _ in range(self.total_control_qubits):
            sub_matrix = linalg.block_diag(np.eye(sub_matrix.shape[0]),
                                                  sub_matrix)
        return sub_matrix

    def __pow__(self, exponent: Any) -> 'ControlledGate':
        new_sub_gate = protocols.pow(self.sub_gate,
                                     exponent,
                                     NotImplemented)
        if new_sub_gate is NotImplemented:
            return NotImplemented
        return ControlledGate(new_sub_gate, self.control_qubits,
                              self.num_unspecified_control_qubits)

    def _is_parameterized_(self):
        return protocols.is_parameterized(self.sub_gate)

    def _resolve_parameters_(self, param_resolver):
        new_sub_gate = protocols.resolve_parameters(self.sub_gate,
                                                    param_resolver)
        return ControlledGate(new_sub_gate, self.control_qubits,
                              self.num_unspecified_control_qubits)

    def _trace_distance_bound_(self):
        return protocols.trace_distance_bound(self.sub_gate)

    def _circuit_diagram_info_(self,
                               args: protocols.CircuitDiagramInfoArgs
                               ) -> protocols.CircuitDiagramInfo:
        sub_info = protocols.circuit_diagram_info(self.sub_gate, args, None)
        if sub_info is None:
            return NotImplemented
        return protocols.CircuitDiagramInfo(
            wire_symbols=('@',)*self.total_control_qubits +
                         sub_info.wire_symbols,
            exponent=sub_info.exponent)

    def __str__(self):
        return 'C'*self.total_control_qubits + str(self.sub_gate)

    def __repr__(self):
        if (len(self.control_qubits) is 0 and
            self.num_unspecified_control_qubits is 1):
            return 'cirq.ControlledGate(sub_gate={!r})'.format(self.sub_gate)
        else:
            return ('cirq.ControlledGate(sub_gate={!r}, control_qubits={!r}, '
                    'num_unspecified_control_qubits={!r})'.
                    format(self.sub_gate, self.control_qubits,
                           self.num_unspecified_control_qubits))
