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

from typing import Any, Union

import numpy as np

import cirq
from cirq import linalg, protocols, value
from cirq.ops import raw_types, controlled_operation as cop
from cirq.type_workarounds import NotImplementedType


@value.value_equality
class ControlledGate(raw_types.Gate):
    """Augments existing gates with a control qubit."""

    def __init__(self, sub_gate: raw_types.Gate) -> None:
        """Initializes the controlled gate.

        Args:
            sub_gate: The gate to add a control qubit to.
        """
        self.sub_gate = sub_gate

    def num_qubits(self) -> int:
        return self.sub_gate.num_qubits() + 1

    def _decompose_(self, qubits):
        result = protocols.decompose_once_with_qubits(self.sub_gate,
                                                      qubits[1:],
                                                      NotImplemented)
        if result is NotImplemented:
            return NotImplemented

        return [cop.ControlledOperation(qubits[0], op) for op in result]

    def validate_args(self, qubits) -> None:
        if len(qubits) < 1:
            raise ValueError('No control qubit specified.')
        self.sub_gate.validate_args(qubits[1:])

    def _value_equality_values_(self):
        return self.sub_gate

    def _apply_unitary_(self, args: protocols.ApplyUnitaryArgs) -> np.ndarray:
        qubits = cirq.LineQubit.range(1 + self.sub_gate.num_qubits())
        c_op = cop.ControlledOperation(qubits[0],
                                       self.sub_gate.on(*qubits[1:]))

        return protocols.apply_unitary(c_op, args, default=NotImplemented)

    def _has_unitary_(self) -> bool:
        return protocols.has_unitary(self.sub_gate)

    def _unitary_(self) -> Union[np.ndarray, NotImplementedType]:
        sub_matrix = protocols.unitary(self.sub_gate, None)
        if sub_matrix is None:
            return NotImplemented
        return linalg.block_diag(np.eye(sub_matrix.shape[0]), sub_matrix)

    def __pow__(self, exponent: Any) -> 'ControlledGate':
        new_sub_gate = protocols.pow(self.sub_gate,
                                     exponent,
                                     NotImplemented)
        if new_sub_gate is NotImplemented:
            return NotImplemented
        return ControlledGate(new_sub_gate)

    def _is_parameterized_(self):
        return protocols.is_parameterized(self.sub_gate)

    def _resolve_parameters_(self, param_resolver):
        new_sub_gate = protocols.resolve_parameters(self.sub_gate,
                                                    param_resolver)
        return ControlledGate(new_sub_gate)

    def _trace_distance_bound_(self):
        return protocols.trace_distance_bound(self.sub_gate)

    def _circuit_diagram_info_(self,
                               args: protocols.CircuitDiagramInfoArgs
                               ) -> protocols.CircuitDiagramInfo:
        sub_args = protocols.CircuitDiagramInfoArgs(
            known_qubit_count=(args.known_qubit_count - 1
                               if args.known_qubit_count is not None else None),
            known_qubits=(args.known_qubits[1:]
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
            wire_symbols=('@',) + sub_info.wire_symbols,
            exponent=sub_info.exponent)

    def __str__(self):
        return 'C' + str(self.sub_gate)

    def __repr__(self):
        return 'cirq.ControlledGate(sub_gate={!r})'.format(self.sub_gate)
