# Copyright 2024 The Cirq Developers
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

"""Target gateset used for compiling circuits to IonQ native gates."""
from typing import Any
from typing import Dict

import cirq
from cirq.protocols.decompose_protocol import DecomposeResult
from ionq_native_gates import GPIGate, GPI2Gate, MSGate, ZZGate

class AriaNativeGateset(cirq.TwoQubitCompilationTargetGateset):
    """Target IonQ native gateset for compiling circuits.

    The gates forming this gateset are: 
    GPIGate, GPI2Gate, MSGate and ZZGate !???
    """

    def __init__(self, *, atol: float = 1e-8):
        """Initializes AriaNativeGateset

        Args:
            atol: A limit on the amount of absolute error introduced by the decomposition.
        """
        super().__init__(
            GPIGate,
            GPI2Gate,
            MSGate,
            # ops.MeasurementGate ?, from cirq import ops
            unroll_circuit_op=False,
        )
        self.atol = atol

    # TODO: implement using native gates
    def _decompose_single_qubit_operation(self, op: cirq.Operation, _) -> cirq.OP_TREE:
        qubit = op.qubits[0]
        mat = cirq.unitary(op)
        for gate in cirq.single_qubit_matrix_to_gates(mat, self.atol):
            yield gate(qubit)

    # TODO - implement
    def _decompose_two_qubit_operation(self, op: cirq.Operation, _) -> cirq.OP_TREE:
        pass
    
    # TODO - implement
    def _decompose_multi_qubit_operation(self, op: 'cirq.Operation', moment_idx: int) -> DecomposeResult:
        pass

    # TODO - implement
    def __repr__(self) -> str:
        return f'cirq_ionq.AriaNativeGateset(atol={self.atol})'

    def _value_equality_values_(self) -> Any:
        return self.atol

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ['atol'])

    @classmethod
    def _from_json_dict_(cls, atol, **kwargs):
        return cls(atol=atol)


class ForteNativeGateset(cirq.TwoQubitCompilationTargetGateset):
    """Target IonQ native gateset for compiling circuits.

    The gates forming this gateset are: 
    GPIGate, GPI2Gate, MSGate and ZZGate !???
    """

    def __init__(self, *, atol: float = 1e-8):
        """Initializes AriaNativeGateset

        Args:
            atol: A limit on the amount of absolute error introduced by the decomposition.
        """
        super().__init__(
            GPIGate,
            GPI2Gate,
            MSGate,
            ZZGate,
            # ops.MeasurementGate ?, from cirq import ops
            unroll_circuit_op=False,
        )
        self.atol = atol

    # TODO: implement using native gates
    def _decompose_single_qubit_operation(self, op: cirq.Operation, _) -> cirq.OP_TREE:
        qubit = op.qubits[0]
        mat = cirq.unitary(op)
        for gate in cirq.single_qubit_matrix_to_gates(mat, self.atol):
            yield gate(qubit)

    # TODO - implement
    def _decompose_two_qubit_operation(self, op: cirq.Operation, _) -> cirq.OP_TREE:
        pass
    
    # TODO - implement
    def _decompose_multi_qubit_operation(self, op: 'cirq.Operation', moment_idx: int) -> DecomposeResult:
        pass

    # TODO - implement
    def __repr__(self) -> str:
        return f'cirq_ionq.AriaNativeGateset(atol={self.atol})'

    def _value_equality_values_(self) -> Any:
        return self.atol

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ['atol'])

    @classmethod
    def _from_json_dict_(cls, atol, **kwargs):
        return cls(atol=atol)