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

from cirq import circuits, ops

class _PadAfterSwapGates(circuits.OptimizationPass):
    @staticmethod
    def is_swap_operation(op: ops.Operation):
        return isinstance(op, ops.GateOperation) and op.gate == ops.SWAP

    def optimize_circuit(self, circuit: circuits.Circuit):
        for i, moment in reversed(tuple(enumerate(circuit[:-1]))):
            swap_qubits = (
                q for op in moment.operations for q in op.qubits
                if self.is_swap_operation(op))
            following_operations = (circuit.operation_at(qubit, i + 1)
                for qubit in swap_qubits)
            if not all(op is None or self.is_swap_operation(op)
                    for op in following_operations):
                circuit.insert(i + 1, circuits.Moment())
            if any(self.is_swap_operation(op)
                    for op in circuit[-1].operations):
                circuit.append(circuits.Moment())

PadAfterSwapGates = _PadAfterSwapGates()
default_optimizers = (PadAfterSwapGates,)
