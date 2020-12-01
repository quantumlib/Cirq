# Copyright 2018 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable

from cirq import circuits, devices, ops


class QubitMapper:
    def __init__(self, qubit_map: Callable[[ops.Qid], ops.Qid]) -> None:
        self.qubit_map = qubit_map

    def map_operation(self, operation: ops.Operation) -> ops.Operation:
        return operation.transform_qubits(self.qubit_map)

    def map_moment(self, moment: ops.Moment) -> ops.Moment:
        return ops.Moment(self.map_operation(op) for op in moment.operations)

    def optimize_circuit(self, circuit: circuits.Circuit):
        circuit[:] = (self.map_moment(m) for m in circuit)


def linearize_circuit_qubits(
    circuit: circuits.Circuit, qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT
) -> None:
    qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(circuit.all_qubits())
    qubit_map = {q: devices.LineQubit(i) for i, q in enumerate(qubits)}
    QubitMapper(qubit_map.__getitem__).optimize_circuit(circuit)
