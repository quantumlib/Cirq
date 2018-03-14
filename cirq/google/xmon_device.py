# Copyright 2018 Google LLC
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

from typing import Iterable

from cirq import ops
from cirq.devices import Device
from cirq.google import xmon_gates
from cirq.google.xmon_gate_extensions import xmon_gate_ext
from cirq.google.xmon_qubit import XmonQubit
from cirq.time import Duration


from cirq.circuits import TextDiagramDrawer

class XmonDevice(Device):
    """A device with qubits placed in a grid. Neighboring qubits can interact.
    """

    def __init__(self,
                 measurement_duration: Duration,
                 exp_w_duration: Duration,
                 exp_11_duration: Duration,
                 qubits: Iterable[XmonQubit]):
        """Initializes the description of an xmon device.

        Args:
            measurement_duration: The maximum duration of a measurement.
            exp_w_duration: The maximum duration of an ExpW operation.
            exp_11_duration: The maximum duration of an ExpZ operation.
            qubits: Qubits on the device, identified by their x, y location.
        """
        self._measurement_duration = measurement_duration
        self._exp_w_duration = exp_w_duration
        self._exp_z_duration = exp_11_duration
        self.qubits = frozenset(qubits)

    def neighbors_of(self, qubit: XmonQubit):
        """Returns the qubits that the given qubit can interact with."""
        possibles = [
            XmonQubit(qubit.row + 1, qubit.col),
            XmonQubit(qubit.row - 1, qubit.col),
            XmonQubit(qubit.row, qubit.col + 1),
            XmonQubit(qubit.row, qubit.col - 1),
        ]
        return [e for e in possibles if e in self.qubits]

    def duration_of(self, operation):
        g = xmon_gate_ext.try_cast(operation.gate, xmon_gates.XmonGate)
        if isinstance(g, xmon_gates.Exp11Gate):
            return self._exp_z_duration
        if isinstance(g, xmon_gates.ExpWGate):
            return self._exp_w_duration
        if isinstance(g, xmon_gates.XmonMeasurementGate):
            return self._measurement_duration
        if isinstance(g, xmon_gates.ExpZGate):
            return Duration()  # Z gates are performed in the control software.
        raise ValueError('Unsupported gate type: {}'.format(repr(g)))

    def validate_gate(self, gate: ops.Gate):
        """Raises an error if the given gate isn't allowed.

        Raises:
            ValueError: Unsupported gate.
        """
        if not isinstance(gate, (xmon_gates.Exp11Gate,
                                 xmon_gates.ExpWGate,
                                 xmon_gates.XmonMeasurementGate,
                                 xmon_gates.ExpZGate)):
            raise ValueError('Unsupported gate type: {}'.format(repr(gate)))

    def validate_operation(self, operation):
        self.validate_gate(operation.gate)

        for q in operation.qubits:
            if not isinstance(q, XmonQubit):
                raise ValueError('Unsupported qubit type: {}'.format(repr(q)))
            if q not in self.qubits:
                raise ValueError('Qubit not on device: {}'.format(repr(q)))

        if len(operation.qubits) == 2:
            p, q = operation.qubits
            if not p.is_adjacent(q):
                raise ValueError(
                    'Non-local interaction: {}.'.format(repr(operation)))

    def check_if_exp11_operation_interacts(self,
                                           exp11_op: ops.Operation,
                                           other_op: ops.Operation) -> bool:
        if isinstance(other_op.gate, xmon_gates.ExpZGate):
            return False
        # Adjacent ExpW operations may be doable.
        # For now we will play it conservatively.

        return any(q.is_adjacent(p)
                   for q in exp11_op.qubits
                   for p in other_op.qubits)

    def validate_scheduled_operation(self, schedule, scheduled_operation):
        self.validate_operation(scheduled_operation.operation)

        if isinstance(scheduled_operation.operation.gate,
                      xmon_gates.Exp11Gate):
            for other in schedule.operations_happening_at_same_time_as(
                    scheduled_operation):
                if self.check_if_exp11_operation_interacts(
                        scheduled_operation.operation,
                        other.operation):
                    raise ValueError(
                        'Adjacent Exp11 operations: {} vs {}.'.format(
                            scheduled_operation, other))

    def validate_circuit(self, circuit):
        for moment in circuit.moments:
            for operation in moment.operations:
                self.validate_operation(operation)

    def validate_schedule(self, schedule):
        for scheduled_operation in schedule.scheduled_operations:
            self.validate_scheduled_operation(schedule, scheduled_operation)

    def __str__(self):
        diagram = TextDiagramDrawer()

        for q in self.qubits:
            diagram.write(q.col, q.row, str(q))
            for q2 in self.neighbors_of(q):
                if q2.col != q.col:
                    diagram.horizontal_line(q.row, q.col, q2.col)
                else:
                    diagram.vertical_line(q.col, q.row, q2.row)

        return diagram.render(
            horizontal_spacing=3,
            vertical_spacing=2,
            use_unicode_characters=True)
