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

from datetime import timedelta
from typing import cast, Iterable, Optional, Union, TYPE_CHECKING

from cirq import circuits, value, devices, ops, protocols
from cirq.line import LineQubit
from cirq.ion import convert_to_ion_gates

if TYPE_CHECKING:
    # pylint: disable=unused-import
    from typing import Set


@value.value_equality
class IonDevice(devices.Device):
    """A device with qubits placed on a line.

    Qubits have all-to-all connectivity.
    """

    def __init__(self, measurement_duration: Union[value.Duration, timedelta],
                 twoq_gates_duration: Union[value.Duration, timedelta],
                 oneq_gates_duration: Union[value.Duration, timedelta],
                 qubits: Iterable[LineQubit]) -> None:
        """Initializes the description of an ion trap device.

        Args:
            measurement_duration: The maximum duration of a measurement.
            twoq_gates_duration: The maximum duration of a two qubit operation.
            oneq_gates_duration: The maximum duration of a single qubit
            operation.
            qubits: Qubits on the device, identified by their x, y location.
        """
        self._measurement_duration = value.Duration.create(measurement_duration)
        self._twoq_gates_duration = value.Duration.create(twoq_gates_duration)
        self._oneq_gates_duration = value.Duration.create(oneq_gates_duration)
        self.qubits = frozenset(qubits)

    def decompose_operation(self, operation: ops.Operation) -> ops.OP_TREE:
        return convert_to_ion_gates.ConvertToIonGates().convert_one(operation)

    def decompose_circuit(self, circuit: circuits.Circuit) -> circuits.Circuit:
        return convert_to_ion_gates.ConvertToIonGates().convert_circuit(circuit)

    def duration_of(self, operation):
        if ops.op_gate_of_type(operation, ops.XXPowGate):
            return self._twoq_gates_duration
        if (ops.op_gate_of_type(operation, ops.XPowGate) or
                ops.op_gate_of_type(operation, ops.YPowGate) or
                ops.op_gate_of_type(operation, ops.ZPowGate) or
                ops.op_gate_of_type(operation, ops.PhasedXPowGate)):
            return self._oneq_gates_duration
        if ops.op_gate_of_type(operation, ops.MeasurementGate):
            return self._measurement_duration
        raise ValueError('Unsupported gate type: {!r}'.format(operation))

    def validate_gate(self, gate: ops.Gate):
        if not isinstance(
                gate, (ops.XPowGate, ops.YPowGate, ops.ZPowGate,
                       ops.PhasedXPowGate, ops.XXPowGate, ops.MeasurementGate)):
            raise ValueError('Unsupported gate type: {!r}'.format(gate))

    def validate_operation(self, operation):
        if not isinstance(operation, ops.GateOperation):
            raise ValueError('Unsupported operation: {!r}'.format(operation))

        self.validate_gate(operation.gate)

        for q in operation.qubits:
            if not isinstance(q, LineQubit):
                raise ValueError('Unsupported qubit type: {!r}'.format(q))
            if q not in self.qubits:
                raise ValueError('Qubit not on device: {!r}'.format(q))

    def _check_if_XXPow_operation_interacts_with_any(
            self,
            XXPow_op: ops.GateOperation,
            others: Iterable[ops.GateOperation]) -> bool:
        return any(self._check_if_XXPow_operation_interacts(XXPow_op, op)
                   for op in others)

    def _check_if_XXPow_operation_interacts(
            self,
            XXPow_op: ops.GateOperation,
            other_op: ops.GateOperation) -> bool:
        if isinstance(other_op.gate, (ops.XPowGate,
                                      ops.YPowGate,
                                      ops.PhasedXPowGate,
                                      ops.MeasurementGate,
                                      ops.ZPowGate)):
            return False

        return any(q == p
                   for q in XXPow_op.qubits
                   for p in other_op.qubits)

    def validate_scheduled_operation(self, schedule, scheduled_operation):
        self.validate_operation(scheduled_operation.operation)

        if isinstance(scheduled_operation.operation.gate, ops.XXPowGate):
            for other in schedule.operations_happening_at_same_time_as(
                    scheduled_operation):
                if self._check_if_XXPow_operation_interacts(
                        cast(ops.GateOperation, scheduled_operation.operation),
                        cast(ops.GateOperation, other.operation)):
                    raise ValueError(
                        'Simultaneous two-qubit '
                        'operations on same qubit: {} vs {}.'.format(
                            scheduled_operation, other))

    def validate_circuit(self, circuit: circuits.Circuit):
        super().validate_circuit(circuit)
        _verify_unique_measurement_keys(circuit.all_operations())

    def can_add_operation_into_moment(self,
                                      operation: ops.Operation,
                                      moment: ops.Moment) -> bool:

        if not super().can_add_operation_into_moment(operation, moment):
            return False
        if ops.op_gate_of_type(operation, ops.XXPowGate):
            return not self._check_if_XXPow_operation_interacts_with_any(
                cast(ops.GateOperation, operation),
                cast(Iterable[ops.GateOperation], moment.operations))
        return True

    def validate_schedule(self, schedule):
        _verify_unique_measurement_keys(
            s.operation for s in schedule.scheduled_operations)
        for scheduled_operation in schedule.scheduled_operations:
            self.validate_scheduled_operation(schedule, scheduled_operation)

    def at(self, position: int) -> Optional[LineQubit]:
        """Returns the qubit at the given position, if there is one, else None.
        """
        q = LineQubit(position)
        return q if q in self.qubits else None

    def neighbors_of(self, qubit: LineQubit):
        """Returns the qubits that the given qubit can interact with."""
        possibles = [
            LineQubit(qubit.x + 1),
            LineQubit(qubit.x - 1),
        ]
        return [e for e in possibles if e in self.qubits]

    def __repr__(self):
        return ('IonDevice(measurement_duration={!r}, '
                'twoq_gates_duration={!r}, '
                'oneq_gates_duration={!r} '
                'qubits={!r})').format(self._measurement_duration,
                                       self._twoq_gates_duration,
                                       self._oneq_gates_duration,
                                       sorted(self.qubits))

    def __str__(self):
        diagram = circuits.TextDiagramDrawer()

        for q in self.qubits:
            diagram.write(q.x, 0, str(q))
            for q2 in self.neighbors_of(q):
                diagram.grid_line(q.x, 0, q2.x, 0)

        return diagram.render(
            horizontal_spacing=3,
            vertical_spacing=2,
            use_unicode_characters=True)

    def _value_equality_values_(self):
        return (self._measurement_duration,
                self._twoq_gates_duration,
                self._oneq_gates_duration,
                self.qubits)


def _verify_unique_measurement_keys(operations: Iterable[ops.Operation]):
    seen = set()  # type: Set[str]
    for op in operations:
        meas = ops.op_gate_of_type(op, ops.MeasurementGate)
        if meas:
            key = protocols.measurement_key(meas)
            if key in seen:
                raise ValueError('Measurement key {} repeated'.format(key))
            seen.add(key)