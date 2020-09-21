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

from typing import (Any, cast, Iterable, List, Optional, Set, TYPE_CHECKING,
                    FrozenSet)

from cirq import circuits, devices, ops, protocols, value
from cirq.google.optimizers import convert_to_xmon_gates
from cirq.devices.grid_qubit import GridQubit

if TYPE_CHECKING:
    import cirq


@value.value_equality
class XmonDevice(devices.Device):
    """A device with qubits placed in a grid. Neighboring qubits can interact.
    """

    def __init__(self, measurement_duration: 'cirq.DURATION_LIKE',
                 exp_w_duration: 'cirq.DURATION_LIKE',
                 exp_11_duration: 'cirq.DURATION_LIKE',
                 qubits: Iterable[GridQubit]) -> None:
        """Initializes the description of an xmon device.

        Args:
            measurement_duration: The maximum duration of a measurement.
            exp_w_duration: The maximum duration of an ExpW operation.
            exp_11_duration: The maximum duration of an ExpZ operation.
            qubits: Qubits on the device, identified by their x, y location.
        """
        self._measurement_duration = value.Duration(measurement_duration)
        self._exp_w_duration = value.Duration(exp_w_duration)
        self._exp_z_duration = value.Duration(exp_11_duration)
        self.qubits = frozenset(qubits)

    def qubit_set(self) -> FrozenSet['cirq.GridQubit']:
        return self.qubits

    def decompose_operation(self,
                            operation: 'cirq.Operation') -> 'cirq.OP_TREE':
        return convert_to_xmon_gates.ConvertToXmonGates().convert(operation)

    def neighbors_of(self, qubit: GridQubit):
        """Returns the qubits that the given qubit can interact with."""
        possibles = [
            GridQubit(qubit.row + 1, qubit.col),
            GridQubit(qubit.row - 1, qubit.col),
            GridQubit(qubit.row, qubit.col + 1),
            GridQubit(qubit.row, qubit.col - 1),
        ]
        return [e for e in possibles if e in self.qubits]

    def duration_of(self, operation):
        if isinstance(operation.gate, ops.CZPowGate):
            return self._exp_z_duration
        if isinstance(operation.gate, ops.MeasurementGate):
            return self._measurement_duration
        if isinstance(operation.gate,
                      (ops.XPowGate, ops.YPowGate, ops.PhasedXPowGate)):
            return self._exp_w_duration
        if isinstance(operation.gate, ops.ZPowGate):
            # Z gates are performed in the control software.
            return value.Duration()
        raise ValueError('Unsupported gate type: {!r}'.format(operation))

    @classmethod
    def is_supported_gate(cls, gate: 'cirq.Gate'):
        """Returns true if the gate is allowed.
        """
        return isinstance(
            gate, (ops.CZPowGate, ops.XPowGate, ops.YPowGate,
                   ops.PhasedXPowGate, ops.MeasurementGate, ops.ZPowGate))

    def validate_gate(self, gate: 'cirq.Gate'):
        """Raises an error if the given gate isn't allowed.

        Raises:
            ValueError: Unsupported gate.
        """
        if not self.is_supported_gate(gate):
            raise ValueError('Unsupported gate type: {!r}'.format(gate))

    def validate_operation(self, operation: 'cirq.Operation'):
        if not isinstance(operation, ops.GateOperation):
            raise ValueError('Unsupported operation: {!r}'.format(operation))

        self.validate_gate(operation.gate)

        for q in operation.qubits:
            if not isinstance(q, GridQubit):
                raise ValueError('Unsupported qubit type: {!r}'.format(q))
            if q not in self.qubits:
                raise ValueError('Qubit not on device: {!r}'.format(q))

        if (len(operation.qubits) == 2 and
                not isinstance(operation.gate, ops.MeasurementGate)):
            p, q = operation.qubits
            if not cast(GridQubit, p).is_adjacent(q):
                raise ValueError(
                    'Non-local interaction: {!r}.'.format(operation))

    def _check_if_exp11_operation_interacts_with_any(
            self, exp11_op: 'cirq.GateOperation',
            others: Iterable['cirq.GateOperation']) -> bool:
        return any(
            self._check_if_exp11_operation_interacts(exp11_op, op)
            for op in others)

    def _check_if_exp11_operation_interacts(self,
                                            exp11_op: 'cirq.GateOperation',
                                            other_op: 'cirq.GateOperation'
                                           ) -> bool:
        if isinstance(other_op.gate,
                      (ops.XPowGate, ops.YPowGate, ops.PhasedXPowGate,
                       ops.MeasurementGate, ops.ZPowGate)):
            return False

        return any(
            cast(GridQubit, q).is_adjacent(cast(GridQubit, p))
            for q in exp11_op.qubits
            for p in other_op.qubits)

    def validate_circuit(self, circuit: 'cirq.Circuit'):
        super().validate_circuit(circuit)
        _verify_unique_measurement_keys(circuit.all_operations())

    def validate_moment(self, moment: 'cirq.Moment'):
        super().validate_moment(moment)
        for op in moment.operations:
            if isinstance(op.gate, ops.CZPowGate):
                for other in moment.operations:
                    if (other is not op and
                            self._check_if_exp11_operation_interacts(
                                cast(ops.GateOperation, op),
                                cast(ops.GateOperation, other))):
                        raise ValueError(
                            'Adjacent Exp11 operations: {}.'.format(moment))

    def can_add_operation_into_moment(self, operation: 'cirq.Operation',
                                      moment: 'cirq.Moment') -> bool:
        self.validate_moment(moment)

        if not super().can_add_operation_into_moment(operation, moment):
            return False
        if isinstance(operation.gate, ops.CZPowGate):
            return not self._check_if_exp11_operation_interacts_with_any(
                cast(ops.GateOperation, operation),
                cast(Iterable['cirq.GateOperation'], moment.operations))
        return True

    def at(self, row: int, col: int) -> Optional[GridQubit]:
        """Returns the qubit at the given position, if there is one, else None.
        """
        q = GridQubit(row, col)
        return q if q in self.qubits else None

    def row(self, row: int) -> List[GridQubit]:
        """Returns the qubits in the given row, in ascending order."""
        return sorted(q for q in self.qubits if q.row == row)

    def col(self, col: int) -> List[GridQubit]:
        """Returns the qubits in the given column, in ascending order."""
        return sorted(q for q in self.qubits if q.col == col)

    def __repr__(self) -> str:
        return ('XmonDevice('
                f'measurement_duration={self._measurement_duration!r}, '
                f'exp_w_duration={self._exp_w_duration!r}, '
                f'exp_11_duration={self._exp_z_duration!r} '
                f'qubits={sorted(self.qubits)!r})')

    def __str__(self) -> str:
        diagram = circuits.TextDiagramDrawer()

        for q in self.qubits:
            diagram.write(q.col, q.row, str(q))
            for q2 in self.neighbors_of(q):
                diagram.grid_line(q.col, q.row, q2.col, q2.row)

        return diagram.render(horizontal_spacing=3,
                              vertical_spacing=2,
                              use_unicode_characters=True)

    def _value_equality_values_(self) -> Any:
        return (self._measurement_duration, self._exp_w_duration,
                self._exp_z_duration, self.qubits)


def _verify_unique_measurement_keys(operations: Iterable['cirq.Operation']):
    seen: Set[str] = set()
    for op in operations:
        if protocols.is_measurement(op):
            key = protocols.measurement_key(op)
            if key in seen:
                raise ValueError('Measurement key {} repeated'.format(key))
            seen.add(key)
