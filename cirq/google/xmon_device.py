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

from typing import Iterable, cast, List

import numpy as np

from cirq import ops
from cirq.devices import Device
from cirq.google import xmon_gates
from cirq.google.xmon_gate_extensions import xmon_gate_ext
from cirq.devices.grid_qubit import GridQubit
from cirq.value import Duration

from cirq.circuits import TextDiagramDrawer


class XmonDevice(Device):
    """A device with qubits placed in a grid. Neighboring qubits can interact.
    """

    def __init__(self,
                 measurement_duration: Duration,
                 exp_w_duration: Duration,
                 exp_11_duration: Duration,
                 qubits: Iterable[GridQubit]) -> None:
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

    def __getitem__(self, item):
        r, c = item

        if isinstance(r, int) and isinstance(c, int):
            q = GridQubit(r, c)
            if q not in self.qubits:
                raise IndexError('{} not on device'.format(q))
            return q

        if isinstance(r, int) and isinstance(c, slice):
            available_cols = _bound_slice(
                c,
                (q.col for q in self.qubits if q.row == r))
            return [GridQubit(r, col)
                    for col in available_cols
                    if GridQubit(r, col) in self.qubits]

        if isinstance(r, slice) and isinstance(c, int):
            available_rows = _bound_slice(
                r,
                (q.row for q in self.qubits if q.col == c))
            return [GridQubit(row, c)
                    for row in available_rows
                    if GridQubit(row, c) in self.qubits]

        if isinstance(r, slice) and isinstance(c, slice):
            available_rows = _bound_slice(r, (q.row for q in self.qubits))
            available_cols = _bound_slice(c, (q.col for q in self.qubits))
            rows = [
                [
                    GridQubit(row, col)
                    for col in available_cols
                    if GridQubit(row, col) in self.qubits
                ]
                for row in available_rows
            ]

            # Make sure it's a grid.
            if len(set(len(row) for row in rows)) > 1:
                raise IndexError('Jagged boundary.')

            result = np.array(rows)
            row_step = 1 if r.step is None else r.step
            col_step = 1 if c.step is None else c.step

            # Make sure it's a grid with the expected step.
            # E.g. we should fail if there's a hole in the middle of the query.
            if result.shape[0] and result.shape[1]:
                q = result[0, 0]
                for i in range(result.shape[0]):
                    for j in range(result.shape[1]):
                        if result[i, j] != GridQubit(q.row + row_step * i,
                                                     q.col + col_step * j):
                            raise IndexError('Jagged boundary.')

            return result

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
        if isinstance(operation, ops.GateOperation):
            g = xmon_gate_ext.try_cast(xmon_gates.XmonGate, operation.gate)
            if isinstance(g, xmon_gates.Exp11Gate):
                return self._exp_z_duration
            if isinstance(g, xmon_gates.ExpWGate):
                return self._exp_w_duration
            if isinstance(g, xmon_gates.XmonMeasurementGate):
                return self._measurement_duration
            if isinstance(g, xmon_gates.ExpZGate):
                # Z gates are performed in the control software.
                return Duration()
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
            raise ValueError('Unsupported gate type: {!r}'.format(gate))

    def validate_operation(self, operation):
        if not isinstance(operation, ops.GateOperation):
            raise ValueError('Unsupported operation: {!r}'.format(operation))

        self.validate_gate(operation.gate)

        for q in operation.qubits:
            if not isinstance(q, GridQubit):
                raise ValueError('Unsupported qubit type: {!r}'.format(q))
            if q not in self.qubits:
                raise ValueError('Qubit not on device: {!r}'.format(q))

        if (len(operation.qubits) == 2
                and not isinstance(operation.gate,
                                   xmon_gates.XmonMeasurementGate)):
            p, q = operation.qubits
            if not cast(GridQubit, p).is_adjacent(q):
                raise ValueError(
                    'Non-local interaction: {!r}.'.format(operation))

    def check_if_exp11_operation_interacts(self,
                                           exp11_op: ops.GateOperation,
                                           other_op: ops.GateOperation) -> bool:
        if isinstance(other_op.gate, xmon_gates.ExpZGate):
            return False
        # Adjacent ExpW operations may be doable.
        # For now we will play it conservatively.

        return any(cast(GridQubit, q).is_adjacent(cast(GridQubit, p))
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
        measurement_keys = set()
        for moment in circuit.moments:
            for operation in moment.operations:
                self.validate_operation(operation)
                self.verify_new_measurement_key(operation, measurement_keys)

    def validate_schedule(self, schedule):
        measurement_keys = set()
        for scheduled_operation in schedule.scheduled_operations:
            self.validate_scheduled_operation(schedule, scheduled_operation)
            self.verify_new_measurement_key(
                scheduled_operation.operation, measurement_keys)

    def verify_new_measurement_key(self, operation, previous_keys):
        if isinstance(operation.gate, xmon_gates.XmonMeasurementGate):
            if operation.gate.key in previous_keys:
                raise ValueError('Measurement key {} repeated'.format(
                    operation.gate.key))
            else:
                previous_keys.add(operation.gate.key)

    def __str__(self):
        diagram = TextDiagramDrawer()

        for q in self.qubits:
            diagram.write(q.col, q.row, str(q))
            for q2 in self.neighbors_of(q):
                diagram.grid_line(q.col, q.row, q2.col, q2.row)

        return diagram.render(
            horizontal_spacing=3,
            vertical_spacing=2,
            use_unicode_characters=True)

    def __eq__(self, other):
        if not isinstance(other, (XmonDevice, type(self))):
            return NotImplemented
        return (self._measurement_duration == other._measurement_duration and
                self._exp_w_duration == other._exp_w_duration and
                self._exp_z_duration == other._exp_z_duration and
                self.qubits == other.qubits)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((XmonDevice, self._measurement_duration,
                     self._exp_w_duration, self._exp_z_duration, self.qubits))


def _bound_slice(s: slice, available: Iterable[int]) -> List[int]:
    step = 1 if s.step is None else s.step

    if s.start is not None and s.stop is not None:
        return list(range(s.start, s.stop, step))

    min_index = None
    max_index = None
    for item in available:
        if min_index is None:
            min_index = item
            max_index = item
        else:
            min_index = min(min_index, item)
            max_index = max(max_index, item)

    if min_index is None:
        raise IndexError('Sliced outside of valid area.')

    if step >= 0:
        if s.start is not None:
            min_index = s.start
        if s.stop is not None:
            max_index = s.stop - 1
        return list(range(min_index, max_index + 1, step))

    if s.start is not None:
        max_index = s.start
    if s.stop is not None:
        min_index = s.stop - 1
    return list(range(max_index, min_index - 1, step))
