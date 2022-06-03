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

from typing import Any, cast, Iterable, List, Optional, Set, TYPE_CHECKING, FrozenSet

import cirq
from cirq import _compat

if TYPE_CHECKING:
    import cirq


@cirq.value_equality
class _XmonDeviceBase(cirq.Device):
    """A device with qubits placed in a grid. Neighboring qubits can interact."""

    def __init__(
        self,
        measurement_duration: cirq.DURATION_LIKE,
        exp_w_duration: cirq.DURATION_LIKE,
        exp_11_duration: cirq.DURATION_LIKE,
        qubits: Iterable[cirq.GridQubit],
    ) -> None:
        """Initializes the description of an xmon device.

        Args:
            measurement_duration: The maximum duration of a measurement.
            exp_w_duration: The maximum duration of an ExpW operation.
            exp_11_duration: The maximum duration of an ExpZ operation.
            qubits: Qubits on the device, identified by their x, y location.
        """
        self._measurement_duration = cirq.Duration(measurement_duration)
        self._exp_w_duration = cirq.Duration(exp_w_duration)
        self._exp_z_duration = cirq.Duration(exp_11_duration)
        self.qubits = frozenset(qubits)
        self._metadata = cirq.GridDeviceMetadata(
            [(q0, q1) for q0 in self.qubits for q1 in self.qubits if q0.is_adjacent(q1)],
            cirq.Gateset(
                cirq.CZPowGate,
                cirq.XPowGate,
                cirq.YPowGate,
                cirq.PhasedXPowGate,
                cirq.PhasedXZGate,
                cirq.MeasurementGate,
                cirq.ZPowGate,
                cirq.GlobalPhaseGate,
            ),
            None,
        )

    @property
    def metadata(self) -> cirq.GridDeviceMetadata:
        """Return the metadata for this device"""
        return self._metadata

    @_compat.deprecated(fix='Use metadata.qubit_set if applicable.', deadline='v0.15')
    def qubit_set(self) -> FrozenSet[cirq.GridQubit]:
        return self.qubits

    def neighbors_of(self, qubit: cirq.GridQubit):
        """Returns the qubits that the given qubit can interact with."""
        possibles = [
            cirq.GridQubit(qubit.row + 1, qubit.col),
            cirq.GridQubit(qubit.row - 1, qubit.col),
            cirq.GridQubit(qubit.row, qubit.col + 1),
            cirq.GridQubit(qubit.row, qubit.col - 1),
        ]
        return [e for e in possibles if e in self.qubits]

    def duration_of(self, operation):
        if isinstance(operation.gate, cirq.CZPowGate):
            return self._exp_z_duration
        if isinstance(operation.gate, cirq.MeasurementGate):
            return self._measurement_duration
        if isinstance(operation.gate, (cirq.XPowGate, cirq.YPowGate, cirq.PhasedXPowGate)):
            return self._exp_w_duration
        if isinstance(operation.gate, cirq.ZPowGate):
            # Z gates are performed in the control software.
            return cirq.Duration()
        raise ValueError(f'Unsupported gate type: {operation!r}')

    @classmethod
    def is_supported_gate(cls, gate: cirq.Gate):
        """Returns true if the gate is allowed."""
        return isinstance(
            gate,
            (
                cirq.CZPowGate,
                cirq.XPowGate,
                cirq.YPowGate,
                cirq.PhasedXPowGate,
                cirq.PhasedXZGate,
                cirq.MeasurementGate,
                cirq.ZPowGate,
            ),
        )

    def validate_gate(self, gate: cirq.Gate):
        """Raises an error if the given gate isn't allowed.

        Raises:
            ValueError: Unsupported gate.
        """
        if not self.is_supported_gate(gate):
            raise ValueError(f'Unsupported gate type: {gate!r}')

    def validate_operation(self, operation: cirq.Operation):
        if operation.gate is None:
            raise ValueError(f'Unsupported operation: {operation!r}')

        self.validate_gate(operation.gate)

        for q in operation.qubits:
            if not isinstance(q, cirq.GridQubit):
                raise ValueError(f'Unsupported qubit type: {q!r}')
            if q not in self.qubits:
                raise ValueError(f'Qubit not on device: {q!r}')

        if len(operation.qubits) == 2 and not isinstance(operation.gate, cirq.MeasurementGate):
            p, q = operation.qubits
            if not cast(cirq.GridQubit, p).is_adjacent(q):
                raise ValueError(f'Non-local interaction: {operation!r}.')

    def _check_if_exp11_operation_interacts_with_any(
        self, exp11_op: cirq.GateOperation, others: Iterable[cirq.GateOperation]
    ) -> bool:
        return any(self._check_if_exp11_operation_interacts(exp11_op, op) for op in others)

    def _check_if_exp11_operation_interacts(
        self, exp11_op: cirq.GateOperation, other_op: cirq.GateOperation
    ) -> bool:
        if isinstance(
            other_op.gate,
            (
                cirq.XPowGate,
                cirq.YPowGate,
                cirq.PhasedXPowGate,
                cirq.MeasurementGate,
                cirq.ZPowGate,
            ),
        ):
            return False

        return any(
            cast(cirq.GridQubit, q).is_adjacent(cast(cirq.GridQubit, p))
            for q in exp11_op.qubits
            for p in other_op.qubits
        )

    def validate_circuit(self, circuit: cirq.AbstractCircuit):
        super().validate_circuit(circuit)
        _verify_unique_measurement_keys(circuit.all_operations())

    def validate_moment(self, moment: cirq.Moment):
        super().validate_moment(moment)
        for op in moment.operations:
            if isinstance(op.gate, cirq.CZPowGate):
                for other in moment.operations:
                    if other is not op and self._check_if_exp11_operation_interacts(
                        cast(cirq.GateOperation, op), cast(cirq.GateOperation, other)
                    ):
                        raise ValueError(f'Adjacent Exp11 operations: {moment}.')

    def at(self, row: int, col: int) -> Optional[cirq.GridQubit]:
        """Returns the qubit at the given position, if there is one, else None."""
        q = cirq.GridQubit(row, col)
        return q if q in self.qubits else None

    def row(self, row: int) -> List[cirq.GridQubit]:
        """Returns the qubits in the given row, in ascending order."""
        return sorted(q for q in self.qubits if q.row == row)

    def col(self, col: int) -> List[cirq.GridQubit]:
        """Returns the qubits in the given column, in ascending order."""
        return sorted(q for q in self.qubits if q.col == col)

    def __repr__(self) -> str:
        return (
            'XmonDevice('
            f'measurement_duration={self._measurement_duration!r}, '
            f'exp_w_duration={self._exp_w_duration!r}, '
            f'exp_11_duration={self._exp_z_duration!r} '
            f'qubits={sorted(self.qubits)!r})'
        )

    def __str__(self) -> str:
        diagram = cirq.TextDiagramDrawer()

        for q in self.qubits:
            diagram.write(q.col, q.row, str(q))
            for q2 in self.neighbors_of(q):
                diagram.grid_line(q.col, q.row, q2.col, q2.row)

        return diagram.render(horizontal_spacing=3, vertical_spacing=2, use_unicode_characters=True)

    def _repr_pretty_(self, p: Any, cycle: bool):
        p.text("cirq_google.XmonDevice(...)" if cycle else self.__str__())

    def _value_equality_values_(self) -> Any:
        return (self._measurement_duration, self._exp_w_duration, self._exp_z_duration, self.qubits)


def _verify_unique_measurement_keys(operations: Iterable[cirq.Operation]):
    seen: Set[str] = set()
    for op in operations:
        if cirq.is_measurement(op):
            key = cirq.measurement_key_name(op)
            if key in seen:
                raise ValueError(f'Measurement key {key} repeated')
            seen.add(key)


@_compat.deprecated_class(deadline='v0.15', fix='XmonDevice will no longer be supported.')
class XmonDevice(_XmonDeviceBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
