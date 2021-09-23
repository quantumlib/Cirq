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

from typing import Any, FrozenSet, Iterable, Optional, Set, TYPE_CHECKING

from cirq import circuits, value, devices, ops, protocols
from cirq.ion import convert_to_ion_gates

if TYPE_CHECKING:
    import cirq


def get_ion_gateset() -> ops.Gateset:
    return ops.Gateset(
        ops.XXPowGate,
        ops.MeasurementGate,
        ops.XPowGate,
        ops.YPowGate,
        ops.ZPowGate,
        ops.PhasedXPowGate,
        unroll_circuit_op=False,
        accept_global_phase=False,
    )


@value.value_equality
class IonDevice(devices.Device):
    """A device with qubits placed on a line.

    Qubits have all-to-all connectivity.
    """

    def __init__(
        self,
        measurement_duration: 'cirq.DURATION_LIKE',
        twoq_gates_duration: 'cirq.DURATION_LIKE',
        oneq_gates_duration: 'cirq.DURATION_LIKE',
        qubits: Iterable[devices.LineQubit],
    ) -> None:
        """Initializes the description of an ion trap device.

        Args:
            measurement_duration: The maximum duration of a measurement.
            twoq_gates_duration: The maximum duration of a two qubit operation.
            oneq_gates_duration: The maximum duration of a single qubit
            operation.
            qubits: Qubits on the device, identified by their x, y location.
        """
        self._measurement_duration = value.Duration(measurement_duration)
        self._twoq_gates_duration = value.Duration(twoq_gates_duration)
        self._oneq_gates_duration = value.Duration(oneq_gates_duration)
        self.qubits = frozenset(qubits)
        self.gateset = get_ion_gateset()

    def qubit_set(self) -> FrozenSet['cirq.LineQubit']:
        return self.qubits

    def qid_pairs(self) -> FrozenSet['cirq.SymmetricalQidPair']:
        """Qubits have all-to-all connectivity, so returns all pairs.

        Returns:
            All qubit pairs on the device.
        """
        qs = self.qubits
        return frozenset([devices.SymmetricalQidPair(q, q2) for q in qs for q2 in qs if q < q2])

    def decompose_operation(self, operation: ops.Operation) -> ops.OP_TREE:
        return convert_to_ion_gates.ConvertToIonGates().convert_one(operation)

    def decompose_circuit(self, circuit: circuits.Circuit) -> circuits.Circuit:
        return convert_to_ion_gates.ConvertToIonGates().convert_circuit(circuit)

    def duration_of(self, operation):
        if isinstance(operation.gate, ops.XXPowGate):
            return self._twoq_gates_duration
        if isinstance(
            operation.gate, (ops.XPowGate, ops.YPowGate, ops.ZPowGate, ops.PhasedXPowGate)
        ):
            return self._oneq_gates_duration
        if isinstance(operation.gate, ops.MeasurementGate):
            return self._measurement_duration
        raise ValueError(f'Unsupported gate type: {operation!r}')

    def validate_gate(self, gate: ops.Gate):
        if gate not in self.gateset:
            raise ValueError(f'Unsupported gate type: {gate!r}')

    def validate_operation(self, operation):
        if not isinstance(operation, ops.GateOperation):
            raise ValueError(f'Unsupported operation: {operation!r}')

        self.validate_gate(operation.gate)

        for q in operation.qubits:
            if not isinstance(q, devices.LineQubit):
                raise ValueError(f'Unsupported qubit type: {q!r}')
            if q not in self.qubits:
                raise ValueError(f'Qubit not on device: {q!r}')

    def validate_circuit(self, circuit: circuits.AbstractCircuit):
        super().validate_circuit(circuit)
        _verify_unique_measurement_keys(circuit.all_operations())

    def at(self, position: int) -> Optional[devices.LineQubit]:
        """Returns the qubit at the given position, if there is one, else None."""
        q = devices.LineQubit(position)
        return q if q in self.qubits else None

    def neighbors_of(self, qubit: devices.LineQubit) -> Iterable[devices.LineQubit]:
        """Returns the qubits that the given qubit can interact with."""
        possibles = [
            devices.LineQubit(qubit.x + 1),
            devices.LineQubit(qubit.x - 1),
        ]
        return [e for e in possibles if e in self.qubits]

    def __repr__(self) -> str:
        return (
            f'IonDevice(measurement_duration={self._measurement_duration!r}, '
            f'twoq_gates_duration={self._twoq_gates_duration!r}, '
            f'oneq_gates_duration={self._oneq_gates_duration!r} '
            f'qubits={sorted(self.qubits)!r})'
        )

    def __str__(self) -> str:
        diagram = circuits.TextDiagramDrawer()

        for q in self.qubits:
            diagram.write(q.x, 0, str(q))
            for q2 in self.neighbors_of(q):
                diagram.grid_line(q.x, 0, q2.x, 0)

        return diagram.render(horizontal_spacing=3, vertical_spacing=2, use_unicode_characters=True)

    def _value_equality_values_(self) -> Any:
        return (
            self._measurement_duration,
            self._twoq_gates_duration,
            self._oneq_gates_duration,
            self.qubits,
        )


def _verify_unique_measurement_keys(operations: Iterable[ops.Operation]):
    seen: Set[str] = set()
    for op in operations:
        if isinstance(op.gate, ops.MeasurementGate):
            meas = op.gate
            key = protocols.measurement_key_name(meas)
            if key in seen:
                raise ValueError(f'Measurement key {key} repeated')
            seen.add(key)
