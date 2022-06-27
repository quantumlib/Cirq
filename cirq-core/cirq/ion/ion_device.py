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

from typing import Any, Iterable, List, Optional, Set, TYPE_CHECKING
import numpy as np
import networkx as nx
from cirq import _compat, circuits, value, devices, ops, protocols, transformers
from cirq.protocols.decompose_protocol import DecomposeResult

if TYPE_CHECKING:
    import cirq


class _IonTargetGateset(transformers.TwoQubitCompilationTargetGateset):
    def __init__(self):
        super().__init__(
            ops.XXPowGate,
            ops.MeasurementGate,
            ops.XPowGate,
            ops.YPowGate,
            ops.ZPowGate,
            ops.PhasedXPowGate,
            unroll_circuit_op=False,
        )

    def _decompose_single_qubit_operation(self, op: 'cirq.Operation', _: int) -> DecomposeResult:
        if isinstance(op.gate, ops.HPowGate) and op.gate.exponent == 1:
            return [ops.rx(np.pi).on(op.qubits[0]), ops.ry(-1 * np.pi / 2).on(op.qubits[0])]
        if protocols.has_unitary(op):
            gates = transformers.single_qubit_matrix_to_phased_x_z(protocols.unitary(op))
            return [g.on(op.qubits[0]) for g in gates]
        return NotImplemented

    def _decompose_two_qubit_operation(self, op: 'cirq.Operation', _) -> DecomposeResult:
        if protocols.has_unitary(op):
            return transformers.two_qubit_matrix_to_ion_operations(
                op.qubits[0], op.qubits[1], protocols.unitary(op)
            )
        return NotImplemented

    @property
    def postprocess_transformers(self) -> List['cirq.TRANSFORMER']:
        """List of transformers which should be run after decomposing individual operations."""
        return [transformers.drop_negligible_operations, transformers.drop_empty_moments]


@value.value_equality
class _IonDeviceImpl(devices.Device):
    """Shared implementation of `cirq.IonDevice` (deprecated) and `cirq_aqt.AQTDevice`.

    This class will be removed once `cirq.IonDevice` is deprecated and removed. The implementation
    will be moved to `cirq_aqt.AQTDevice`.
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
            qubits: Qubits on the device, identified by their x location.

        Raises:
            TypeError: If not all the qubits supplied are `cirq.LineQubit`s.
        """
        self._measurement_duration = value.Duration(measurement_duration)
        self._twoq_gates_duration = value.Duration(twoq_gates_duration)
        self._oneq_gates_duration = value.Duration(oneq_gates_duration)
        if not all(isinstance(qubit, devices.LineQubit) for qubit in qubits):
            raise TypeError(
                "All qubits were not of type cirq.LineQubit, instead were "
                f"{set(type(qubit) for qubit in qubits)}"
            )
        self.qubits = frozenset(qubits)
        self.gateset = _IonTargetGateset()

        graph = nx.Graph()
        graph.add_edges_from([(a, b) for a in qubits for b in qubits if a != b], directed=False)
        self._metadata = devices.DeviceMetadata(self.qubits, graph)

    @property
    def metadata(self) -> devices.DeviceMetadata:
        return self._metadata

    def decompose_circuit(self, circuit: circuits.Circuit) -> circuits.Circuit:
        return transformers.optimize_for_target_gateset(circuit, gateset=self.gateset)

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
        possibles = [devices.LineQubit(qubit.x + 1), devices.LineQubit(qubit.x - 1)]
        return [e for e in possibles if e in self.qubits]

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


@_compat.deprecated_class(deadline='v0.16', fix='Use cirq_aqt.aqt_device.AQTDevice.')
class IonDevice(_IonDeviceImpl):
    """A device with qubits placed on a line.

    Qubits have all-to-all connectivity.
    """

    def __repr__(self) -> str:
        return (
            f'IonDevice(measurement_duration={self._measurement_duration!r}, '
            f'twoq_gates_duration={self._twoq_gates_duration!r}, '
            f'oneq_gates_duration={self._oneq_gates_duration!r} '
            f'qubits={sorted(self.qubits)!r})'
        )

    def _repr_pretty_(self, p: Any, cycle: bool):
        """iPython (Jupyter) pretty print."""
        p.text("IonDevice(...)" if cycle else self.__str__())
