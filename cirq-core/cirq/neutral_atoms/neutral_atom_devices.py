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

import itertools
import collections
from typing import Any, Iterable, cast, DefaultDict, TYPE_CHECKING, FrozenSet
from numpy import sqrt
from cirq import _compat, devices, ops, circuits, value
from cirq.devices.grid_qubit import GridQubit
from cirq.ops import raw_types
from cirq.value import Duration
from cirq.neutral_atoms.neutral_atom_gateset import NeutralAtomGateset

if TYPE_CHECKING:
    import cirq


def _subgate_if_parallel_gate(gate: 'cirq.Gate') -> 'cirq.Gate':
    """Returns gate.sub_gate if gate is a ParallelGate, else returns gate"""
    return gate.sub_gate if isinstance(gate, ops.ParallelGate) else gate


@value.value_equality
class NeutralAtomDevice(devices.Device):
    """A device with qubits placed on a grid."""

    def __init__(
        self,
        measurement_duration: 'cirq.DURATION_LIKE',
        gate_duration: 'cirq.DURATION_LIKE',
        control_radius: float,
        max_parallel_z: int,
        max_parallel_xy: int,
        max_parallel_c: int,
        qubits: Iterable[GridQubit],
    ) -> None:
        """Initializes the description of the AQuA device.

        Args:
            measurement_duration: the maximum duration of a measurement.
            gate_duration: the maximum duration of a gate
            control_radius: the maximum distance between qubits for a controlled
                gate. Distance is measured in units of the indices passed into
                the GridQubit constructor.
            max_parallel_z: The maximum number of qubits that can be acted on
                in parallel by a Z gate
            max_parallel_xy: The maximum number of qubits that can be acted on
                in parallel by a local XY gate
            max_parallel_c: the maximum number of qubits that can be acted on in
                parallel by a controlled gate. Must be less than or equal to the
                lesser of max_parallel_z and max_parallel_xy
            qubits: Qubits on the device, identified by their x, y location.
                Must be of type GridQubit

        Raises:
            ValueError: if the wrong qubit type is provided or if invalid
                parallel parameters are provided
        """
        self._measurement_duration = Duration(measurement_duration)
        self._gate_duration = Duration(gate_duration)
        self._control_radius = control_radius
        self._max_parallel_z = max_parallel_z
        self._max_parallel_xy = max_parallel_xy
        if max_parallel_c > min(max_parallel_z, max_parallel_xy):
            raise ValueError(
                "max_parallel_c must be less than or equal to the"
                "min of max_parallel_z and max_parallel_xy"
            )
        self._max_parallel_c = max_parallel_c
        self.xy_gateset_all_allowed = ops.Gateset(
            ops.ParallelGateFamily(ops.XPowGate),
            ops.ParallelGateFamily(ops.YPowGate),
            ops.ParallelGateFamily(ops.PhasedXPowGate),
            unroll_circuit_op=False,
        )
        self.controlled_gateset = ops.Gateset(
            ops.AnyIntegerPowerGateFamily(ops.CNotPowGate),
            ops.AnyIntegerPowerGateFamily(ops.CCNotPowGate),
            ops.AnyIntegerPowerGateFamily(ops.CZPowGate),
            ops.AnyIntegerPowerGateFamily(ops.CCZPowGate),
            unroll_circuit_op=False,
        )
        self.gateset = NeutralAtomGateset(max_parallel_z, max_parallel_xy)
        for q in qubits:
            if not isinstance(q, GridQubit):
                raise ValueError(f'Unsupported qubit type: {q!r}')
        self.qubits = frozenset(qubits)

        self._metadata = devices.GridDeviceMetadata(
            [(a, b) for a in self.qubits for b in self.qubits if a.is_adjacent(b)], self.gateset
        )

    @property
    def metadata(self) -> devices.GridDeviceMetadata:
        return self._metadata

    @_compat.deprecated(fix='Use metadata.qubit_set if applicable.', deadline='v0.15')
    def qubit_set(self) -> FrozenSet['cirq.GridQubit']:
        return self.qubits

    def qubit_list(self):
        return [qubit for qubit in self.qubits]

    def duration_of(self, operation: ops.Operation):
        """Provides the duration of the given operation on this device.

        Args:
            operation: the operation to get the duration of

        Returns:
            The duration of the given operation on this device

        Raises:
            ValueError: If the operation provided doesn't correspond to a native
                gate
        """
        self.validate_operation(operation)
        if isinstance(operation, ops.GateOperation):
            if isinstance(operation.gate, ops.MeasurementGate):
                return self._measurement_duration
        return self._gate_duration

    def validate_gate(self, gate: ops.Gate):
        """Raises an error if the provided gate isn't part of the native gate set.

        Args:
            gate: the gate to validate

        Raises:
            ValueError: If the given gate is not part of the native gate set.
        """
        if gate not in self.gateset:
            if isinstance(gate, (ops.CNotPowGate, ops.CZPowGate, ops.CCXPowGate, ops.CCZPowGate)):
                raise ValueError('controlled gates must have integer exponents')
            raise ValueError(f'Unsupported gate: {gate!r}')

    def validate_operation(self, operation: ops.Operation):
        """Raises an error if the given operation is invalid on this device.

        Args:
            operation: the operation to validate

        Raises:
            ValueError: If the operation is not valid
        """
        if not isinstance(operation, ops.GateOperation):
            raise ValueError(f'Unsupported operation: {operation!r}')

        # All qubits the operation acts on must be on the device
        for q in operation.qubits:
            if q not in self.qubits:
                raise ValueError(f'Qubit not on device: {q!r}')

        if operation not in self.gateset and not (
            operation in self.xy_gateset_all_allowed and len(operation.qubits) == len(self.qubits)
        ):
            raise ValueError(f'Unsupported operation: {operation!r}')

        if operation in self.controlled_gateset:
            if len(operation.qubits) > self._max_parallel_c:
                raise ValueError(
                    'Too many qubits acted on in parallel by a controlled gate operation'
                )
            for p in operation.qubits:
                for q in operation.qubits:
                    if self.distance(p, q) > self._control_radius:
                        raise ValueError(f"Qubits {p!r}, {q!r} are too far away")

    def validate_moment(self, moment: circuits.Moment):
        """Raises an error if the given moment is invalid on this device.

        Args:
            moment: The moment to validate

        Raises:
            ValueError: If the given moment is invalid
        """
        super().validate_moment(moment)

        CATEGORIES = {
            'Z': (ops.ZPowGate,),
            'XY': (ops.XPowGate, ops.YPowGate, ops.PhasedXPowGate),
            'controlled': (ops.CNotPowGate, ops.CZPowGate, ops.CCXPowGate, ops.CCZPowGate),
            'measure': (ops.MeasurementGate,),
        }

        categorized_ops: DefaultDict = collections.defaultdict(list)
        for op in moment.operations:
            assert isinstance(op, ops.GateOperation)
            for k, v in CATEGORIES.items():
                assert isinstance(v, tuple)
                gate = _subgate_if_parallel_gate(op.gate)
                if isinstance(gate, v):
                    categorized_ops[k].append(op)

        for k in ['Z', 'XY', 'controlled']:
            if len(set(_subgate_if_parallel_gate(op.gate) for op in categorized_ops[k])) > 1:
                raise ValueError(f"Non-identical simultaneous {k} gates")

        num_parallel_xy = sum([len(op.qubits) for op in categorized_ops['XY']])
        num_parallel_z = sum([len(op.qubits) for op in categorized_ops['Z']])
        has_measurement = len(categorized_ops['measure']) > 0
        controlled_qubits_lists = [op.qubits for op in categorized_ops['controlled']]

        if sum([len(l) for l in controlled_qubits_lists]) > self._max_parallel_c:
            raise ValueError("Too many qubits acted on by controlled gates")
        if controlled_qubits_lists and (num_parallel_xy or num_parallel_z):
            raise ValueError(
                "Can't perform non-controlled operations at same time as controlled operations"
            )
        if self._are_qubit_lists_too_close(*controlled_qubits_lists):
            raise ValueError("Interacting controlled operations")

        if num_parallel_z > self._max_parallel_z:
            raise ValueError("Too many simultaneous Z gates")

        if num_parallel_xy > self._max_parallel_xy and num_parallel_xy != len(self.qubits):
            raise ValueError("Bad number of simultaneous XY gates")

        if has_measurement:
            if controlled_qubits_lists or num_parallel_z or num_parallel_xy:
                raise ValueError("Measurements can't be simultaneous with other operations")

    def _are_qubit_lists_too_close(self, *qubit_lists: Iterable[raw_types.Qid]) -> bool:
        if len(qubit_lists) < 2:
            return False
        if len(qubit_lists) == 2:
            a, b = qubit_lists
            return any(self.distance(p, q) <= self._control_radius for p in a for q in b)
        return any(
            self._are_qubit_lists_too_close(a, b) for a, b in itertools.combinations(qubit_lists, 2)
        )

    def validate_circuit(self, circuit: circuits.AbstractCircuit):
        """Raises an error if the given circuit is invalid on this device.

        A circuit is invalid if any of its moments are invalid or if there is a
        non-empty moment after a moment with a measurement.

        Args:
            circuit: The circuit to validate

        Raises:
            ValueError: If the given circuit can't be run on this device
        """
        super().validate_circuit(circuit)

        # Measurements must be in the last non-empty moment
        has_measurement_occurred = False
        for moment in circuit:
            if has_measurement_occurred:
                if len(moment.operations) > 0:
                    raise ValueError("Non-empty moment after measurement")
            for operation in moment.operations:
                if isinstance(operation.gate, ops.MeasurementGate):
                    has_measurement_occurred = True

    def _value_equality_values_(self) -> Any:
        return (
            self._measurement_duration,
            self._gate_duration,
            self._max_parallel_z,
            self._max_parallel_xy,
            self._max_parallel_c,
            self._control_radius,
            self.qubits,
        )

    def __repr__(self) -> str:
        return (
            'cirq.NeutralAtomDevice('
            f'measurement_duration={self._measurement_duration!r}, '
            f'gate_duration={self._gate_duration!r}, '
            f'max_parallel_z={self._max_parallel_z!r}, '
            f'max_parallel_xy={self._max_parallel_xy!r}, '
            f'max_parallel_c={self._max_parallel_c!r}, '
            f'control_radius={self._control_radius!r}, '
            f'qubits={sorted(self.qubits)!r})'
        )

    def neighbors_of(self, qubit: 'cirq.GridQubit') -> Iterable['cirq.GridQubit']:
        """Returns the qubits that the given qubit can interact with."""
        possibles = [
            GridQubit(qubit.row + 1, qubit.col),
            GridQubit(qubit.row - 1, qubit.col),
            GridQubit(qubit.row, qubit.col + 1),
            GridQubit(qubit.row, qubit.col - 1),
        ]
        return [e for e in possibles if e in self.qubits]

    def distance(self, p: 'cirq.Qid', q: 'cirq.Qid') -> float:
        p = cast(GridQubit, p)
        q = cast(GridQubit, q)
        return sqrt((p.row - q.row) ** 2 + (p.col - q.col) ** 2)

    def __str__(self) -> str:
        diagram = circuits.TextDiagramDrawer()

        for q in self.qubits:
            diagram.write(q.col, q.row, str(q))
            for q2 in self.neighbors_of(q):
                diagram.grid_line(q.col, q.row, q2.col, q2.row)

        return diagram.render(horizontal_spacing=3, vertical_spacing=2, use_unicode_characters=True)

    def _repr_pretty_(self, p: Any, cycle: bool):
        """iPython (Jupyter) pretty print."""
        p.text("cirq.NeutralAtomDevice(...)" if cycle else self.__str__())
