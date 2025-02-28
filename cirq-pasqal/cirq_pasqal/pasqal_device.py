# Copyright 2020 The Cirq Developers
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

from typing import Sequence, Any, Union, Dict
import numpy as np
import networkx as nx

import cirq
from cirq import GridQubit, LineQubit
from cirq.ops import NamedQubit
from cirq_pasqal import ThreeDQubit, TwoDQubit, PasqalGateset


@cirq.value.value_equality(unhashable=True)
class PasqalDevice(cirq.devices.Device):
    """A generic Pasqal device.

    The most general of Pasqal devices, enforcing only restrictions expected to
    be shared by all future devices. Serves as the parent class of all Pasqal
    devices, but can also be used on its own for hosting a nearly unconstrained
    device. When used as a circuit's device, the qubits have to be of the type
    cirq.NamedQubit and assumed to be all connected, the idea behind it being
    that after submission, all optimization and transpilation necessary for its
    execution on the specified device are handled internally by Pasqal.
    """

    def __init__(self, qubits: Sequence[cirq.Qid]) -> None:
        """Initializes a device with some qubits.

        Args:
            qubits (NamedQubit): Qubits on the device, exclusively unrelated to
                a physical position.
        Raises:
            TypeError: If the wrong qubit type is provided.
            ValueError: If the number of qubits is greater than the devices maximum.

        """
        if len(qubits) > 0:
            q_type = type(qubits[0])

        for q in qubits:
            if not isinstance(q, self.supported_qubit_type):
                raise TypeError(
                    f'Unsupported qubit type: {q!r}. This device '
                    f'supports qubit types: {self.supported_qubit_type}'
                )
            if not type(q) is q_type:
                raise TypeError("All qubits must be of same type.")

        if len(qubits) > self.maximum_qubit_number:
            raise ValueError(
                f'Too many qubits. {type(self)} accepts at most {self.maximum_qubit_number} qubits.'
            )

        self.gateset = PasqalGateset()
        self.qubits = qubits
        self._metadata = cirq.DeviceMetadata(
            qubits, nx.from_edgelist([(a, b) for a in qubits for b in qubits if a != b])
        )

    # pylint: enable=missing-raises-doc
    @property
    def supported_qubit_type(self):
        return (NamedQubit,)

    @property
    def maximum_qubit_number(self):
        return 100

    @property
    def metadata(self):
        return self._metadata

    def qubit_list(self):
        return [qubit for qubit in self.qubits]

    def is_pasqal_device_op(self, op: cirq.Operation) -> bool:
        if not isinstance(op, cirq.Operation):
            raise ValueError('Got unknown operation:', op)
        return op in self.gateset

    def validate_operation(self, operation: cirq.Operation):
        """Raises an error if the given operation is invalid on this device.

        Args:
            operation: The operation to validate.

        Raises:
            ValueError: If the operation is not valid.
            NotImplementedError: If the operation is a measurement with an invert
                mask.
        """

        if not isinstance(operation, cirq.GateOperation):
            raise ValueError("Unsupported operation")

        if not self.is_pasqal_device_op(operation):
            raise ValueError(f'{operation.gate!r} is not a supported gate')

        for qub in operation.qubits:
            if not isinstance(qub, self.supported_qubit_type):
                raise ValueError(
                    f'{qub} is not a valid qubit for gate {operation.gate!r}. This '
                    f'device accepts gates on qubits of type: '
                    f'{self.supported_qubit_type}'
                )
            if qub not in self.metadata.qubit_set:
                raise ValueError(f'{qub} is not part of the device.')

        if isinstance(operation.gate, cirq.MeasurementGate):
            if operation.gate.invert_mask != ():
                raise NotImplementedError(
                    "Measurements on Pasqal devices don't support invert_mask."
                )

    def validate_circuit(self, circuit: 'cirq.AbstractCircuit') -> None:
        """Raises an error if the given circuit is invalid on this device.

        A circuit is invalid if any of its moments are invalid or if there
        is a non-empty moment after a moment with a measurement.

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
                if isinstance(operation.gate, cirq.MeasurementGate):
                    has_measurement_occurred = True

    def __repr__(self):
        return f'pasqal.PasqalDevice(qubits={sorted(self.qubits)!r})'

    def _value_equality_values_(self):
        return self.qubits

    def _json_dict_(self):
        return cirq.protocols.obj_to_dict_helper(self, ['qubits'])


class PasqalVirtualDevice(PasqalDevice):
    """A Pasqal virtual device with qubits in 3d.

    A virtual representation of a Pasqal device, enforcing the constraints
    typically found in a physical device. The qubits can be positioned in 3d
    space, although 2d layouts will be supported sooner and are thus
    recommended. Only accepts qubits with physical placement.
    """

    def __init__(
        self, control_radius: float, qubits: Sequence[Union[ThreeDQubit, GridQubit, LineQubit]]
    ) -> None:
        """Initializes a device with some qubits.

        Args:
            control_radius: the maximum distance between qubits for a controlled
                gate. Distance is measured in units of the coordinates passed
                into the qubit constructor.
            qubits: Qubits on the device, identified by their x, y, z position.
                Must be of type ThreeDQubit, TwoDQubit, LineQubit or GridQubit.

        Raises:
            ValueError: if the wrong qubit type is provided or if invalid
                parameter is provided for control_radius."""

        super().__init__(qubits)

        if not control_radius >= 0:
            raise ValueError('Control_radius needs to be a non-negative float.')

        if len(self.qubits) > 1:
            if control_radius > 3.0 * self.minimal_distance():
                raise ValueError(
                    'Control_radius cannot be larger than 3 times'
                    ' the minimal distance between qubits.'
                )
        self.control_radius = control_radius
        self.gateset = PasqalGateset(include_additional_controlled_ops=False)
        self.controlled_gateset = cirq.Gateset(cirq.AnyIntegerPowerGateFamily(cirq.CZPowGate))

    @property
    def supported_qubit_type(self):
        return (ThreeDQubit, TwoDQubit, GridQubit, LineQubit)

    def validate_operation(self, operation: cirq.Operation):
        """Raises an error if the given operation is invalid on this device.

        Args:
            operation: the operation to validate
        Raises:
            ValueError: If the operation is not valid
        """
        super().validate_operation(operation)

        # Verify that a controlled gate operation is valid
        if operation in self.controlled_gateset:
            for p in operation.qubits:
                for q in operation.qubits:
                    if self.distance(p, q) > self.control_radius:
                        raise ValueError(f"Qubits {p!r}, {q!r} are too far away")

    def validate_moment(self, moment: cirq.Moment):
        """Raises an error if the given moment is invalid on this device.

        Args:
            moment: The moment to validate.
        Raises:
            ValueError: If the given moment is invalid.
        """

        super().validate_moment(moment)
        if len(moment) > 1:
            for operation in moment:
                if not isinstance(operation.gate, cirq.MeasurementGate):
                    raise ValueError("Cannot do simultaneous gates. Use cirq.InsertStrategy.NEW.")

    def minimal_distance(self) -> float:
        """Returns the minimal distance between two qubits in qubits.

        Args:
            qubits: qubit involved in the distance computation

        Raises:
            ValueError: If the device has only one qubit

        Returns:
            The minimal distance between qubits, in spacial coordinate units.
        """
        if len(self.qubits) <= 1:
            raise ValueError("Two qubits to compute a minimal distance.")

        return min([self.distance(q1, q2) for q1 in self.qubits for q2 in self.qubits if q1 != q2])

    def distance(self, p: Any, q: Any) -> float:
        """Returns the distance between two qubits.

        Args:
            p: qubit involved in the distance computation
            q: qubit involved in the distance computation

        Raises:
            ValueError: If p or q not part of the device

        Returns:
            The distance between qubits p and q.
        """
        all_qubits = self.qubit_list()
        if p not in all_qubits or q not in all_qubits:
            raise ValueError("Qubit not part of the device.")

        if isinstance(p, GridQubit):
            return np.sqrt((p.row - q.row) ** 2 + (p.col - q.col) ** 2)

        if isinstance(p, LineQubit):
            return abs(p.x - q.x)

        return np.sqrt((p.x - q.x) ** 2 + (p.y - q.y) ** 2 + (p.z - q.z) ** 2)

    def __repr__(self):
        return (
            'pasqal.PasqalVirtualDevice('
            f'control_radius={self.control_radius!r}, '
            f'qubits={sorted(self.qubits)!r})'
        )

    def _value_equality_values_(self) -> Any:
        return (self.control_radius, self.qubits)

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.protocols.obj_to_dict_helper(self, ['control_radius', 'qubits'])
