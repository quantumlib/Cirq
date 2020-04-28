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
from typing import Any, Dict, FrozenSet, Iterable
from numpy import sqrt

import cirq

from cirq.pasqal import ThreeDGridQubit


@cirq.value.value_equality
class PasqalDevice(cirq.devices.Device):
    """A Pasqal Device with qubits placed on a 3D grid."""

    def __init__(self, control_radius: float,
                 qubits: Iterable[ThreeDGridQubit]) -> None:
        """ Initializes the description of the device.

        Args:
            control_radius: the maximum distance between qubits for a controlled
                gate. Distance is measured in units of the indices passed into
                the qubit constructor.
            qubits: Qubits on the device, identified by their x, y, z position.
                Must be of type ThreeDGridQubit.

        Raises:
            ValueError: if the wrong qubit type is provided or if invalid
                parameter is provided for control_radius. """
        us = cirq.value.Duration(micros=1)
        ms = cirq.value.Duration(millis=1)

        self._measurement_duration = 5 * ms
        self._gate_duration = 2 * us
        self._max_parallel_z = 2
        self._max_parallel_xy = 2
        self._max_parallel_c = 10
        self._max_parallel_t = 1

        for q in qubits:
            if not isinstance(q, ThreeDGridQubit):
                raise TypeError('Unsupported qubit type: {!r}'.format(q))

        if not control_radius >= 0:
            raise ValueError("control_radius needs to be a non-negative float")

        self.control_radius = control_radius
        self.qubits = qubits

    def qubit_set(self) -> FrozenSet[cirq.Qid]:
        return frozenset(self.qubits)

    def qubit_list(self):
        return [qubit for qubit in self.qubits]

    def decompose_operation(self,
                            operation: cirq.ops.Operation) -> 'cirq.OP_TREE':

        decomposition = [operation]

        if not isinstance(operation,
                          (cirq.ops.GateOperation, cirq.ParallelGateOperation)):
            raise TypeError("{!r} is not a gate operation.".format(operation))

        # Try to decompose the operation into elementary device operations
        if not PasqalDevice.is_pasqal_device_op(operation):
            decomposition = cirq.protocols.decompose(
                operation, keep=PasqalDevice.is_pasqal_device_op)

        return decomposition

    @staticmethod
    def is_pasqal_device_op(op: cirq.ops.Operation) -> bool:

        if not isinstance(op, cirq.ops.Operation):
            raise ValueError('Got unknown operation:', op)
        return (len(op.qubits) > 1) or isinstance(
            op.gate, (cirq.ops.IdentityGate, cirq.ops.MeasurementGate,
                      cirq.ops.PhasedXPowGate, cirq.ops.XPowGate,
                      cirq.ops.YPowGate, cirq.ops.ZPowGate))

    def validate_operation(self, operation: cirq.ops.Operation):
        """
        Raises an error if the given operation is invalid on this device.

        Args:
            operation: the operation to validate

        Raises:
            ValueError: If the operation is not valid
        """
        if not isinstance(operation,
                          (cirq.GateOperation, cirq.ParallelGateOperation)):
            raise ValueError("Unsupported operation")

        if not PasqalDevice.is_pasqal_device_op(operation):
            raise ValueError('{!r} is not a supported '
                             'gate'.format(operation.gate))

        for qub in operation.qubits:
            if not isinstance(qub, ThreeDGridQubit):
                raise ValueError('{} is not a 3D grid qubit '
                                 'for gate {!r}'.format(qub, operation.gate))

        if isinstance(operation.gate,
                      (cirq.ops.MeasurementGate, cirq.ops.IdentityGate)):
            return

        # Verify that a controlled gate operation is valid
        if isinstance(operation, cirq.ops.GateOperation):
            if len(operation.qubits) > self._max_parallel_c + \
                    self._max_parallel_t:
                raise ValueError("Too many qubits acted on in parallel by a"
                                 "controlled gate operation")
            if len(operation.qubits) > 1:
                for p in operation.qubits:
                    for q in operation.qubits:
                        if self.distance(p, q) > self.control_radius:
                            raise ValueError("Qubits {!r}, {!r} are too "
                                             "far away".format(p, q))

        # Verify that a valid number of Z gates are applied in parallel
        if isinstance(operation.gate, cirq.ops.ZPowGate):
            if len(operation.qubits) > self._max_parallel_z:
                raise ValueError("Too many Z gates in parallel")

        # Verify that a valid number of XY gates are applied in parallel
        if isinstance(
                operation.gate,
            (cirq.ops.XPowGate, cirq.ops.YPowGate, cirq.ops.PhasedXPowGate)):
            if (len(operation.qubits) > self._max_parallel_xy and
                    len(operation.qubits) != len(self.qubit_list())):
                raise ValueError("Bad number of X/Y gates in parallel")

    def duration_of(self, operation: cirq.ops.Operation):
        """
        Provides the duration of the given operation on this device.

        Args:
            operation: the operation to get the duration of

        Returns:
            The duration of the given operation on this device

        Raises:
            ValueError: If the operation provided doesn't correspond to a native
                gate
        """
        self.validate_operation(operation)
        if isinstance(operation,
                      (cirq.ops.GateOperation, cirq.ops.ParallelGateOperation)):
            if isinstance(operation.gate, cirq.ops.MeasurementGate):
                return self._measurement_duration
        return self._gate_duration

    def distance(self, p: 'cirq.Qid', q: 'cirq.Qid') -> float:
        """
        Returns the distance between two qubits.

        Args:
            p: qubit involved in the distance computation
            q: qubit involved in the distance computation

        Returns:
            The distance between qubits p and q, in lattice spacing units.

        """
        if not isinstance(q, ThreeDGridQubit):
            raise TypeError('Unsupported qubit type: {!r}'.format(q))
        if not isinstance(p, ThreeDGridQubit):
            raise TypeError('Unsupported qubit type: {!r}'.format(p))
        return sqrt((p.row - q.row)**2 + (p.col - q.col)**2 +
                    (p.lay - q.lay)**2)

    def __repr__(self) -> str:
        return (f'pasqal.PasqalDevice(control_radius={self.control_radius!r}, '
                f'qubits={sorted(self.qubits)!r})')

    def _value_equality_values_(self) -> Any:
        return (self.control_radius, self.qubits)

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.protocols.obj_to_dict_helper(self,
                                                 ['control_radius', 'qubits'])
