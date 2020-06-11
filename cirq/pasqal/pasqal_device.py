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
from typing import FrozenSet, Callable, List, Sequence, Any, Union
import numpy as np
from numpy import sqrt

import cirq
from cirq import GridQubit, LineQubit
from cirq.ops import NamedQubit

from cirq.pasqal import ThreeDQubit, TwoDQubit


@cirq.value.value_equality
class PasqalDevice(cirq.devices.Device):
    """A generic Pasqal device."""

    def __init__(self, qubits: Sequence[cirq.ops.Qid]) -> None:
        """Initializes a device with some qubits.

        Args:
            qubits (NamedQubit): Qubits on the device, exclusively unrelated to
                a physical position.
        Raises:
            TypeError: if the wrong qubit type is provided.
        """
        if len(qubits) > 0:
            q_type = type(qubits[0])

        for q in qubits:
            if not isinstance(q, self.supported_qubit_type):
                raise TypeError('Unsupported qubit type: {!r}'.format(q))
            if not type(q) is q_type:
                raise TypeError("All qubits must be of same type.")

        self.qubits = qubits

    @property
    def supported_qubit_type(self):
        return (NamedQubit,)

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
        if not self.is_pasqal_device_op(operation):
            decomposition = PasqalConverter().pasqal_convert(
                operation, keep=self.is_pasqal_device_op)

        return decomposition

    def is_pasqal_device_op(self, op: cirq.ops.Operation) -> bool:

        if not isinstance(op, cirq.ops.Operation):
            raise ValueError('Got unknown operation:', op)

        valid_op = isinstance(op.gate,
                              (cirq.ops.IdentityGate, cirq.ops.MeasurementGate,
                               cirq.ops.PhasedXPowGate, cirq.ops.XPowGate,
                               cirq.ops.YPowGate, cirq.ops.ZPowGate))

        if not valid_op:  # To prevent further checking if already passed
            if isinstance(
                    op.gate,
                (cirq.ops.HPowGate, cirq.ops.CNotPowGate, cirq.ops.CZPowGate,
                 cirq.ops.CCZPowGate, cirq.ops.CCXPowGate)):
                expo = op.gate.exponent
                valid_op = np.isclose(expo, np.around(expo, decimals=0))

        return valid_op

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

        if not self.is_pasqal_device_op(operation):
            raise ValueError('{!r} is not a supported '
                             'gate'.format(operation.gate))

        all_qubits = self.qubit_list()
        for qub in operation.qubits:
            if not isinstance(qub, self.supported_qubit_type):
                raise ValueError('{} is not a valid qubit '
                                 'for gate {!r}'.format(qub, operation.gate))
            try:
                all_qubits.remove(qub)
            except ValueError:
                raise ValueError('{} is not part of the device.'.format(qub))

        if isinstance(operation.gate, cirq.ops.MeasurementGate):
            if all_qubits:  # We enforce that all qubits are measured at once
                raise ValueError('All qubits have to be measured at once '
                                 'on a PasqalDevice.')
            if operation.gate.invert_mask != ():
                raise NotImplementedError("Measurements on Pasqal devices "
                                          "don't support invert_mask.")

    def __repr__(self):
        return 'pasqal.PasqalDevice(qubits={!r})'.format(sorted(self.qubits))

    def _value_equality_values_(self):
        return (self.qubits)

    def _json_dict_(self):
        return cirq.protocols.obj_to_dict_helper(self, ['qubits'])


class PasqalVirtualDevice(PasqalDevice):
    """A Pasqal virtual device with qubits in 3D."""

    def __init__(self, control_radius: float,
                 qubits: Sequence[Union[ThreeDQubit, GridQubit, LineQubit]]
                ) -> None:
        """Initializes a device with some qubits.

        Args:
            control_radius: the maximum distance between qubits for a controlled
                gate. Distance is measured in units of the coordinates passed
                into the qubit constructor.
            qubits: Qubits on the device, identified by their x, y, z position.
                Must be of type ThreeDQubit, LineQubit or GridQubit.

        Raises:
            ValueError: if the wrong qubit type is provided or if invalid
                parameter is provided for control_radius. """

        super().__init__(qubits)

        if not control_radius >= 0:
            raise ValueError('Control_radius needs to be a non-negative float')

        if len(self.qubits) > 1:
            if control_radius >= 3. * self.minimal_distance():
                raise ValueError('Control_radius cannot be larger than 3 times'
                                 ' the minimal distance between qubits.')

        self.control_radius = control_radius

    @property
    def supported_qubit_type(self):
        return (
            ThreeDQubit,
            TwoDQubit,
            GridQubit,
            LineQubit,
        )

    def is_pasqal_device_op(self, op: cirq.ops.Operation) -> bool:
        return (super().is_pasqal_device_op(op) and not isinstance(
            op.gate,
            (cirq.ops.CNotPowGate, cirq.ops.CCZPowGate, cirq.ops.CCXPowGate)))

    def validate_operation(self, operation: cirq.ops.Operation):
        """Raises an error if the given operation is invalid on this device.

        Args:
            operation: the operation to validate
        Raises:
            ValueError: If the operation is not valid
        """
        super().validate_operation(operation)

        # Verify that a controlled gate operation is valid
        if isinstance(operation, cirq.ops.GateOperation):
            if (len(operation.qubits) > 1 and
                    not isinstance(operation.gate, cirq.ops.MeasurementGate)):
                for p in operation.qubits:
                    for q in operation.qubits:
                        if self.distance(p, q) > self.control_radius:
                            raise ValueError("Qubits {!r}, {!r} are too "
                                             "far away".format(p, q))

    def validate_moment(self, moment: cirq.ops.Moment):
        """Raises an error if the given moment is invalid on this device.

        Args:
            moment: The moment to validate.
        Raises:
            ValueError: If the given moment is invalid.
        """

        super().validate_moment(moment)

        if len(set(operation.gate for operation in moment.operations)) > 1:
            raise ValueError("Cannot do simultaneous gates")

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

        return min([
            self.distance(q1, q2)
            for q1 in self.qubits
            for q2 in self.qubits
            if q1 != q2
        ])

    def distance(self, p: Any, q: Any) -> float:
        """
        Returns the distance between two qubits.
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
            return sqrt((p.row - q.row)**2 + (p.col - q.col)**2)

        if isinstance(p, LineQubit):
            return abs(p.x - q.x)

        return sqrt((p.x - q.x)**2 + (p.y - q.y)**2 + (p.z - q.z)**2)

    def __repr__(self):
        return ('pasqal.PasqalVirtualDevice(control_radius={!r}, '
                'qubits={!r})').format(self.control_radius, sorted(self.qubits))

    def _value_equality_values_(self) -> Any:
        return (self.control_radius, self.qubits)

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.protocols.obj_to_dict_helper(self,
                                                 ['control_radius', 'qubits'])


class PasqalConverter(cirq.neutral_atoms.ConvertToNeutralAtomGates):
    """A gate converter for compatibility with Pasqal processors.

    Modified version of ConvertToNeutralAtomGates, where a new 'convert' method
    'pasqal_convert' takes the 'keep' function as an input.
    """

    def pasqal_convert(self, op: cirq.ops.Operation,
                       keep: Callable[[cirq.ops.Operation], bool]
                      ) -> List[cirq.ops.Operation]:

        def on_stuck_raise(bad):
            return TypeError("Don't know how to work with {!r}. "
                             "It isn't a native PasqalDevice operation, "
                             "a 1 or 2 qubit gate with a known unitary, "
                             "or composite.".format(bad))

        return cirq.protocols.decompose(
            op,
            keep=keep,
            intercepting_decomposer=self._convert_one,
            on_stuck_raise=None if self.ignore_failures else on_stuck_raise)
