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
from typing import FrozenSet, Callable, List, Sequence, Any, Union, Dict

import numpy as np

import cirq
from cirq import GridQubit, LineQubit
from cirq.ops import NamedQubit
from cirq_pasqal import ThreeDQubit, TwoDQubit


@cirq.value.value_equality
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

    # TODO(#3388) Add documentation for Raises.
    # pylint: disable=missing-raises-doc
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
                raise TypeError(
                    'Unsupported qubit type: {!r}. This device '
                    'supports qubit types: {}'.format(q, self.supported_qubit_type)
                )
            if not type(q) is q_type:
                raise TypeError("All qubits must be of same type.")

        if len(qubits) > self.maximum_qubit_number:
            raise ValueError(
                'Too many qubits. {} accepts at most {} '
                'qubits.'.format(type(self), self.maximum_qubit_number)
            )

        self.gateset = cirq.Gateset(
            cirq.ParallelGateFamily(cirq.H),
            cirq.ParallelGateFamily(cirq.PhasedXPowGate),
            cirq.ParallelGateFamily(cirq.XPowGate),
            cirq.ParallelGateFamily(cirq.YPowGate),
            cirq.ParallelGateFamily(cirq.ZPowGate),
            cirq.AnyIntegerPowerGateFamily(cirq.CNotPowGate),
            cirq.AnyIntegerPowerGateFamily(cirq.CCNotPowGate),
            cirq.AnyIntegerPowerGateFamily(cirq.CZPowGate),
            cirq.AnyIntegerPowerGateFamily(cirq.CCZPowGate),
            cirq.IdentityGate,
            cirq.MeasurementGate,
            unroll_circuit_op=False,
            accept_global_phase=False,
        )
        self.qubits = qubits

    # pylint: enable=missing-raises-doc
    @property
    def supported_qubit_type(self):
        return (NamedQubit,)

    @property
    def maximum_qubit_number(self):
        return 100

    def qubit_set(self) -> FrozenSet[cirq.Qid]:
        return frozenset(self.qubits)

    def qubit_list(self):
        return [qubit for qubit in self.qubits]

    def decompose_operation(self, operation: cirq.ops.Operation) -> 'cirq.OP_TREE':

        decomposition = [operation]

        if not isinstance(operation, (cirq.ops.GateOperation, cirq.ParallelGateOperation)):
            raise TypeError(f"{operation!r} is not a gate operation.")

        # Try to decompose the operation into elementary device operations
        if not self.is_pasqal_device_op(operation):
            decomposition = PasqalConverter().pasqal_convert(
                operation, keep=self.is_pasqal_device_op
            )

        return decomposition

    def is_pasqal_device_op(self, op: cirq.ops.Operation) -> bool:
        if not isinstance(op, cirq.ops.Operation):
            raise ValueError('Got unknown operation:', op)
        return op in self.gateset

    # TODO(#3388) Add documentation for Raises.
    # pylint: disable=missing-raises-doc
    def validate_operation(self, operation: cirq.ops.Operation):
        """Raises an error if the given operation is invalid on this device.

        Args:
            operation: the operation to validate

        Raises:
            ValueError: If the operation is not valid
        """

        if not isinstance(operation, (cirq.GateOperation, cirq.ParallelGateOperation)):
            raise ValueError("Unsupported operation")

        if not self.is_pasqal_device_op(operation):
            raise ValueError(f'{operation.gate!r} is not a supported gate')

        for qub in operation.qubits:
            if not isinstance(qub, self.supported_qubit_type):
                raise ValueError(
                    '{} is not a valid qubit for gate {!r}. This '
                    'device accepts gates on qubits of type: '
                    '{}'.format(qub, operation.gate, self.supported_qubit_type)
                )
            if qub not in self.qubit_set():
                raise ValueError(f'{qub} is not part of the device.')

        if isinstance(operation.gate, cirq.ops.MeasurementGate):
            if operation.gate.invert_mask != ():
                raise NotImplementedError(
                    "Measurements on Pasqal devices don't support invert_mask."
                )

    # pylint: enable=missing-raises-doc
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
                if isinstance(operation.gate, cirq.ops.MeasurementGate):
                    has_measurement_occurred = True

    def can_add_operation_into_moment(
        self, operation: cirq.ops.Operation, moment: cirq.ops.Moment
    ) -> bool:
        """Determines if it's possible to add an operation into a moment.

        An operation can be added if the moment with the operation added is
        valid.

        Args:
            operation: The operation being added.
            moment: The moment being transformed.

        Returns:
            Whether or not the moment will validate after adding the operation.

        Raises:
            ValueError: If either of the given moment or operation is invalid
        """
        if not super().can_add_operation_into_moment(operation, moment):
            return False
        try:
            self.validate_moment(moment.with_operation(operation))
        except ValueError:
            return False
        return True

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
        self.exclude_gateset = cirq.Gateset(
            cirq.AnyIntegerPowerGateFamily(cirq.CNotPowGate),
            cirq.AnyIntegerPowerGateFamily(cirq.CCNotPowGate),
            cirq.AnyIntegerPowerGateFamily(cirq.CCZPowGate),
        )
        self.controlled_gateset = cirq.Gateset(
            *self.exclude_gateset.gates,
            cirq.AnyIntegerPowerGateFamily(cirq.CZPowGate),
        )

    @property
    def supported_qubit_type(self):
        return (
            ThreeDQubit,
            TwoDQubit,
            GridQubit,
            LineQubit,
        )

    def is_pasqal_device_op(self, op: cirq.ops.Operation) -> bool:
        return super().is_pasqal_device_op(op) and op not in self.exclude_gateset

    def validate_operation(self, operation: cirq.ops.Operation):
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

    def validate_moment(self, moment: cirq.ops.Moment):
        """Raises an error if the given moment is invalid on this device.

        Args:
            moment: The moment to validate.
        Raises:
            ValueError: If the given moment is invalid.
        """

        super().validate_moment(moment)
        if len(moment) > 1:
            for operation in moment:
                if not isinstance(operation.gate, cirq.ops.MeasurementGate):
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
        return ('pasqal.PasqalVirtualDevice(control_radius={!r}, qubits={!r})').format(
            self.control_radius, sorted(self.qubits)
        )

    def _value_equality_values_(self) -> Any:
        return (self.control_radius, self.qubits)

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.protocols.obj_to_dict_helper(self, ['control_radius', 'qubits'])

    def qid_pairs(self) -> FrozenSet['cirq.SymmetricalQidPair']:
        """Returns a list of qubit edges on the device.

        Returns:
            All qubit pairs that are less or equal to the control radius apart.
        """
        qs = self.qubits
        return frozenset(
            [
                cirq.SymmetricalQidPair(q, q2)
                for q in qs
                for q2 in qs
                if q < q2 and self.distance(q, q2) <= self.control_radius
            ]
        )


class PasqalConverter(cirq.neutral_atoms.ConvertToNeutralAtomGates):
    """A gate converter for compatibility with Pasqal processors.

    Modified version of ConvertToNeutralAtomGates, where a new 'convert' method
    'pasqal_convert' takes the 'keep' function as an input.
    """

    def pasqal_convert(
        self, op: cirq.ops.Operation, keep: Callable[[cirq.ops.Operation], bool]
    ) -> List[cirq.ops.Operation]:
        def on_stuck_raise(bad):
            return TypeError(
                "Don't know how to work with {!r}. "
                "It isn't a native PasqalDevice operation, "
                "a 1 or 2 qubit gate with a known unitary, "
                "or composite.".format(bad)
            )

        return cirq.protocols.decompose(
            op,
            keep=keep,
            intercepting_decomposer=self._convert_one,
            on_stuck_raise=None if self.ignore_failures else on_stuck_raise,
        )
