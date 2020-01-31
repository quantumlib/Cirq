from typing import Iterable, cast
from numpy import sqrt

import cirq
from cirq import ops, protocols
from cirq.value import Duration
from cirq.neutral_atoms import NeutralAtomDevice
from cirq.pasqal import ThreeDGridQubit


class PasqalDevice(NeutralAtomDevice):

    def __init__(self, control_radius: float,
                 qubits: Iterable[ThreeDGridQubit]) -> None:

        us = 1000 * Duration(nanos=1)

        self._measurement_duration = 5000 * us
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

    def qubit_set(self) -> Iterable[ThreeDGridQubit]:
        return set(self.qubits)

    def decompose_operation(self, operation: ops.Operation) -> 'cirq.OP_TREE':

        # default value
        decomposition = [operation]

        if not isinstance(operation, ops.GateOperation):
            raise TypeError("{!r} is not a gate operation.".format(operation))


        #Try to decompose the operation into elementary device operations
        if not PasqalDevice.is_pasqal_device_op(operation):
            decomposition = cirq.decompose(operation,
                                           keep=PasqalDevice.is_pasqal_device_op)

        for dec in decomposition:
            if not PasqalDevice.is_pasqal_device_op(dec):
                raise TypeError("Don't know how to work with {!r}.".format(
                    operation.gate))

        return decomposition

    @staticmethod
    def is_pasqal_device_op(op: ops.Operation) -> bool:

        if isinstance(op, ops.MeasurementGate):
            return True

        if not isinstance(op, ops.GateOperation):
            return False

        keep = False

        # Currently accepting all multi-qubit operations
        keep = keep or (len(op.qubits) > 1)

        keep = keep or (isinstance(op.gate, ops.YPowGate))

        keep = keep or (isinstance(op.gate, ops.ZPowGate))

        keep = keep or (isinstance(op.gate, ops.XPowGate))

        keep = keep or (isinstance(op.gate, ops.PhasedXPowGate))

        keep = keep or (isinstance(op.gate, ops.IdentityGate))

        return keep

    def validate_operation(self, operation: ops.Operation):

        try:
            if isinstance(operation.gate,
                          (ops.MeasurementGate, ops.IdentityGate)):
                return
        except AttributeError:
            pass

        if not isinstance(operation, cirq.GateOperation):
            raise ValueError('{!r} is not a supported '
                             'operation'.format(operation))

        if not PasqalDevice.is_pasqal_device_op(operation):
            raise ValueError('{!r} is not a supported '
                             'gate'.format(operation.gate))

        for qub in operation.qubits:
            if not isinstance(qub, ThreeDGridQubit):
                raise ValueError('{} is not a 3D grid qubit '
                                 'for gate {!r}'.format(qub, operation.gate))

        # Verify that a controlled gate operation is valid
        if isinstance(operation, ops.GateOperation):
            if len(operation.qubits) > self._max_parallel_c + self._max_parallel_t:
                raise ValueError("Too many qubits acted on in parallel by a"
                                 "controlled gate operation")
            if len(operation.qubits) > 1:
                for p in operation.qubits:
                    for q in operation.qubits:
                        if self.distance(p, q) > self.control_radius:
                            raise ValueError("Qubits {!r}, {!r} are too "
                                             "far away".format(p, q))

        # Verify that a valid number of Z gates are applied in parallel
        if isinstance(operation.gate, ops.ZPowGate):
            if len(operation.qubits) > self._max_parallel_z:
                raise ValueError("Too many Z gates in parallel")

        # Verify that a valid number of XY gates are applied in parallel
        if isinstance(operation.gate,
                      (ops.XPowGate, ops.YPowGate, ops.PhasedXPowGate)):
            if (len(operation.qubits) > self._max_parallel_xy and
                    len(operation.qubits) != len(self.qubits)):
                raise ValueError("Bad number of X/Y gates in parallel")

    def distance(self, p: 'cirq.Qid', q: 'cirq.Qid') -> float:
        if not isinstance(q, ThreeDGridQubit):
            raise ValueError('Unsupported qubit type: {!r}'.format(q))
        if not isinstance(p, ThreeDGridQubit):
            raise ValueError('Unsupported qubit type: {!r}'.format(p))
        p = cast(ThreeDGridQubit, p)
        q = cast(ThreeDGridQubit, q)
        return sqrt((p.row - q.row) ** 2 + (p.col - q.col) ** 2 +
                    (p.lay - q.lay) ** 2)

    def __repr__(self):
        return ('pasqal.PasqalDevice(control_radius={!r}, '
                'qubits={!r})').format(self.control_radius,
                                       sorted(self.qubits))

    def _value_equality_values_(self):
        return (self._measurement_duration,
                self._gate_duration,
                self._max_parallel_z,
                self._max_parallel_xy,
                self._max_parallel_c,
                self.control_radius,
                self.qubits)

    def _json_dict_(self):
        return protocols.obj_to_dict_helper(self, ['control_radius',
                                                   'qubits'
                                                   ])
