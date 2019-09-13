from typing import Set, Tuple

from cirq import devices
from cirq.google import serializable_gate_set
from cirq.google.api.v2 import device_pb2
from cirq.value import Duration


class SerializableDevice(devices.Device):
    """Device object generated from a device specification proto.

    Given a device specification proto and a gate_set to translate the
    serialized gate_ids to cirq Gates, this will generate a Device that can
    verify operations and circuits for the hardware specified by the device.
    """

    def __init__(self, proto: device_pb2.DeviceSpecification,
        gate_set: serializable_gate_set.SerializableGateSet):
        """

        Args:
            proto: A proto describing the qubits on the device, as well as the
                supported gates and timing information.
            gate_set: A SerializableGateSet that can translate the gate_ids
                into cirq Gates.
        """

        self.qubits = self._qubits_from_ids(proto.valid_qubits)
        self.allowed_targets = dict()
        for ts in proto.valid_targets:
            self.allowed_targets[ts.name] = self._create_target_set(ts)

        gate_definitions = dict()
        for gs in proto.valid_gate_sets:
            for gate_def in gs.valid_gates:
                gate_definitions[gate_def.id] = gate_def

        self.durations = dict()
        self.target_sets = dict()
        for type in gate_set.supported_gate_types():
            for gate_id in gate_set.deserializers:
                if gate_id not in gate_definitions:
                    raise ValueError(f'Serializer has {gate_id} which is not ' +
                                     'supported by the device specification')
                gate_picos = gate_definitions[gate_id].gate_duration_picos
                self.durations[type] = Duration(picos = gate_picos)
                self.target_sets[type] = gate_definitions[gate_id].valid_targets


    def _qubits_from_ids(self, id_list):
        """Translates a list of ids in proto format e.g. '4_3'
        into cirq.GridQubit objects"""
        # TODO(dstrain): Add support for non-grid qubits
        return [devices.GridQubit.from_proto_id(id) for id in id_list]

    def _create_target_set(self, ts: device_pb2.TargetSet) -> Set[Tuple]:
        """Transform a TargetSet proto into a set of qubit tuples"""
        # TODO(dstrain): add support for measurement qubits
        target_set = set()
        for target in ts.targets:
            qid_list = self._qubits_from_ids(target.ids)
            target_set.add(tuple(qid_list))
            if ts.target_ordering == device_pb2.TargetSet.SYMMETRIC:
                qid_list.reverse()
                target_set.add(tuple(qid_list))
        return target_set

    def duration_of(self, operation: 'cirq.Operation') -> Duration:
        if type(operation) in self.durations:
            return self.durations[type(operation)]
        else:
            for t in self.durations:
                if isinstance(operation, t):
                    return self.durations[type(operation)]
        raise ValueError(
            f'Operation {operation} does not have a duration listed')

    def validate_operation(self, operation: 'cirq.Operation') -> None:
        for q in operation.qubits:
            if q not in self.qubits:
                raise ValueError('Qubit not on device: {!r}'.format(q))

        if (len(operation.qubits) > 1):
            # TODO(dstrain): verify number of qubits and args

            qubit_tuple = tuple(operation.qubits)
            for t in self.target_sets:
                if isinstance(operation.gate, t):
                    for ts in self.target_sets[t]:
                        if qubit_tuple in self.allowed_targets[ts]:
                            # Valid
                            return
            # Target is not within any of the target sets specified by the gate.
            raise ValueError(
                f'Operation does not use valid qubit target: {operation}.')

    def validate_scheduled_operation(
        self,
        schedule: 'cirq.Schedule',
        scheduled_operation: 'cirq.ScheduledOperation'
    ) -> None:
        pass

    def validate_schedule(self, schedule: 'cirq.Schedule') -> None:
        pass