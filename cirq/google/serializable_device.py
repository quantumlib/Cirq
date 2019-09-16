# Copyright 2019 The Cirq Developers
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
"""Device object for converting from device specification protos"""
from typing import Any, cast, Dict, List, Set, Tuple, Type, TYPE_CHECKING

from cirq import devices, ops
from cirq.google import serializable_gate_set
from cirq.google.api.v2 import device_pb2
from cirq.value import Duration

if TYPE_CHECKING:
    import cirq


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
        self.allowed_targets: Dict[str, Set[Tuple['cirq.Qid', ...]]] = dict()
        for ts in proto.valid_targets:
            self.allowed_targets[ts.name] = self._create_target_set(ts)

        gate_defs: Dict[str, device_pb2.GateDefinition] = dict()
        for gs in proto.valid_gate_sets:
            for gate_def in gs.valid_gates:
                gate_defs[gate_def.id] = gate_def

        self.durations: Dict[Any, Duration] = dict()
        self.target_sets: Dict[Any, List[str]] = dict()
        for gate_type in gate_set.supported_gate_types():
            for serializer in gate_set.serializers[gate_type]:
                gate_id = serializer.serialized_gate_id
                if gate_id not in gate_defs:
                    raise ValueError(f'Serializer has {gate_id} which is not ' +
                                     'supported by the device specification')
                gate_picos = gate_defs[gate_id].gate_duration_picos
                self.durations[gate_type] = Duration(picos=gate_picos)
                self.target_sets[gate_type] = gate_defs[gate_id].valid_targets

    def _qid_from_str(self, id_str: str) -> 'cirq.Qid':
        try:
            return devices.GridQubit.from_proto_id(id_str)
        except ValueError:
            return ops.NamedQubit(id_str)

    def _qubits_from_ids(self, id_list) -> List['cirq.Qid']:
        """Translates a list of ids in proto format e.g. '4_3'
        into cirq.GridQubit objects"""
        return [self._qid_from_str(id) for id in id_list]

    def _create_target_set(self, ts: device_pb2.TargetSet
                          ) -> Set[Tuple['cirq.Qid', ...]]:
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

    def _find_operation_type(self, op_key: 'cirq.Operation',
                             type_dict: Dict[Type['cirq.Gate'], Any]) -> Any:
        """Finds the type (or a compatible type) of an operation from within
        a dictionary with keys of Gate type.

        Returns:
             the value corresponding to that key or None if no type matches
        """
        gate_op = cast(ops.GateOperation, op_key)
        gate_key = gate_op.gate
        for type_key in type_dict:
            if isinstance(gate_key, type_key):
                return type_dict[type_key]
        return None

    def duration_of(self, operation: 'cirq.Operation') -> Duration:
        duration = self._find_operation_type(operation, self.durations)
        if duration is not None:
            return duration
        raise ValueError(
            f'Operation {operation} does not have a duration listed')

    def validate_operation(self, operation: 'cirq.Operation') -> None:
        for q in operation.qubits:
            if q not in self.qubits:
                raise ValueError('Qubit not on device: {!r}'.format(q))

        if (len(operation.qubits) > 1):
            # TODO(dstrain): verify number of qubits and args

            qubit_tuple = tuple(operation.qubits)
            gate_op = cast(ops.GateOperation, operation)
            for t in self.target_sets:
                if isinstance(gate_op.gate, t):
                    for ts in self.target_sets[t]:
                        if qubit_tuple in self.allowed_targets[ts]:
                            # Valid
                            return
            # Target is not within any of the target sets specified by the gate.
            raise ValueError(
                f'Operation does not use valid qubit target: {operation}.')

    def validate_scheduled_operation(
            self, schedule: 'cirq.Schedule',
            scheduled_operation: 'cirq.ScheduledOperation') -> None:
        pass

    def validate_schedule(self, schedule: 'cirq.Schedule') -> None:
        pass
