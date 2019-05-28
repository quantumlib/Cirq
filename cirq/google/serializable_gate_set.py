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
"""Support for serializing and deserializing cirq.api.google.v2 protos."""

from typing import cast, Dict, Iterable, Optional, Tuple, Union

from cirq.google import op_deserializer, op_serializer

from cirq import circuits, devices, ops, schedules, value


class SerializableGateSet:
    """A class for serializing and deserializing programs and operations.

    This class is for cirq.api.google.v2. protos.
    """

    def __init__(self, gate_set_name: str,
                 serializers: Iterable[op_serializer.GateOpSerializer],
                 deserializers: Iterable[op_deserializer.GateOpDeserializer]):
        self.gate_set_name = gate_set_name
        self.serializers = {s.gate_type: s for s in serializers}
        self.deserializers = {d.serialized_gate_id: d for d in deserializers}

    def supported_gate_types(self) -> Tuple:
        return tuple(self.serializers.keys())

    def serialize(self,
                  program: Union[circuits.Circuit, schedules.Schedule]) -> Dict:
        """Serialize a Circuit or Schedule to cirq.api.google.v2.Program proto.

        Args:
            program: The Circuit or Schedule to serialize.

        Returns:
            A dictionary corresponding to the cirq.api.google.v2.Program proto.
        """
        proto = {'language': {'gate_set': self.gate_set_name}}
        if isinstance(program, circuits.Circuit):
            proto['circuit'] = self._serialize_circuit(program)
        else:
            proto['schedule'] = self._serialize_schedule(program)
        return proto

    def serialize_op(self, op: ops.Operation) -> Dict:
        """Serialize an Operation to cirq.api.google.v2.Operation proto.

        Args:
            op: The operation to serialize.

        Returns:
            A dictionary corresponds to the cirq.api.google.v2.Operation proto.
        """
        gate_op = cast(ops.GateOperation, op)
        gate_type = type(gate_op.gate)
        for gate_type_mro in gate_type.mro():
            # Check all super classes in method resolution order.
            if gate_type_mro in self.serializers:
                return self.serializers[gate_type_mro].to_proto_dict(gate_op)
        raise ValueError('Cannot serialize op of type {}'.format(gate_type))

    def deserialize(self, proto: Dict, device: Optional[devices.Device] = None
                   ) -> Union[circuits.Circuit, schedules.Schedule]:
        """Deserialize a Circuit or Schedule from a cirq.api.google.v2.Program.

        Args:
            proto: A dictionary representing a cirq.api.google.v2.Program proto.
            device: If the proto is for a schedule, a device is required
                Otherwise optional.

        Returns:
            The deserialized Circuit or Schedule, with a device if device was
            not None.
        """
        if 'language' not in proto or 'gate_set' not in proto['language']:
            raise ValueError('Missing gate set specification.')
        if proto['language']['gate_set'] != self.gate_set_name:
            raise ValueError('Gate set in proto was {} but expected {}'.format(
                proto['language']['gate_set'], self.gate_set_name))
        if 'circuit' in proto:
            circuit = self._deserialize_circuit(proto['circuit'])
            return circuit if device is None else circuit.with_device(device)
        elif 'schedule' in proto:
            if device is None:
                raise ValueError(
                    'Deserializing schedule requires a device but None was '
                    'given.')
            return self._deserialize_schedule(proto['schedule'], device)
        else:
            raise ValueError(
                'Program proto does not contain a circuit or schedule.')

    def deserialize_op(self, operation_proto) -> ops.Operation:
        """Deserialize an Operation from a cirq.api.google.v2.Operation.

        Args:
            operation_proto: A dictionary representing a
                cirq.api.google.v2.Operation proto.

        Returns:
            The deserialized Operation.
        """
        if 'gate' not in operation_proto or 'id' not in operation_proto['gate']:
            raise ValueError('Operation proto does not have a gate.')

        gate_id = operation_proto['gate']['id']
        if gate_id in self.deserializers.keys():
            return self.deserializers[gate_id].from_proto_dict(operation_proto)
        else:
            raise ValueError(
                'Unsupported serialized gate with id {}'.format(gate_id))

    def _serialize_circuit(self, circuit: circuits.Circuit) -> Dict:
        moment_protos = []
        for moment in circuit:
            moment_proto = {
                'operations': [self.serialize_op(op) for op in moment]
            }
            moment_protos.append(moment_proto)
        return {
            'scheduling_strategy': 1,  # MOMENT_BY_MOMENT
            'moments': moment_protos
        }

    def _serialize_schedule(self, schedule: schedules.Schedule) -> Dict:
        scheduled_op_protos = []
        for scheduled_op in schedule.scheduled_operations:
            scheduled_op_protos.append({
                'operation':
                self.serialize_op(scheduled_op.operation),
                'start_time_picos':
                scheduled_op.time.raw_picos()
            })
        return {'scheduled_operations': scheduled_op_protos}

    def _deserialize_circuit(self, circuit_proto: Dict) -> circuits.Circuit:
        moments = []
        if 'moments' not in circuit_proto:
            raise ValueError('Circuit proto has no moments.')
        for moment_proto in circuit_proto['moments']:
            if 'operations' not in moment_proto:
                moments.append(ops.Moment())
                continue
            moment_ops = [
                self.deserialize_op(o) for o in moment_proto['operations']
            ]
            moments.append(ops.Moment(moment_ops))
        return circuits.Circuit(moments)

    def _deserialize_schedule(self, schedule_proto: Dict,
                              device: devices.Device) -> schedules.Schedule:
        if 'scheduled_operations' not in schedule_proto:
            raise ValueError('Schedule proto missing scheduled operations.')
        scheduled_ops = []
        for scheduled_op_proto in schedule_proto['scheduled_operations']:
            if 'operation' not in scheduled_op_proto:
                raise ValueError('Scheduled op missing an operation {}'.format(
                    scheduled_op_proto))
            if 'start_time_picos' not in scheduled_op_proto:
                raise ValueError('Scheduled op missing a start time {}'.format(
                    scheduled_op_proto))
            scheduled_op = schedules.ScheduledOperation.op_at_on(
                operation=self.deserialize_op(scheduled_op_proto['operation']),
                time=value.Timestamp(
                    picos=scheduled_op_proto['start_time_picos']),
                device=device)
            scheduled_ops.append(scheduled_op)
        return schedules.Schedule(device, scheduled_ops)
