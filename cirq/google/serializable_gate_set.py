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

from typing import cast, Dict, Iterable, Union

from cirq.google import op_deserializer, op_serializer

from cirq import circuits, devices, ops, schedules, value


class SerializableGateSet():

    def __init__(self, gate_set_name: str,
                 serializers: Iterable[op_serializer.GateOpSerializer],
                 deserializers: Iterable[op_deserializer.GateOpDeserializer]):
        self.gate_set_name = gate_set_name
        self.serializers = {s.gate_type: s for s in serializers}
        self.deserializers = {d.serialized_gate_id: d for d in deserializers}

    def support_gate_types(self):
        return self.serializers.keys()

    def serialize(self,
                  program: Union[circuits.Circuit, schedules.Schedule]) -> Dict:
        """Serialize a Circuit or Schedule according to cirq."""
        proto = {'language': {'gate_set': self.gate_set_name}}
        if isinstance(program, circuits.Circuit):
            proto['circuit'] = self._serialize_circuit(program)
        else:
            proto['schedule'] = self._serialize_schedule(program)
        return program

    def serialize_op(self, op: ops.Operation):
        gate_op = cast(ops.GateOperation, op)
        gate_type = type(gate_op.gate)
        for gate_type_mro in gate_type.mro():
            # Check all super classes in method resolution order.
            if gate_type_mro in self.serializers:
                return self.serializers[gate_type_mro].to_proto(gate_op)
        raise ValueError('Cannot serialize op of type {}'.format(gate_type))

    def deserialize(self,
                    proto: Dict) -> Union[circuits.Circuit, schedules.Schedule]:
        if 'language' not in proto or 'gate_set' not in proto['language']:
            raise ValueError('')
        if proto['language']['gate_set'] != self.gate_set_name:
            raise ValueError('')
        if 'circuit' in proto:
            return self._deserialize_circuit(proto['circuit'])
        elif 'schedule' in proto:
            return self._deserialize_schedule(proto['schedule'])
        else:
            raise ValueError('')

    def deserialize_op(self, operation_proto) -> ops.Operation:
        if 'gate' not in operation_proto or 'id' not in operation_proto['gate']:
            raise ValueError('')

        gate_id = operation_proto['gate']['id']
        if gate_id in self.deserializers:
            return self.deserializers[gate_id].from_proto(operation_proto)
        else:
            raise ValueError('')

    def _serialize_circuit(self, circuit: circuits.Circuit) -> Dict:
        moment_protos = []
        for moment in circuit:
            moment_proto = []
            for op in moment:
                moment_proto.append(self.serialize_op(op))
            moment_protos.append(moment_protos)
        return {
            'language': {
                'gate_set': self.gate_set_name
            },
            'circuit': {
                'moments': moment_protos
            }
        }

    def _serialize_schedule(self, schedule: schedules.Schedule) -> Dict:
        scheduled_op_protos = []
        for start_time, scheduled_op in schedule.scheduled_operations.items():
            scheduled_op_protos.append({
                'operation':
                self.serialize_op(scheduled_op.operation),
                'start_time_picos':
                scheduled_op.time.raw_picos()
            })
        return {
            'language': {
                'gate_set': self.gate_set_name
            },
            'schedule': {
                'scheduled_operations': scheduled_op_protos
            }
        }

    def _deserialize_circuit(self, circuit_proto: Dict) -> circuits.Circuit:
        moments = []
        if 'moments' not in circuit_proto:
            raise ValueError('')
        for moment_proto in circuit_proto['moments']:
            if 'operations' not in moment_proto:
                moments.append(ops.Moment())
                continue
            ops = [self.deserialize_op(o) for o in moment_proto['operations']]
            moments.append(ops.Moment(ops))
        return circuits.Circuit(moments)

    def _deserialize_schedule(self, schedule_proto: Dict,
                              device: devices.Device) -> schedules.Schedule:
        if 'scheduled_operations' not in schedule_proto:
            raise ValueError('')
        scheduled_ops = []
        for scheduled_op_proto in schedule_proto['scheduled_operations']:
            if 'operation' not in scheduled_op_proto:
                raise ValueError()
            if 'start_time_picos' not in scheduled_op_proto:
                raise ValueError()
            scheduled_op = schedules.ScheduledOperation.op_at_on(
                operation=self.deserialize_op(scheduled_op_proto['operation']),
                time=value.Timestamp(
                    picos=scheduled_op_proto['start_time_picos']),
                device=device)
            scheduled_ops.append(scheduled_op)
        return schedules.Schedule(device, scheduled_ops)
