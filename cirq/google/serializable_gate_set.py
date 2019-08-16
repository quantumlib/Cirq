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

from collections import defaultdict

from typing import cast, Dict, Iterable, List, Optional, Tuple, Type, Union

from google.protobuf import json_format

from cirq import circuits, devices, ops, schedules, value
from cirq.api.google import v2
from cirq.google import op_deserializer, op_serializer


class SerializableGateSet:
    """A class for serializing and deserializing programs and operations.

    This class is for cirq.api.google.v2. protos.
    """

    def __init__(self, gate_set_name: str,
                 serializers: Iterable[op_serializer.GateOpSerializer],
                 deserializers: Iterable[op_deserializer.GateOpDeserializer]):
        """Construct the gate set.

        Args:
            gate_set_name: The name used to identify the gate set.
            serializers: The GateOpSerializers to use for serialization.
                Multiple serializers for a given gate type are allowed and
                will be checked for a given type in the order specified here.
                This allows for a given gate type to be serialized into
                different serialized form depending on the parameters of the
                gate.
            deserializers: The GateOpDeserializers to convert serialized
                forms of gates to GateOperations.
        """
        self.gate_set_name = gate_set_name
        self.serializers = defaultdict(
            list)  # type: Dict[Type, List[op_serializer.GateOpSerializer]]
        for s in serializers:
            self.serializers[s.gate_type].append(s)
        self.deserializers = {d.serialized_gate_id: d for d in deserializers}

    def supported_gate_types(self) -> Tuple:
        return tuple(self.serializers.keys())

    def is_supported_gate(self, gate: ops.Gate) -> bool:
        """Whether or not the given gate can be serialized by this gate set."""
        for gate_type_mro in type(gate).mro():
            if gate_type_mro in self.serializers:
                for serializer in self.serializers[gate_type_mro]:
                    if serializer.can_serialize_gate(gate):
                        return True
        return False

    def serialize_dict(self,
                       program: Union[circuits.Circuit, schedules.Schedule]
                      ) -> Dict:
        """Serialize a Circuit or Schedule to cirq.api.google.v2.Program proto.

        Args:
            program: The Circuit or Schedule to serialize.

        Returns:
            A dictionary corresponding to the cirq.api.google.v2.Program proto.
        """
        return json_format.MessageToDict(self.serialize(program),
                                         including_default_value_fields=True,
                                         preserving_proto_field_name=True,
                                         use_integers_for_enums=True)

    def serialize(self,
                  program: Union[circuits.Circuit, schedules.Schedule],
                  msg: Optional[v2.program_pb2.Program] = None
                 ) -> v2.program_pb2.Program:
        """Serialize a Circuit or Schedule to cirq.api.google.v2.Program proto.

        Args:
            program: The Circuit or Schedule to serialize.
        """
        if msg is None:
            msg = v2.program_pb2.Program()
        msg.language.gate_set = self.gate_set_name
        if isinstance(program, circuits.Circuit):
            self._serialize_circuit(program, msg.circuit)
        else:
            self._serialize_schedule(program, msg.schedule)
        return msg

    def serialize_op_dict(self, op: ops.Operation) -> Dict:
        """Serialize an Operation to cirq.api.google.v2.Operation proto.

        Args:
            op: The operation to serialize.

        Returns:
            A dictionary corresponds to the cirq.api.google.v2.Operation proto.
        """
        return json_format.MessageToDict(self.serialize_op(op),
                                         including_default_value_fields=True,
                                         preserving_proto_field_name=True,
                                         use_integers_for_enums=True)

    def serialize_op(self,
                     op: ops.Operation,
                     msg: Optional[v2.program_pb2.Operation] = None
                    ) -> v2.program_pb2.Operation:
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
                # Check each serializer in turn, if serializer proto returns
                # None, then skip.
                for serializer in self.serializers[gate_type_mro]:
                    proto_msg = serializer.to_proto(gate_op, msg)
                    if proto_msg is not None:
                        return proto_msg
        raise ValueError('Cannot serialize op {!r} of type {}'.format(
            gate_op, gate_type))

    def deserialize_dict(self,
                         proto: Dict,
                         device: Optional[devices.Device] = None
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
        msg = v2.program_pb2.Program()
        json_format.ParseDict(proto, msg)
        return self.deserialize(msg, device)

    def deserialize(self,
                    proto: v2.program_pb2.Program,
                    device: Optional[devices.Device] = None
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
        if not proto.HasField('language') or not proto.language.gate_set:
            raise ValueError('Missing gate set specification.')
        if proto.language.gate_set != self.gate_set_name:
            raise ValueError('Gate set in proto was {} but expected {}'.format(
                proto.language.gate_set, self.gate_set_name))
        which = proto.WhichOneof('program')
        if which == 'circuit':
            circuit = self._deserialize_circuit(proto.circuit)
            return circuit if device is None else circuit.with_device(device)
        if which == 'schedule':
            if device is None:
                raise ValueError(
                    'Deserializing schedule requires a device but None was '
                    'given.')
            return self._deserialize_schedule(proto.schedule, device)

        raise ValueError(
            'Program proto does not contain a circuit or schedule.')

    def deserialize_op_dict(self, operation_proto: Dict) -> ops.Operation:
        """Deserialize an Operation from a cirq.api.google.v2.Operation.

        Args:
            operation_proto: A dictionary representing a
                cirq.api.google.v2.Operation proto.

        Returns:
            The deserialized Operation.
        """
        msg = v2.program_pb2.Operation()
        json_format.ParseDict(operation_proto, msg)
        return self.deserialize_op(msg)

    def deserialize_op(self, operation_proto: v2.program_pb2.Operation
                      ) -> ops.Operation:
        """Deserialize an Operation from a cirq.api.google.v2.Operation.

        Args:
            operation_proto: A dictionary representing a
                cirq.api.google.v2.Operation proto.

        Returns:
            The deserialized Operation.
        """
        if not operation_proto.gate.id:
            raise ValueError('Operation proto does not have a gate.')

        gate_id = operation_proto.gate.id
        if gate_id not in self.deserializers.keys():
            raise ValueError(
                'Unsupported serialized gate with id {}'.format(gate_id))

        return self.deserializers[gate_id].from_proto(operation_proto)

    def _serialize_circuit(self, circuit: circuits.Circuit,
                           msg: v2.program_pb2.Circuit) -> None:
        msg.scheduling_strategy = v2.program_pb2.Circuit.MOMENT_BY_MOMENT
        for moment in circuit:
            moment_proto = msg.moments.add()
            for op in moment:
                self.serialize_op(op, moment_proto.operations.add())

    def _serialize_schedule(self, schedule: schedules.Schedule,
                            msg: v2.program_pb2.Schedule) -> None:
        for scheduled_op in schedule.scheduled_operations:
            scheduled_op_proto = msg.scheduled_operations.add()
            scheduled_op_proto.start_time_picos = scheduled_op.time.raw_picos()
            self.serialize_op(scheduled_op.operation,
                              scheduled_op_proto.operation)

    def _deserialize_circuit(self, circuit_proto: v2.program_pb2.Circuit
                            ) -> circuits.Circuit:
        moments = []
        for moment_proto in circuit_proto.moments:
            moment_ops = [
                self.deserialize_op(o) for o in moment_proto.operations
            ]
            moments.append(ops.Moment(moment_ops))
        return circuits.Circuit(moments)

    def _deserialize_schedule(self, schedule_proto: v2.program_pb2.Schedule,
                              device: devices.Device) -> schedules.Schedule:
        scheduled_ops = []
        for scheduled_op_proto in schedule_proto.scheduled_operations:
            if not scheduled_op_proto.HasField('operation'):
                raise ValueError('Scheduled op missing an operation {}'.format(
                    scheduled_op_proto))
            scheduled_op = schedules.ScheduledOperation.op_at_on(
                operation=self.deserialize_op(scheduled_op_proto.operation),
                time=value.Timestamp(picos=scheduled_op_proto.start_time_picos),
                device=device)
            scheduled_ops.append(scheduled_op)
        return schedules.Schedule(device, scheduled_ops)
