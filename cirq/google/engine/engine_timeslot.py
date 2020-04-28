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
import dataclasses
import datetime
from typing import Optional
from google.protobuf import timestamp_pb2
from cirq.google.engine.client.quantum import enums as qenums
from cirq.google.engine.client.quantum import types as qtypes

_DEFAULT_TYPE = qenums.QuantumTimeSlot.TimeSlotType.TIME_SLOT_TYPE_UNSPECIFIED


def _to_timestamp(dt: datetime.datetime):
    return timestamp_pb2.Timestamp(seconds=int(dt.timestamp()))


@dataclasses.dataclass(frozen=True)
class EngineTimeSlot:
    """A python wrapping of a Quantum Engine timeslot.

    Args:
       processor_id: The processor whose schedule the time slot exists on.
       start_time: starting datetime of the time slot, usually in local time.
       end_time: ending datetime of the time slot, usually in local time.
       slot_type: type of time slot (reservation, open swim, etc)
       project_id: Google Cloud Platform id of the project, as a string
       maintenance_title: If a MAINTENANCE period, a string title describing the
          type of maintenance being done.
       maintenance_description: If a MAINTENANCE period, a string describing the
          particulars of the maintenancethe title of the slot
    """
    processor_id: str
    start_time: datetime.datetime
    end_time: datetime.datetime
    slot_type: qenums.QuantumTimeSlot.TimeSlotType = _DEFAULT_TYPE
    project_id: Optional[str] = None
    maintenance_title: Optional[str] = None
    maintenance_description: Optional[str] = None

    @classmethod
    def from_proto(cls, proto: qtypes.QuantumTimeSlot):
        slot_type = qenums.QuantumTimeSlot.TimeSlotType(proto.slot_type)
        if proto.HasField('reservation_config'):
            return cls(processor_id=proto.processor_name,
                       start_time=datetime.datetime.fromtimestamp(
                           proto.start_time.seconds),
                       end_time=datetime.datetime.fromtimestamp(
                           proto.end_time.seconds),
                       slot_type=slot_type,
                       project_id=proto.reservation_config.project_id)
        if proto.HasField('maintenance_config'):
            return cls(
                processor_id=proto.processor_name,
                start_time=datetime.datetime.fromtimestamp(
                    proto.start_time.seconds),
                end_time=datetime.datetime.fromtimestamp(
                    proto.end_time.seconds),
                slot_type=slot_type,
                maintenance_title=proto.maintenance_config.title,
                maintenance_description=proto.maintenance_config.description)
        return cls(processor_id=proto.processor_name,
                   start_time=datetime.datetime.fromtimestamp(
                       proto.start_time.seconds),
                   end_time=datetime.datetime.fromtimestamp(
                       proto.end_time.seconds),
                   slot_type=slot_type)

    def to_proto(self):
        time_slot = qtypes.QuantumTimeSlot(
            processor_name=self.processor_id,
            start_time=_to_timestamp(self.start_time),
            end_time=_to_timestamp(self.end_time),
            slot_type=self.slot_type,
        )
        if self.project_id:
            time_slot.reservation_config.project_id = self.project_id
        config = time_slot.maintenance_config
        if self.maintenance_title:
            config.title = self.maintenance_title
        if self.maintenance_description:
            config.description = self.maintenance_description
        return time_slot
