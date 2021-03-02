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
import datetime

from google.protobuf.timestamp_pb2 import Timestamp
import cirq
import cirq.google as cg

from cirq.google.engine.client.quantum import types as qtypes
from cirq.google.engine.client.quantum import enums


def test_timeslot_equality():
    start = datetime.datetime.fromtimestamp(1582592400)
    end = datetime.datetime.fromtimestamp(1582596000)
    eq = cirq.testing.equals_tester.EqualsTester()
    eq.add_equality_group(
        cg.EngineTimeSlot(processor_id='raining', start_time=start, end_time=end),
        cg.EngineTimeSlot(processor_id='raining', start_time=start, end_time=end),
        cg.EngineTimeSlot(
            processor_id='raining',
            start_time=start,
            end_time=end,
            slot_type=enums.QuantumTimeSlot.TimeSlotType.TIME_SLOT_TYPE_UNSPECIFIED,
        ),
    )
    eq.add_equality_group(
        cg.EngineTimeSlot(
            processor_id='raining', start_time=start, end_time=end, project_id='123456'
        )
    )
    eq.add_equality_group(
        cg.EngineTimeSlot(
            processor_id='raining',
            start_time=start,
            end_time=end,
            slot_type=enums.QuantumTimeSlot.TimeSlotType.RESERVATION,
            project_id='123456',
        )
    )
    eq.add_equality_group(
        cg.EngineTimeSlot(
            processor_id='raining',
            start_time=start,
            end_time=end,
            slot_type=enums.QuantumTimeSlot.TimeSlotType.MAINTENANCE,
            project_id='123456',
        )
    )
    eq.add_equality_group(
        cg.EngineTimeSlot(
            processor_id='raining',
            start_time=start,
            end_time=end,
            slot_type=enums.QuantumTimeSlot.TimeSlotType.MAINTENANCE,
            project_id='123456',
            maintenance_title='Testing',
            maintenance_description='Testing some new configuration.',
        )
    )


def test_from_to_proto_plain():
    slot = enums.QuantumTimeSlot.TimeSlotType.RESERVATION
    proto = qtypes.QuantumTimeSlot(
        processor_name='potofgold',
        start_time=Timestamp(seconds=1500000000),
        end_time=Timestamp(seconds=1500010000),
        slot_type=slot,
    )
    time_slot = cg.EngineTimeSlot(
        processor_id='potofgold',
        start_time=datetime.datetime.fromtimestamp(1500000000),
        end_time=datetime.datetime.fromtimestamp(1500010000),
        slot_type=slot,
    )
    actual_from_proto = cg.EngineTimeSlot.from_proto(proto)
    assert actual_from_proto == time_slot
    actual_to_proto = cg.EngineTimeSlot.to_proto(time_slot)
    assert actual_to_proto == proto


def test_from_to_proto_reservation():
    slot = enums.QuantumTimeSlot.TimeSlotType.RESERVATION
    proto = qtypes.QuantumTimeSlot(
        processor_name='potofgold',
        start_time=Timestamp(seconds=1500000000),
        end_time=Timestamp(seconds=1500010000),
        slot_type=slot,
        reservation_config=qtypes.QuantumTimeSlot.ReservationConfig(
            project_id='super_secret_quantum'
        ),
    )
    time_slot = cg.EngineTimeSlot(
        processor_id='potofgold',
        start_time=datetime.datetime.fromtimestamp(1500000000),
        end_time=datetime.datetime.fromtimestamp(1500010000),
        slot_type=slot,
        project_id='super_secret_quantum',
    )
    actual_from_proto = cg.EngineTimeSlot.from_proto(proto)
    assert actual_from_proto == time_slot
    actual_to_proto = cg.EngineTimeSlot.to_proto(time_slot)
    assert actual_to_proto == proto


def test_from_to_proto_maintenance():
    slot = enums.QuantumTimeSlot.TimeSlotType.MAINTENANCE
    proto = qtypes.QuantumTimeSlot(
        processor_name='potofgold',
        start_time=Timestamp(seconds=1500020000),
        end_time=Timestamp(seconds=1500040000),
        slot_type=slot,
        maintenance_config=qtypes.QuantumTimeSlot.MaintenanceConfig(
            title='Testing',
            description='Testing some new configuration.',
        ),
    )
    time_slot = cg.EngineTimeSlot(
        processor_id='potofgold',
        start_time=datetime.datetime.fromtimestamp(1500020000),
        end_time=datetime.datetime.fromtimestamp(1500040000),
        slot_type=slot,
        maintenance_title='Testing',
        maintenance_description='Testing some new configuration.',
    )
    actual_from_proto = cg.EngineTimeSlot.from_proto(proto)
    assert actual_from_proto == time_slot
    actual_to_proto = cg.EngineTimeSlot.to_proto(time_slot)
    assert actual_to_proto == proto


def test_from_to_proto_no_end_time():
    slot = enums.QuantumTimeSlot.TimeSlotType.MAINTENANCE
    proto = qtypes.QuantumTimeSlot(
        processor_name='potofgold',
        end_time=Timestamp(seconds=1500040000),
        slot_type=slot,
        maintenance_config=qtypes.QuantumTimeSlot.MaintenanceConfig(
            title='Testing',
            description='Testing some new configuration.',
        ),
    )
    time_slot = cg.EngineTimeSlot(
        processor_id='potofgold',
        end_time=datetime.datetime.fromtimestamp(1500040000),
        slot_type=slot,
        maintenance_title='Testing',
        maintenance_description='Testing some new configuration.',
    )
    actual_from_proto = cg.EngineTimeSlot.from_proto(proto)
    assert actual_from_proto == time_slot
    actual_to_proto = cg.EngineTimeSlot.to_proto(time_slot)
    assert actual_to_proto == proto


def test_from_to_proto_no_start_time():
    slot = enums.QuantumTimeSlot.TimeSlotType.MAINTENANCE
    proto = qtypes.QuantumTimeSlot(
        processor_name='potofgold',
        start_time=Timestamp(seconds=1500040000),
        slot_type=slot,
        maintenance_config=qtypes.QuantumTimeSlot.MaintenanceConfig(
            title='Testing',
            description='Testing some new configuration.',
        ),
    )
    time_slot = cg.EngineTimeSlot(
        processor_id='potofgold',
        start_time=datetime.datetime.fromtimestamp(1500040000),
        slot_type=slot,
        maintenance_title='Testing',
        maintenance_description='Testing some new configuration.',
    )
    actual_from_proto = cg.EngineTimeSlot.from_proto(proto)
    assert actual_from_proto == time_slot
    actual_to_proto = cg.EngineTimeSlot.to_proto(time_slot)
    assert actual_to_proto == proto
