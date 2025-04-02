# Copyright 2021 The Cirq Developers
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

import pytest
from google.protobuf.timestamp_pb2 import Timestamp

from cirq_google.cloud import quantum
from cirq_google.engine.abstract_local_processor import AbstractLocalProcessor


def _time(seconds_from_epoch: int):
    """Shorthand to abbreviate datetimes from epochs."""
    return datetime.datetime.fromtimestamp(seconds_from_epoch)


class NothingProcessor(AbstractLocalProcessor):
    """A processor for testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_calibration(self, *args, **kwargs):
        pass

    def get_latest_calibration(self, *args, **kwargs):
        pass

    def get_current_calibration(self, *args, **kwargs):
        pass

    def get_device(self, *args, **kwargs):
        pass

    def get_device_specification(self, *args, **kwargs):
        pass

    def health(self, *args, **kwargs):
        pass

    def list_calibrations(self, *args, **kwargs):
        pass

    async def run_sweep_async(self, *args, **kwargs):
        pass

    def get_sampler(self, *args, **kwargs):
        pass

    def supported_languages(self, *args, **kwargs):
        pass

    def list_programs(self, *args, **kwargs):
        pass

    def get_program(self, *args, **kwargs):
        pass


def test_datetime():
    recovery_time = datetime.datetime.now()
    down_time = datetime.datetime.now() - datetime.timedelta(hours=2)

    p = NothingProcessor(
        processor_id='test', expected_down_time=down_time, expected_recovery_time=recovery_time
    )
    assert p.expected_down_time() == down_time
    assert p.expected_recovery_time() == recovery_time


def test_bad_reservation():
    p = NothingProcessor(processor_id='test')
    with pytest.raises(ValueError, match='after the start time'):
        _ = p.create_reservation(start_time=_time(2000000), end_time=_time(1000000))


def test_reservations():
    p = NothingProcessor(processor_id='test')
    start_reservation = datetime.datetime.now()
    end_reservation = datetime.datetime.now() + datetime.timedelta(hours=2)
    users = ['gooduser@test.com']

    # Create Reservation
    reservation = p.create_reservation(
        start_time=start_reservation, end_time=end_reservation, whitelisted_users=users
    )
    assert reservation.start_time.timestamp() == int(start_reservation.timestamp())
    assert reservation.end_time.timestamp() == int(end_reservation.timestamp())
    assert reservation.whitelisted_users == users

    # Get Reservation
    assert p.get_reservation(reservation.name) == reservation
    assert p.get_reservation('nothing_to_see_here') is None

    # Update reservation
    end_reservation = datetime.datetime.now() + datetime.timedelta(hours=3)
    p.update_reservation(reservation_id=reservation.name, end_time=end_reservation)
    reservation = p.get_reservation(reservation.name)
    assert reservation.end_time.timestamp() == int(end_reservation.timestamp())
    start_reservation = datetime.datetime.now() + datetime.timedelta(hours=1)
    p.update_reservation(reservation_id=reservation.name, start_time=start_reservation)
    reservation = p.get_reservation(reservation.name)
    assert reservation.start_time.timestamp() == int(start_reservation.timestamp())
    users = ['gooduser@test.com', 'otheruser@prod.com']
    p.update_reservation(reservation_id=reservation.name, whitelisted_users=users)
    reservation = p.get_reservation(reservation.name)
    assert reservation.whitelisted_users == users

    with pytest.raises(ValueError, match='does not exist'):
        p.update_reservation(reservation_id='invalid', whitelisted_users=users)


def test_list_reservations():
    p = NothingProcessor(processor_id='test')
    now = datetime.datetime.now()
    hour = datetime.timedelta(hours=1)
    users = ['abc@def.com']

    reservation1 = p.create_reservation(
        start_time=now - hour, end_time=now, whitelisted_users=users
    )
    reservation2 = p.create_reservation(
        start_time=now, end_time=now + hour, whitelisted_users=users
    )
    reservation3 = p.create_reservation(
        start_time=now + hour, end_time=now + 2 * hour, whitelisted_users=users
    )

    assert p.list_reservations(now - 2 * hour, now + 3 * hour) == [
        reservation1,
        reservation2,
        reservation3,
    ]
    assert p.list_reservations(now + 0.5 * hour, now + 3 * hour) == [reservation2, reservation3]
    assert p.list_reservations(now + 1.5 * hour, now + 3 * hour) == [reservation3]
    assert p.list_reservations(now + 0.5 * hour, now + 0.75 * hour) == [reservation2]
    assert p.list_reservations(now - 1.5 * hour, now + 0.5 * hour) == [reservation1, reservation2]

    assert p.list_reservations(0.5 * hour, 3 * hour) == [reservation2, reservation3]
    assert p.list_reservations(1.5 * hour, 3 * hour) == [reservation3]
    assert p.list_reservations(0.25 * hour, 0.5 * hour) == [reservation2]
    assert p.list_reservations(-1.5 * hour, 0.5 * hour) == [reservation1, reservation2]

    assert p.list_reservations(now - 2 * hour, None) == [reservation1, reservation2, reservation3]

    p.remove_reservation(reservation1.name)
    assert p.list_reservations(now - 2 * hour, None) == [reservation2, reservation3]


def test_bad_schedule():
    time_slot1 = quantum.QuantumTimeSlot(
        processor_name='test',
        start_time=Timestamp(seconds=1000000),
        end_time=Timestamp(seconds=3000000),
        time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.OPEN_SWIM,
    )
    time_slot2 = quantum.QuantumTimeSlot(
        processor_name='test',
        start_time=Timestamp(seconds=2000000),
        end_time=Timestamp(seconds=4000000),
        time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.OPEN_SWIM,
    )
    with pytest.raises(ValueError, match='cannot overlap'):
        _ = NothingProcessor(processor_id='test', schedule=[time_slot1, time_slot2])


def test_get_schedule():
    time_slot = quantum.QuantumTimeSlot(
        processor_name='test',
        start_time=Timestamp(seconds=1000000),
        end_time=Timestamp(seconds=2000000),
        time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.OPEN_SWIM,
    )
    p = NothingProcessor(processor_id='test', schedule=[time_slot])
    assert p.get_schedule(from_time=_time(500000), to_time=_time(2500000)) == [time_slot]
    assert p.get_schedule(from_time=_time(1500000), to_time=_time(2500000)) == [time_slot]
    assert p.get_schedule(from_time=_time(500000), to_time=_time(1500000)) == [time_slot]
    assert p.get_schedule(from_time=_time(500000), to_time=_time(750000)) == []
    assert p.get_schedule(from_time=_time(2500000), to_time=_time(300000)) == []
    # check unbounded cases
    unbounded_start = quantum.QuantumTimeSlot(
        processor_name='test',
        end_time=Timestamp(seconds=1000000),
        time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.OPEN_SWIM,
    )
    unbounded_end = quantum.QuantumTimeSlot(
        processor_name='test',
        start_time=Timestamp(seconds=2000000),
        time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.OPEN_SWIM,
    )
    p = NothingProcessor(processor_id='test', schedule=[unbounded_start, unbounded_end])
    assert p.get_schedule(from_time=_time(500000), to_time=_time(2500000)) == [
        unbounded_start,
        unbounded_end,
    ]
    assert p.get_schedule(from_time=_time(1500000), to_time=_time(2500000)) == [unbounded_end]
    assert p.get_schedule(from_time=_time(500000), to_time=_time(1500000)) == [unbounded_start]
    assert p.get_schedule(from_time=_time(1200000), to_time=_time(1500000)) == []


@pytest.mark.parametrize(
    ('time_slot'),
    (
        quantum.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=1000000),
            end_time=Timestamp(seconds=2000000),
            time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.OPEN_SWIM,
        ),
        quantum.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=1000000),
            time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.OPEN_SWIM,
        ),
        quantum.QuantumTimeSlot(
            processor_name='test',
            end_time=Timestamp(seconds=2000000),
            time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.OPEN_SWIM,
        ),
    ),
)
def test_create_reservation_not_available(time_slot):
    p = NothingProcessor(processor_id='test', schedule=[time_slot])
    with pytest.raises(ValueError, match='Time slot is not available for reservations'):
        p.create_reservation(start_time=_time(500000), end_time=_time(1500000))


def test_create_reservation_open_time_slots():
    time_slot = quantum.QuantumTimeSlot(
        processor_name='test',
        start_time=Timestamp(seconds=1000000),
        end_time=Timestamp(seconds=2000000),
        time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
    )
    p = NothingProcessor(processor_id='test', schedule=[time_slot])
    p.create_reservation(start_time=_time(500000), end_time=_time(1500000))
    assert p.get_schedule(from_time=_time(200000), to_time=_time(2500000)) == [
        quantum.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=500000),
            end_time=Timestamp(seconds=1500000),
            time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.RESERVATION,
        ),
        quantum.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=1500000),
            end_time=Timestamp(seconds=2000000),
            time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
        ),
    ]


def test_create_reservation_split_time_slots():
    time_slot = quantum.QuantumTimeSlot(
        processor_name='test',
        start_time=Timestamp(seconds=1000000),
        end_time=Timestamp(seconds=2000000),
        time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
    )
    p = NothingProcessor(processor_id='test', schedule=[time_slot])
    p.create_reservation(start_time=_time(1200000), end_time=_time(1500000))
    assert p.get_schedule(from_time=_time(200000), to_time=_time(2500000)) == [
        quantum.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=1000000),
            end_time=Timestamp(seconds=1200000),
            time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
        ),
        quantum.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=1200000),
            end_time=Timestamp(seconds=1500000),
            time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.RESERVATION,
        ),
        quantum.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=1500000),
            end_time=Timestamp(seconds=2000000),
            time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
        ),
    ]


def test_create_reservation_add_at_end():
    time_slot = quantum.QuantumTimeSlot(
        processor_name='test',
        start_time=Timestamp(seconds=1000000),
        end_time=Timestamp(seconds=2000000),
        time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
    )
    p = NothingProcessor(processor_id='test', schedule=[time_slot])
    p.create_reservation(start_time=_time(2500000), end_time=_time(3500000))
    assert p.get_schedule(from_time=_time(500000), to_time=_time(2500000)) == [
        quantum.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=1000000),
            end_time=Timestamp(seconds=2000000),
            time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
        ),
        quantum.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=2500000),
            end_time=Timestamp(seconds=3500000),
            time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.RESERVATION,
        ),
    ]


def test_create_reservation_border_conditions():
    time_slot = quantum.QuantumTimeSlot(
        processor_name='test',
        start_time=Timestamp(seconds=1000000),
        end_time=Timestamp(seconds=2000000),
        time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
    )
    p = NothingProcessor(processor_id='test', schedule=[time_slot])
    p.create_reservation(start_time=_time(1900000), end_time=_time(2000000))
    p.create_reservation(start_time=_time(1000000), end_time=_time(1100000))
    assert p.get_schedule(from_time=_time(200000), to_time=_time(2500000)) == [
        quantum.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=1000000),
            end_time=Timestamp(seconds=1100000),
            time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.RESERVATION,
        ),
        quantum.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=1100000),
            end_time=Timestamp(seconds=1900000),
            time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
        ),
        quantum.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=1900000),
            end_time=Timestamp(seconds=2000000),
            time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.RESERVATION,
        ),
    ]


def test_create_reservation_unbounded():
    time_slot_begin = quantum.QuantumTimeSlot(
        processor_name='test',
        end_time=Timestamp(seconds=2000000),
        time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
    )
    time_slot_end = quantum.QuantumTimeSlot(
        processor_name='test',
        start_time=Timestamp(seconds=5000000),
        time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
    )
    p = NothingProcessor(processor_id='test', schedule=[time_slot_begin, time_slot_end])
    p.create_reservation(start_time=_time(1000000), end_time=_time(3000000))
    p.create_reservation(start_time=_time(4000000), end_time=_time(6000000))
    assert p.get_schedule(from_time=_time(200000), to_time=_time(10000000)) == [
        quantum.QuantumTimeSlot(
            processor_name='test',
            end_time=Timestamp(seconds=1000000),
            time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
        ),
        quantum.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=1000000),
            end_time=Timestamp(seconds=3000000),
            time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.RESERVATION,
        ),
        quantum.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=4000000),
            end_time=Timestamp(seconds=6000000),
            time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.RESERVATION,
        ),
        quantum.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=6000000),
            time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
        ),
    ]


def test_create_reservation_splitunbounded():
    time_slot_begin = quantum.QuantumTimeSlot(
        processor_name='test',
        end_time=Timestamp(seconds=3000000),
        time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
    )
    time_slot_end = quantum.QuantumTimeSlot(
        processor_name='test',
        start_time=Timestamp(seconds=5000000),
        time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
    )
    p = NothingProcessor(processor_id='test', schedule=[time_slot_begin, time_slot_end])
    p.create_reservation(start_time=_time(1000000), end_time=_time(2000000))
    p.create_reservation(start_time=_time(6000000), end_time=_time(7000000))
    assert p.get_schedule(from_time=_time(200000), to_time=_time(10000000)) == [
        quantum.QuantumTimeSlot(
            processor_name='test',
            end_time=Timestamp(seconds=1000000),
            time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
        ),
        quantum.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=1000000),
            end_time=Timestamp(seconds=2000000),
            time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.RESERVATION,
        ),
        quantum.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=2000000),
            end_time=Timestamp(seconds=3000000),
            time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
        ),
        quantum.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=5000000),
            end_time=Timestamp(seconds=6000000),
            time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
        ),
        quantum.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=6000000),
            end_time=Timestamp(seconds=7000000),
            time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.RESERVATION,
        ),
        quantum.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=7000000),
            time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
        ),
    ]
