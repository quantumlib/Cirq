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

from cirq_google.engine.client.quantum import types as qtypes
from cirq_google.engine.client.quantum import enums as qenums
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

    def run(self, *args, **kwargs):
        pass

    def run_batch(self, *args, **kwargs):
        pass

    def run_calibration(self, *args, **kwargs):
        pass

    def run_sweep(self, *args, **kwargs):
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
        _ = p.create_reservation(
            start_time=_time(200000),
            end_time=_time(100000),
        )


def test_reservations():
    p = NothingProcessor(processor_id='test')
    start_reservation = datetime.datetime.now()
    end_reservation = datetime.datetime.now() + datetime.timedelta(hours=2)
    users = ['dstrain@google.com']

    # Create Reservation
    reservation = p.create_reservation(
        start_time=start_reservation, end_time=end_reservation, whitelisted_users=users
    )
    assert reservation.start_time.seconds == int(start_reservation.timestamp())
    assert reservation.end_time.seconds == int(end_reservation.timestamp())
    assert reservation.whitelisted_users == users

    # Get Reservation
    assert p.get_reservation(reservation.name) == reservation
    assert p.get_reservation('nothing_to_see_here') is None

    # Update reservation
    end_reservation = datetime.datetime.now() + datetime.timedelta(hours=3)
    p.update_reservation(reservation_id=reservation.name, end_time=end_reservation)
    reservation = p.get_reservation(reservation.name)
    assert reservation.end_time.seconds == int(end_reservation.timestamp())
    start_reservation = datetime.datetime.now() + datetime.timedelta(hours=1)
    p.update_reservation(reservation_id=reservation.name, start_time=start_reservation)
    reservation = p.get_reservation(reservation.name)
    assert reservation.start_time.seconds == int(start_reservation.timestamp())
    users = ['dstrain@google.com', 'dabacon@google.com']
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
    assert p.list_reservations(now - 0.5 * hour, now + 3 * hour) == [reservation2, reservation3]
    assert p.list_reservations(now + 0.5 * hour, now + 3 * hour) == [reservation3]
    assert p.list_reservations(now - 0.5 * hour, now + 1.5 * hour) == [reservation2]
    assert p.list_reservations(now - 1.5 * hour, now + 1.5 * hour) == [reservation1, reservation2]

    assert p.list_reservations(-0.5 * hour, 3 * hour) == [reservation2, reservation3]
    assert p.list_reservations(0.5 * hour, 3 * hour) == [reservation3]
    assert p.list_reservations(-0.5 * hour, 1.5 * hour) == [reservation2]
    assert p.list_reservations(-1.5 * hour, 1.5 * hour) == [reservation1, reservation2]

    assert p.list_reservations(now - 2 * hour, None) == [reservation1, reservation2, reservation3]

    p.remove_reservation(reservation1.name)
    assert p.list_reservations(now - 2 * hour, None) == [reservation2, reservation3]


def test_bad_schedule():
    time_slot1 = qtypes.QuantumTimeSlot(
        processor_name='test',
        start_time=Timestamp(seconds=100000),
        end_time=Timestamp(seconds=300000),
        slot_type=qenums.QuantumTimeSlot.TimeSlotType.OPEN_SWIM,
    )
    time_slot2 = qtypes.QuantumTimeSlot(
        processor_name='test',
        start_time=Timestamp(seconds=200000),
        end_time=Timestamp(seconds=400000),
        slot_type=qenums.QuantumTimeSlot.TimeSlotType.OPEN_SWIM,
    )
    with pytest.raises(ValueError, match='cannot overlap'):
        _ = NothingProcessor(processor_id='test', schedule=[time_slot1, time_slot2])


def test_get_schedule():
    time_slot = qtypes.QuantumTimeSlot(
        processor_name='test',
        start_time=Timestamp(seconds=100000),
        end_time=Timestamp(seconds=200000),
        slot_type=qenums.QuantumTimeSlot.TimeSlotType.OPEN_SWIM,
    )
    p = NothingProcessor(processor_id='test', schedule=[time_slot])
    assert p.get_schedule(from_time=_time(50000), to_time=_time(250000)) == [time_slot]
    assert p.get_schedule(from_time=_time(150000), to_time=_time(250000)) == [time_slot]
    assert p.get_schedule(from_time=_time(50000), to_time=_time(150000)) == [time_slot]
    assert p.get_schedule(from_time=_time(50000), to_time=_time(75000)) == []
    assert p.get_schedule(from_time=_time(250000), to_time=_time(300000)) == []
    # check unbounded cases
    unbounded_start = qtypes.QuantumTimeSlot(
        processor_name='test',
        end_time=Timestamp(seconds=100000),
        slot_type=qenums.QuantumTimeSlot.TimeSlotType.OPEN_SWIM,
    )
    unbounded_end = qtypes.QuantumTimeSlot(
        processor_name='test',
        start_time=Timestamp(seconds=200000),
        slot_type=qenums.QuantumTimeSlot.TimeSlotType.OPEN_SWIM,
    )
    p = NothingProcessor(processor_id='test', schedule=[unbounded_start, unbounded_end])
    assert (
        p.get_schedule(
            from_time=_time(50000),
            to_time=_time(250000),
        )
        == [unbounded_start, unbounded_end]
    )
    assert (
        p.get_schedule(
            from_time=_time(150000),
            to_time=_time(250000),
        )
        == [unbounded_end]
    )
    assert (
        p.get_schedule(
            from_time=_time(50000),
            to_time=_time(150000),
        )
        == [unbounded_start]
    )
    assert (
        p.get_schedule(
            from_time=_time(120000),
            to_time=_time(150000),
        )
        == []
    )


@pytest.mark.parametrize(
    ('time_slot'),
    (
        qtypes.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=100000),
            end_time=Timestamp(seconds=200000),
            slot_type=qenums.QuantumTimeSlot.TimeSlotType.OPEN_SWIM,
        ),
        qtypes.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=100000),
            slot_type=qenums.QuantumTimeSlot.TimeSlotType.OPEN_SWIM,
        ),
        qtypes.QuantumTimeSlot(
            processor_name='test',
            end_time=Timestamp(seconds=200000),
            slot_type=qenums.QuantumTimeSlot.TimeSlotType.OPEN_SWIM,
        ),
    ),
)
def test_create_reservation_not_available(time_slot):
    p = NothingProcessor(processor_id='test', schedule=[time_slot])
    with pytest.raises(ValueError, match='Time slot is not available for reservations'):
        p.create_reservation(
            start_time=_time(50000),
            end_time=_time(150000),
        )


def test_create_reservation_open_time_slots():
    time_slot = qtypes.QuantumTimeSlot(
        processor_name='test',
        start_time=Timestamp(seconds=100000),
        end_time=Timestamp(seconds=200000),
        slot_type=qenums.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
    )
    p = NothingProcessor(processor_id='test', schedule=[time_slot])
    p.create_reservation(
        start_time=_time(50000),
        end_time=_time(150000),
    )
    assert p.get_schedule(from_time=_time(0), to_time=_time(250000)) == [
        qtypes.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=50000),
            end_time=Timestamp(seconds=150000),
            slot_type=qenums.QuantumTimeSlot.TimeSlotType.RESERVATION,
        ),
        qtypes.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=150000),
            end_time=Timestamp(seconds=200000),
            slot_type=qenums.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
        ),
    ]


def test_create_reservation_split_time_slots():
    time_slot = qtypes.QuantumTimeSlot(
        processor_name='test',
        start_time=Timestamp(seconds=100000),
        end_time=Timestamp(seconds=200000),
        slot_type=qenums.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
    )
    p = NothingProcessor(processor_id='test', schedule=[time_slot])
    p.create_reservation(
        start_time=_time(120000),
        end_time=_time(150000),
    )
    assert p.get_schedule(from_time=_time(0), to_time=_time(250000)) == [
        qtypes.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=100000),
            end_time=Timestamp(seconds=120000),
            slot_type=qenums.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
        ),
        qtypes.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=120000),
            end_time=Timestamp(seconds=150000),
            slot_type=qenums.QuantumTimeSlot.TimeSlotType.RESERVATION,
        ),
        qtypes.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=150000),
            end_time=Timestamp(seconds=200000),
            slot_type=qenums.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
        ),
    ]


def test_create_reservation_add_at_end():
    time_slot = qtypes.QuantumTimeSlot(
        processor_name='test',
        start_time=Timestamp(seconds=100000),
        end_time=Timestamp(seconds=200000),
        slot_type=qenums.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
    )
    p = NothingProcessor(processor_id='test', schedule=[time_slot])
    p.create_reservation(
        start_time=_time(250000),
        end_time=_time(350000),
    )
    assert p.get_schedule(from_time=_time(50000), to_time=_time(250000)) == [
        qtypes.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=100000),
            end_time=Timestamp(seconds=200000),
            slot_type=qenums.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
        ),
        qtypes.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=250000),
            end_time=Timestamp(seconds=350000),
            slot_type=qenums.QuantumTimeSlot.TimeSlotType.RESERVATION,
        ),
    ]


def test_create_reservation_border_conditions():
    time_slot = qtypes.QuantumTimeSlot(
        processor_name='test',
        start_time=Timestamp(seconds=100000),
        end_time=Timestamp(seconds=200000),
        slot_type=qenums.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
    )
    p = NothingProcessor(processor_id='test', schedule=[time_slot])
    p.create_reservation(
        start_time=_time(190000),
        end_time=_time(200000),
    )
    p.create_reservation(
        start_time=_time(100000),
        end_time=_time(110000),
    )
    assert p.get_schedule(from_time=_time(0), to_time=_time(250000)) == [
        qtypes.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=100000),
            end_time=Timestamp(seconds=110000),
            slot_type=qenums.QuantumTimeSlot.TimeSlotType.RESERVATION,
        ),
        qtypes.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=110000),
            end_time=Timestamp(seconds=190000),
            slot_type=qenums.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
        ),
        qtypes.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=190000),
            end_time=Timestamp(seconds=200000),
            slot_type=qenums.QuantumTimeSlot.TimeSlotType.RESERVATION,
        ),
    ]


def test_create_reservation_unbounded():
    time_slot_begin = qtypes.QuantumTimeSlot(
        processor_name='test',
        end_time=Timestamp(seconds=200000),
        slot_type=qenums.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
    )
    time_slot_end = qtypes.QuantumTimeSlot(
        processor_name='test',
        start_time=Timestamp(seconds=500000),
        slot_type=qenums.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
    )
    p = NothingProcessor(processor_id='test', schedule=[time_slot_begin, time_slot_end])
    p.create_reservation(
        start_time=_time(100000),
        end_time=_time(300000),
    )
    p.create_reservation(
        start_time=_time(400000),
        end_time=_time(600000),
    )
    assert p.get_schedule(from_time=_time(0), to_time=_time(1000000)) == [
        qtypes.QuantumTimeSlot(
            processor_name='test',
            end_time=Timestamp(seconds=100000),
            slot_type=qenums.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
        ),
        qtypes.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=100000),
            end_time=Timestamp(seconds=300000),
            slot_type=qenums.QuantumTimeSlot.TimeSlotType.RESERVATION,
        ),
        qtypes.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=400000),
            end_time=Timestamp(seconds=600000),
            slot_type=qenums.QuantumTimeSlot.TimeSlotType.RESERVATION,
        ),
        qtypes.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=600000),
            slot_type=qenums.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
        ),
    ]


def test_create_reservation_splitunbounded():
    time_slot_begin = qtypes.QuantumTimeSlot(
        processor_name='test',
        end_time=Timestamp(seconds=300000),
        slot_type=qenums.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
    )
    time_slot_end = qtypes.QuantumTimeSlot(
        processor_name='test',
        start_time=Timestamp(seconds=500000),
        slot_type=qenums.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
    )
    p = NothingProcessor(processor_id='test', schedule=[time_slot_begin, time_slot_end])
    p.create_reservation(
        start_time=_time(100000),
        end_time=_time(200000),
    )
    p.create_reservation(
        start_time=_time(600000),
        end_time=_time(700000),
    )
    assert p.get_schedule(from_time=_time(0), to_time=_time(1000000)) == [
        qtypes.QuantumTimeSlot(
            processor_name='test',
            end_time=Timestamp(seconds=100000),
            slot_type=qenums.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
        ),
        qtypes.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=100000),
            end_time=Timestamp(seconds=200000),
            slot_type=qenums.QuantumTimeSlot.TimeSlotType.RESERVATION,
        ),
        qtypes.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=200000),
            end_time=Timestamp(seconds=300000),
            slot_type=qenums.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
        ),
        qtypes.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=500000),
            end_time=Timestamp(seconds=600000),
            slot_type=qenums.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
        ),
        qtypes.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=600000),
            end_time=Timestamp(seconds=700000),
            slot_type=qenums.QuantumTimeSlot.TimeSlotType.RESERVATION,
        ),
        qtypes.QuantumTimeSlot(
            processor_name='test',
            start_time=Timestamp(seconds=700000),
            slot_type=qenums.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
        ),
    ]
