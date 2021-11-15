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
from abc import abstractmethod
import copy
import datetime

from typing import Dict, List, Optional, TYPE_CHECKING, Union
from google.protobuf.timestamp_pb2 import Timestamp

from cirq_google.engine import calibration
from cirq_google.engine.client.quantum import types as qtypes
from cirq_google.engine.client.quantum import enums as qenums
from cirq_google.engine.abstract_processor import AbstractProcessor
from cirq_google.engine.abstract_program import AbstractProgram

if TYPE_CHECKING:
    from cirq_google.engine.abstract_engine import AbstractEngine
    from cirq_google.engine.abstract_local_program import AbstractLocalProgram


def _to_timestamp(union_time: Union[None, datetime.datetime, datetime.timedelta]):
    """Translate a datetime or timedelta into a number of seconds since epoch."""
    if isinstance(union_time, datetime.timedelta):
        return int((datetime.datetime.now() + union_time).timestamp())
    elif isinstance(union_time, datetime.datetime):
        return int(union_time.timestamp())
    return None


class AbstractLocalProcessor(AbstractProcessor):
    """Partial implementation of AbstractProcessor using in-memory objects.

    This implements reservation creation and scheduling using an in-memory
    list for time slots and reservations.  Any time slot not specified by
    initialization is assumed to be UNALLOCATED (available for reservation).

    Attributes:
        processor_id: Unique string id of the processor.
        engine: The parent `AbstractEngine` object, if available.
        expected_down_time: Optional datetime of the next expected downtime.
            For informational purpose only.
        expected_recovery_time: Optional datetime when the processor is
            expected to be available again.  For informational purpose only.
        schedule:  List of time slots that the scheduling/reservation should
            use.  All time slots must be non-overlapping.
        project_name: A project_name for resource naming.
    """

    def __init__(
        self,
        *,
        processor_id: str,
        engine: Optional['AbstractEngine'] = None,
        expected_down_time: Optional[datetime.datetime] = None,
        expected_recovery_time: Optional[datetime.datetime] = None,
        schedule: Optional[List[qtypes.QuantumTimeSlot]] = None,
        project_name: str = 'fake_project',
    ):
        self._engine = engine
        self._expected_recovery_time = expected_recovery_time
        self._expected_down_time = expected_down_time
        self._reservations: Dict[str, qtypes.QuantumReservation] = {}
        self._resource_id_counter = 0
        self._processor_id = processor_id
        self._project_name = project_name

        if schedule is None:
            self._schedule = [
                qtypes.QuantumTimeSlot(
                    processor_name=self._processor_id,
                    slot_type=qenums.QuantumTimeSlot.TimeSlotType.UNALLOCATED,
                )
            ]
        else:
            self._schedule = copy.copy(schedule)
            self._schedule.sort(key=lambda t: t.start_time.seconds or -1)

        for idx in range(len(self._schedule) - 1):
            if self._schedule[idx].end_time.seconds > self._schedule[idx + 1].start_time.seconds:
                raise ValueError('Time slots cannot overlap!')

    @property
    def processor_id(self) -> str:
        """Unique string id of the processor."""
        return self._processor_id

    def engine(self) -> Optional['AbstractEngine']:
        """Returns the parent Engine object.

        Returns:
            The program's parent Engine.

        Raises:
            ValueError: if no engine has been defined for this processor.
        """
        return self._engine

    def set_engine(self, engine):
        """Sets the parent processor."""
        self._engine = engine

    def expected_down_time(self) -> 'Optional[datetime.datetime]':
        """Returns the start of the next expected down time of the processor, if
        set."""
        return self._expected_down_time

    def expected_recovery_time(self) -> 'Optional[datetime.datetime]':
        """Returns the expected the processor should be available, if set."""
        return self._expected_recovery_time

    def _create_id(self, id_type: str = 'reservation') -> str:
        """Creates a unique resource id for child objects."""
        self._resource_id_counter += 1
        return (
            f'projects/{self._project_name}/'
            f'processors/{self._processor_id}/'
            f'{id_type}/{self._resource_id_counter}'
        )

    def _reservation_to_time_slot(
        self, reservation: qtypes.QuantumReservation
    ) -> qtypes.QuantumTimeSlot:
        """Changes a reservation object into a time slot object."""
        return qtypes.QuantumTimeSlot(
            processor_name=self._processor_id,
            start_time=Timestamp(seconds=reservation.start_time.seconds),
            end_time=Timestamp(seconds=reservation.end_time.seconds),
            slot_type=qenums.QuantumTimeSlot.TimeSlotType.RESERVATION,
        )

    def _insert_reservation_into(self, time_slot: qtypes.QuantumTimeSlot) -> None:
        """Inserts a new reservation time slot into the ordered schedule.

        If this reservation overlaps with existing time slots, these slots will be
        shortened, removed, or split to insert the new reservation.
        """
        new_schedule = []
        time_slot_inserted = False
        for t in self._schedule:
            if t.end_time.seconds and t.end_time.seconds <= time_slot.start_time.seconds:
                #          [--time_slot--]
                # [--t--]
                new_schedule.append(t)
                continue
            if t.start_time.seconds and t.start_time.seconds >= time_slot.end_time.seconds:
                # [--time_slot--]
                #                   [--t--]
                new_schedule.append(t)
                continue
            if t.start_time.seconds and time_slot.start_time.seconds <= t.start_time.seconds:
                if not time_slot_inserted:
                    new_schedule.append(time_slot)
                    time_slot_inserted = True
                if not t.end_time.seconds or t.end_time.seconds > time_slot.end_time.seconds:
                    # [--time_slot---]
                    #          [----t-----]
                    t.start_time.seconds = time_slot.end_time.seconds
                    new_schedule.append(t)
                # if t.end_time < time_slot.end_time
                # [------time_slot-----]
                #      [-----t-----]
                # t should be removed
            else:
                if not t.end_time.seconds or t.end_time.seconds > time_slot.end_time.seconds:
                    #    [---time_slot---]
                    # [-------------t---------]
                    # t should be split
                    start = qtypes.QuantumTimeSlot(
                        processor_name=self._processor_id,
                        end_time=Timestamp(seconds=time_slot.start_time.seconds),
                        slot_type=t.slot_type,
                    )
                    if t.start_time.seconds:
                        start.start_time.seconds = t.start_time.seconds
                    end = qtypes.QuantumTimeSlot(
                        processor_name=self._processor_id,
                        start_time=Timestamp(seconds=time_slot.end_time.seconds),
                        slot_type=t.slot_type,
                    )
                    if t.end_time.seconds:
                        end.end_time.seconds = t.end_time.seconds

                    new_schedule.append(start)
                    new_schedule.append(time_slot)
                    new_schedule.append(end)

                else:
                    #       [---time_slot---]
                    # [----t-----]
                    t.end_time.seconds = time_slot.start_time.seconds
                    new_schedule.append(t)
                    new_schedule.append(time_slot)
                time_slot_inserted = True

        if not time_slot_inserted:
            new_schedule.append(time_slot)
        self._schedule = new_schedule

    def _is_available(self, time_slot: qtypes.QuantumTimeSlot) -> bool:
        """Returns True if the slot is available for reservation."""
        for t in self._schedule:
            if t.slot_type == qenums.QuantumTimeSlot.TimeSlotType.UNALLOCATED:
                continue
            if t.end_time.seconds and t.end_time.seconds <= time_slot.start_time.seconds:
                continue
            if t.start_time.seconds and t.start_time.seconds >= time_slot.end_time.seconds:
                continue
            return False
        return True

    def create_reservation(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        whitelisted_users: Optional[List[str]] = None,
    ) -> qtypes.QuantumReservation:
        """Creates a reservation on this processor.

        Args:
            start_time: the starting date/time of the reservation.
            end_time: the ending date/time of the reservation.
            whitelisted_users: a list of emails that are allowed
              to send programs during this reservation (in addition to users
              with permission "quantum.reservations.use" on the project).

        Raises:
            ValueError: if start_time is after end_time.
        """
        if end_time < start_time:
            raise ValueError('End time of reservation must be after the start time')
        reservation_id = self._create_id()
        new_reservation = qtypes.QuantumReservation(
            name=reservation_id,
            start_time=Timestamp(seconds=int(start_time.timestamp())),
            end_time=Timestamp(seconds=int(end_time.timestamp())),
            whitelisted_users=whitelisted_users,
        )
        time_slot = self._reservation_to_time_slot(new_reservation)
        if not self._is_available(time_slot):
            raise ValueError('Time slot is not available for reservations')

        self._reservations[reservation_id] = new_reservation
        self._insert_reservation_into(time_slot)
        return new_reservation

    def remove_reservation(self, reservation_id: str) -> None:
        """Removes a reservation on this processor."""
        if reservation_id in self._reservations:
            del self._reservations[reservation_id]

    def get_reservation(self, reservation_id: str) -> qtypes.QuantumReservation:
        """Retrieve a reservation given its id."""
        if reservation_id in self._reservations:
            return self._reservations[reservation_id]
        else:
            return None

    def update_reservation(
        self,
        reservation_id: str,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        whitelisted_users: Optional[List[str]] = None,
    ) -> None:
        """Updates a reservation with new information.

        Updates a reservation with a new start date, end date, or
        list of additional users.  For each field, it the argument is left as
        None, it will not be updated.

        Args:
            reservation_id: The string identifier of the reservation to change.
            start_time: New starting time  of the reservation.  If unspecified,
                starting time is left unchanged.
            end_time: New ending time  of the reservation.  If unspecified,
                ending time is left unchanged.
            whitelisted_users: The new list of whitelisted users to allow on
                the reservation.  If unspecified, the users are left unchanged.

        Raises:
            ValueError: if reservation_id does not exist.
        """
        if reservation_id not in self._reservations:
            raise ValueError(f'Reservation id {reservation_id} does not exist.')
        if start_time:
            self._reservations[reservation_id].start_time.seconds = _to_timestamp(start_time)
        if end_time:
            self._reservations[reservation_id].end_time.seconds = _to_timestamp(end_time)
        if whitelisted_users:
            del self._reservations[reservation_id].whitelisted_users[:]
            self._reservations[reservation_id].whitelisted_users.extend(whitelisted_users)

    def list_reservations(
        self,
        from_time: Union[None, datetime.datetime, datetime.timedelta] = datetime.timedelta(),
        to_time: Union[None, datetime.datetime, datetime.timedelta] = datetime.timedelta(weeks=2),
    ) -> List[qtypes.QuantumReservation]:
        """Retrieves the reservations from a processor.

        Only reservations from this processor and project will be
        returned. The schedule may be filtered by starting and ending time.

        Args:
            from_time: Filters the returned reservations to only include entries
                that end no earlier than the given value. Specified either as an
                absolute time (datetime.datetime) or as a time relative to now
                (datetime.timedelta). Defaults to now (a relative time of 0).
                Set to None to omit this filter.
            to_time: Filters the returned reservations to only include entries
                that start no later than the given value. Specified either as an
                absolute time (datetime.datetime) or as a time relative to now
                (datetime.timedelta). Defaults to two weeks from now (a relative
                time of two weeks). Set to None to omit this filter.

        Returns:
            A list of reservations.
        """
        start_timestamp = _to_timestamp(from_time)
        end_timestamp = _to_timestamp(to_time)
        reservation_list = []
        for reservation in self._reservations.values():
            if end_timestamp and reservation.start_time.seconds > end_timestamp:
                continue
            if start_timestamp and reservation.end_time.seconds < start_timestamp:
                continue
            reservation_list.append(reservation)
        return reservation_list

    def get_schedule(
        self,
        from_time: Union[None, datetime.datetime, datetime.timedelta] = datetime.timedelta(),
        to_time: Union[None, datetime.datetime, datetime.timedelta] = datetime.timedelta(weeks=2),
        time_slot_type: Optional[qenums.QuantumTimeSlot.TimeSlotType] = None,
    ) -> List[qtypes.QuantumTimeSlot]:
        """Retrieves the schedule for a processor.

        The schedule may be filtered by time.

        Args:
            from_time: Filters the returned schedule to only include entries
                that end no earlier than the given value. Specified either as an
                absolute time (datetime.datetime) or as a time relative to now
                (datetime.timedelta). Defaults to now (a relative time of 0).
                Set to None to omit this filter.
            to_time: Filters the returned schedule to only include entries
                that start no later than the given value. Specified either as an
                absolute time (datetime.datetime) or as a time relative to now
                (datetime.timedelta). Defaults to two weeks from now (a relative
                time of two weeks). Set to None to omit this filter.
            time_slot_type: Filters the returned schedule to only include
                entries with a given type (e.g. maintenance, open swim).
                Defaults to None. Set to None to omit this filter.

        Returns:
            Time slots that fit the criteria.
        """
        time_slots: List[qtypes.QuantumTimeSlot] = []
        start_timestamp = _to_timestamp(from_time)
        end_timestamp = _to_timestamp(to_time)
        for slot in self._schedule:
            if (
                start_timestamp
                and slot.end_time.seconds
                and slot.end_time.seconds < start_timestamp
            ):
                continue
            if (
                end_timestamp
                and slot.start_time.seconds
                and slot.start_time.seconds > end_timestamp
            ):
                continue
            time_slots.append(slot)
        return time_slots

    @abstractmethod
    def get_latest_calibration(self, timestamp: int) -> Optional[calibration.Calibration]:
        """Returns the latest calibration with the provided timestamp or earlier."""

    @abstractmethod
    def get_program(self, program_id: str) -> AbstractProgram:
        """Returns an AbstractProgram for an existing Quantum Engine program.

        Args:
            program_id: Unique ID of the program within the parent project.

        Returns:
            An AbstractProgram for the program.
        """

    @abstractmethod
    def list_programs(
        self,
        created_before: Optional[Union[datetime.datetime, datetime.date]] = None,
        created_after: Optional[Union[datetime.datetime, datetime.date]] = None,
        has_labels: Optional[Dict[str, str]] = None,
    ) -> List['AbstractLocalProgram']:
        """Returns a list of previously executed quantum programs.

        Args:
            created_after: retrieve programs that were created after this date
                or time.
            created_before: retrieve programs that were created before this date
                or time.
            has_labels: retrieve programs that have labels on them specified by
                this dict. If the value is set to `*`, filters having the label
                regardless of the label value will be filtered. For example, to
                query programs that have the shape label and have the color
                label with value red can be queried using
                `{'color: red', 'shape:*'}`
        """
