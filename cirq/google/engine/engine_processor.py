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

from typing import List, Optional, TYPE_CHECKING

from cirq.google.engine.client.quantum import types as qtypes
from cirq.google.engine.client.quantum import enums as qenums
from cirq.google.api import v2
from cirq.google.engine import calibration
from cirq.google.engine.engine_timeslot import EngineTimeSlot

if TYPE_CHECKING:
    import cirq.google.engine.engine as engine_base


class EngineProcessor:
    """A processor available via the Quantum Engine API.

    Attributes:
        project_id: A project_id of the parent Google Cloud Project.
        processor_id: Unique ID of the processor.
    """

    def __init__(self,
                 project_id: str,
                 processor_id: str,
                 context: 'engine_base.EngineContext',
                 _processor: Optional[qtypes.QuantumProcessor] = None) -> None:
        """A processor available via the engine.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            processor_id: Unique ID of the processor.
            context: Engine configuration and context to use.
            _processor: The optional current processor state.
        """
        self.project_id = project_id
        self.processor_id = processor_id
        self.context = context
        self._processor = _processor

    def engine(self) -> 'engine_base.Engine':
        """Returns the parent Engine object.

        Returns:
            The program's parent Engine.
        """
        import cirq.google.engine.engine as engine_base
        return engine_base.Engine(self.project_id, context=self.context)

    def _inner_processor(self) -> qtypes.QuantumProcessor:
        if not self._processor:
            self._processor = self.context.client.get_processor(
                self.project_id, self.processor_id)
        return self._processor

    def health(self) -> str:
        """Returns the current health of processor."""
        self._processor = self.context.client.get_processor(
            self.project_id, self.processor_id)
        return qtypes.QuantumProcessor.Health.Name(self._processor.health)

    def expected_down_time(self) -> 'Optional[datetime.datetime]':
        """Returns the start of the next expected down time of the processor, if
        set."""
        if self._inner_processor().HasField('expected_down_time'):
            return self._inner_processor().expected_down_time.ToDatetime()
        else:
            return None

    def expected_recovery_time(self) -> 'Optional[datetime.datetime]':
        """Returns the expected the processor should be available, if set."""
        if self._inner_processor().HasField('expected_recovery_time'):
            return self._inner_processor().expected_recovery_time.ToDatetime()
        else:
            return None

    def supported_languages(self) -> List[str]:
        """Returns the list of processor supported program languages."""
        return self._inner_processor().supported_languages

    def get_device_specification(
            self) -> Optional[v2.device_pb2.DeviceSpecification]:
        """Returns a device specification proto for use in determining
        information about the device.

        Returns:
            Device specification proto if present.
        """
        if self._inner_processor().HasField('device_spec'):
            device_spec = v2.device_pb2.DeviceSpecification()
            device_spec.ParseFromString(
                self._inner_processor().device_spec.value)
            return device_spec
        else:
            return None

    @staticmethod
    def _to_calibration(calibration_any: qtypes.any_pb2.Any
                       ) -> calibration.Calibration:
        metrics = v2.metrics_pb2.MetricsSnapshot()
        metrics.ParseFromString(calibration_any.value)
        return calibration.Calibration(metrics)

    def list_calibrations(self,
                          earliest_timestamp_seconds: Optional[int] = None,
                          latest_timestamp_seconds: Optional[int] = None
                         ) -> List[calibration.Calibration]:
        """Retrieve metadata about a specific calibration run.

        Params:
            earliest_timestamp_seconds: The earliest timestamp of a calibration
                to return in UTC.
            latest_timestamp_seconds: The latest timestamp of a calibration to
                return in UTC.

        Returns:
            The list of calibration data with the most recent first.
        """
        if earliest_timestamp_seconds and latest_timestamp_seconds:
            filter_str = 'timestamp >= %d AND timestamp <= %d' % (
                earliest_timestamp_seconds, latest_timestamp_seconds)
        elif earliest_timestamp_seconds:
            filter_str = 'timestamp >= %d' % earliest_timestamp_seconds
        elif latest_timestamp_seconds:
            filter_str = 'timestamp <= %d' % latest_timestamp_seconds
        else:
            filter_str = ''
        response = self.context.client.list_calibrations(
            self.project_id, self.processor_id, filter_str)
        return [self._to_calibration(c.data) for c in list(response)]

    def get_calibration(self, calibration_timestamp_seconds: int
                       ) -> calibration.Calibration:
        """Retrieve metadata about a specific calibration run.

        Params:
            calibration_timestamp_seconds: The timestamp of the calibration in
                seconds since epoch.

        Returns:
            The calibration data.
        """
        response = self.context.client.get_calibration(
            self.project_id, self.processor_id, calibration_timestamp_seconds)
        return self._to_calibration(response.data)

    def get_current_calibration(self,) -> Optional[calibration.Calibration]:
        """Returns metadata about the current calibration for a processor.

        Returns:
            The calibration data or None if there is no current calibration.
        """
        response = self.context.client.get_current_calibration(
            self.project_id, self.processor_id)
        if response:
            return self._to_calibration(response.data)
        else:
            return None

    def create_reservation(self, start_date: datetime.datetime,
                           end_date: datetime.datetime,
                           whitelisted_users: List[str]):
        """Creates a reservation on this processor.

        Args:
            start_date: the starting date/time of the reservation.
            end_date: the ending date/time of the reservation.
            whitelisted_users: a list of emails that are allowed
              to send programs during this reservation.
        """
        response = self.context.client.create_reservation(
            self.project_id, self.processor_id, None, start_date, end_date,
            whitelisted_users)
        return response

    def _delete_reservation(self, reservation_id: str):
        """Delete a reservation.

        This will only work for reservations outside the processor's
        forzen window.  If you are not sure whether the reservation
        falls within this window, use remove_reservation
        """
        return self.context.client.delete_reservation(self.project_id,
                                                      self.processor_id,
                                                      reservation_id)

    def _cancel_reservation(self, reservation_id: str):
        """Cancel a reservation.

        This will only work for reservations inside the processor's
        forzen window.  If you are not sure whether the reservation
        falls within this window, use remove_reservation
        """
        return self.context.client.cancel_reservation(self.project_id,
                                                      self.processor_id,
                                                      reservation_id)

    #TODO(dstrain): add remove reservation

    def get_reservation(self, reservation_id: str):
        """Retrieve a reservation given its id."""
        return self.context.client.get_reservation(self.project_id,
                                                   self.processor_id,
                                                   reservation_id)

    def update_reservation(self,
                           reservation_id: str,
                           start_date: datetime.datetime = None,
                           end_date: datetime.datetime = None,
                           whitelisted_users: List[str] = None):
        """Updates a reservation with new information.

        Updates a reservation with a new start date, end date, or
        list of users.  For each field, it the argument is left as
        None, it will not be updated.
        """
        rtn = None
        if start_date is not None:
            rtn = self.context.client.set_reservation_start_time(
                self.project_id,
                self.processor_id,
                reservation_id,
                start=start_date)
        if end_date is not None:
            rtn = self.context.client.set_reservation_end_time(
                self.project_id,
                self.processor_id,
                reservation_id,
                end=end_date)
        if whitelisted_users is not None:
            rtn = self.context.client.set_reservation_whitelisted_users(
                self.project_id,
                self.processor_id,
                reservation_id,
                whitelisted_users=whitelisted_users)
        return rtn

    # TODO(dstrain): add update_reservation

    def list_reservations(
            self,
            from_date: datetime.datetime = datetime.datetime.now(),
            to_date: datetime.datetime = datetime.datetime.now() +
            datetime.timedelta(weeks=2)) -> List[EngineTimeSlot]:
        """Retrieves the reservations from a processor.


        The schedule may be filtered by time and by the type of time
        (e.g. maintenance, open swim, unallocated, reservation).
        """
        filters = []
        if from_date:
            filters.append(f'start_time >= {from_date}')
        if to_date:
            filters.append(f'start_time <= {to_date}')
        filter_str = ' AND '.join(filters)
        return self.context.client.list_reservations(self.project_id,
                                                     self.processor_id,
                                                     filter_str)

    def get_schedule(self,
                     from_date: datetime.datetime = datetime.datetime.now(),
                     to_date: datetime.datetime = datetime.datetime.now() +
                     datetime.timedelta(weeks=2),
                     time_slot_type: qenums.QuantumTimeSlot.TimeSlotType = None
                    ) -> List[EngineTimeSlot]:
        """Retrieves the schedule for a processor.

        The schedule may be filtered by time and by the type of time
        (e.g. maintenance, open swim, unallocated, reservation).
        """
        filters = []
        if from_date:
            filters.append(f'start_time >= {from_date}')
        if to_date:
            filters.append(f'start_time <= {to_date}')
        if time_slot_type:
            filters.append(f'time_slot_type = "{time_slot_type}"')
        filter_str = ' AND '.join(filters)
        return self.context.client.list_time_slots(self.project_id,
                                                   self.processor_id,
                                                   filter_str)

    def __str__(self):
        return 'EngineProcessor(project_id=\'{}\', processor_id=\'{}\')'.format(
            self.project_id, self.processor_id)
