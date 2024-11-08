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

from typing import Dict, List, Optional, TYPE_CHECKING, Union

from google.protobuf import any_pb2

import cirq
from cirq_google.api import v2
from cirq_google.cloud import quantum
from cirq_google.devices import grid_device
from cirq_google.engine import abstract_processor, calibration, processor_sampler, util

if TYPE_CHECKING:
    import cirq_google as cg
    import cirq_google.engine.engine as engine_base
    import cirq_google.engine.abstract_job as abstract_job


def _date_to_timestamp(
    union_time: Optional[Union[datetime.datetime, datetime.date, int]]
) -> Optional[int]:
    if isinstance(union_time, int):
        return union_time
    elif isinstance(union_time, datetime.datetime):
        return int(union_time.timestamp())
    elif isinstance(union_time, datetime.date):
        return int(datetime.datetime.combine(union_time, datetime.datetime.min.time()).timestamp())
    return None


class EngineProcessor(abstract_processor.AbstractProcessor):
    """A processor available via the Quantum Engine API.

    Attributes:
        project_id: A project_id of the parent Google Cloud Project.
        processor_id: Unique ID of the processor.
    """

    def __init__(
        self,
        project_id: str,
        processor_id: str,
        context: 'engine_base.EngineContext',
        _processor: Optional[quantum.QuantumProcessor] = None,
    ) -> None:
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

    def __repr__(self) -> str:
        return (
            f'<EngineProcessor: processor_id={self.processor_id!r}, '
            f'project_id={self.project_id!r}>'
        )

    def engine(self) -> 'engine_base.Engine':
        """Returns the parent Engine object.

        Returns:
            The program's parent Engine.
        """
        import cirq_google.engine.engine as engine_base

        return engine_base.Engine(self.project_id, context=self.context)

    def get_sampler(
        self,
        run_name: str = "",
        device_config_name: str = "",
        snapshot_id: str = "",
        max_concurrent_jobs: int = 10,
    ) -> 'cg.engine.ProcessorSampler':
        """Returns a sampler backed by the engine.
        Args:
            run_name: A unique identifier representing an automation run for the
                processor. An Automation Run contains a collection of device
                configurations for the processor.
            device_config_name: An identifier used to select the processor configuration
                utilized to run the job. A configuration identifies the set of
                available qubits, couplers, and supported gates in the processor.
            snapshot_id: A unique identifier for an immutable snapshot reference.
                A snapshot contains a collection of device configurations for the
                processor.
            max_concurrent_jobs: The maximum number of jobs to be sent
                simultaneously to the Engine. This client-side throttle can be
                used to proactively reduce load to the backends and avoid quota
                violations when pipelining circuit executions.

        Returns:
            A `cirq.Sampler` instance (specifically a `engine_sampler.ProcessorSampler`
            that will send circuits to the Quantum Computing Service
            when sampled.

        Raises:
            ValueError: If only one of `run_name` and `device_config_name` are specified.
            ValueError: If both `run_name` and `snapshot_id` are specified.

        """
        processor = self._inner_processor()
        if run_name and snapshot_id:
            raise ValueError('Cannot specify both `run_name` and `snapshot_id`')
        if (bool(run_name) or bool(snapshot_id)) ^ bool(device_config_name):
            raise ValueError(
                'Cannot specify only one of top level identifier and `device_config_name`'
            )
        # If not provided, initialize the sampler with the Processor's default values.
        if not run_name and not device_config_name and not snapshot_id:
            run_name = processor.default_device_config_key.run
            device_config_name = processor.default_device_config_key.config_alias
            snapshot_id = processor.default_device_config_key.snapshot_id
        return processor_sampler.ProcessorSampler(
            processor=self,
            run_name=run_name,
            snapshot_id=snapshot_id,
            device_config_name=device_config_name,
            max_concurrent_jobs=max_concurrent_jobs,
        )

    async def run_sweep_async(
        self,
        program: cirq.AbstractCircuit,
        *,
        device_config_name: str,
        run_name: str = "",
        snapshot_id: str = "",
        program_id: Optional[str] = None,
        job_id: Optional[str] = None,
        params: cirq.Sweepable = None,
        repetitions: int = 1,
        program_description: Optional[str] = None,
        program_labels: Optional[Dict[str, str]] = None,
        job_description: Optional[str] = None,
        job_labels: Optional[Dict[str, str]] = None,
    ) -> 'abstract_job.AbstractJob':
        """Runs the supplied Circuit on this processor.

        In contrast to run, this runs across multiple parameter sweeps, and
        does not block until a result is returned.

        Args:
            program: The Circuit to execute. If a circuit is
                provided, a moment by moment schedule will be used.
            run_name: A unique identifier representing an automation run for the
                processor. An Automation Run contains a collection of device
                configurations for the processor.
            device_config_name: An identifier used to select the processor configuration
                utilized to run the job. A configuration identifies the set of
                available qubits, couplers, and supported gates in the processor.
            snapshot_id: A unique identifier for an immutable snapshot reference.
                A snapshot contains a collection of device configurations for the
                processor.
            program_id: A user-provided identifier for the program. This must
                be unique within the Google Cloud project being used. If this
                parameter is not provided, a random id of the format
                'prog-################YYMMDD' will be generated, where # is
                alphanumeric and YYMMDD is the current year, month, and day.
            job_id: Job identifier to use. If this is not provided, a random id
                of the format 'job-################YYMMDD' will be generated,
                where # is alphanumeric and YYMMDD is the current year, month,
                and day.
            params: Parameters to run with the program.
            repetitions: The number of circuit repetitions to run.
            program_description: An optional description to set on the program.
            program_labels: Optional set of labels to set on the program.
            job_description: An optional description to set on the job.
            job_labels: Optional set of labels to set on the job.
        Returns:
            An AbstractJob. If this is iterated over it returns a list of
            `cirq.Result`, one for each parameter sweep.
        Raises:
            ValueError: If neither `processor_id` or `processor_ids` are set.
            ValueError: If  only one of `run_name` and `device_config_name` are specified.
            ValueError: If `processor_ids` has more than one processor id.
            ValueError: If either `run_name` and `device_config_name` are set but
                `processor_id` is empty.
        """
        return await self.engine().run_sweep_async(
            program=program,
            program_id=program_id,
            job_id=job_id,
            params=params,
            repetitions=repetitions,
            program_description=program_description,
            program_labels=program_labels,
            job_description=job_description,
            job_labels=job_labels,
            processor_id=self.processor_id,
            run_name=run_name,
            snapshot_id=snapshot_id,
            device_config_name=device_config_name,
        )

    def _inner_processor(self) -> quantum.QuantumProcessor:
        if self._processor is None:
            self._processor = self.context.client.get_processor(self.project_id, self.processor_id)
        return self._processor

    def health(self) -> str:
        """Returns the current health of processor."""
        self._processor = self.context.client.get_processor(self.project_id, self.processor_id)
        return self._processor.health.name

    def expected_down_time(self) -> 'Optional[datetime.datetime]':
        """Returns the start of the next expected down time of the processor, if
        set."""
        return self._inner_processor().expected_down_time

    def expected_recovery_time(self) -> 'Optional[datetime.datetime]':
        """Returns the expected the processor should be available, if set."""
        return self._inner_processor().expected_recovery_time

    def supported_languages(self) -> List[str]:
        """Returns the list of processor supported program languages."""
        return self._inner_processor().supported_languages

    def get_device_specification(self) -> Optional[v2.device_pb2.DeviceSpecification]:
        """Returns a device specification proto for use in determining
        information about the device.

        Returns:
            Device specification proto if present.
        """
        device_spec = self._inner_processor().device_spec
        if device_spec and device_spec.type_url:
            return util.unpack_any(device_spec, v2.device_pb2.DeviceSpecification())
        else:
            return None

    def get_device(self) -> cirq.Device:
        """Returns a `Device` created from the processor's device specification.

        This method queries the processor to retrieve the device specification,
        which is then use to create a `cirq_google.GridDevice` that will
        validate that operations are supported and use the correct qubits.
        """
        spec = self.get_device_specification()
        if not spec:
            raise ValueError('Processor does not have a device specification')
        return grid_device.GridDevice.from_proto(spec)

    def list_calibrations(
        self,
        earliest_timestamp: Optional[Union[datetime.datetime, datetime.date, int]] = None,
        latest_timestamp: Optional[Union[datetime.datetime, datetime.date, int]] = None,
    ) -> List[calibration.Calibration]:
        """Retrieve metadata about a specific calibration run.

        Params:
            earliest_timestamp: The earliest timestamp of a calibration to return in UTC.
            latest_timestamp: The latest timestamp of a calibration to return in UTC.

        Returns:
            The list of calibration data with the most recent first.
        """
        earliest_timestamp_seconds = _date_to_timestamp(earliest_timestamp)
        latest_timestamp_seconds = _date_to_timestamp(latest_timestamp)

        if earliest_timestamp_seconds and latest_timestamp_seconds:
            filter_str = (
                f'timestamp >= {earliest_timestamp_seconds:d} AND '
                f'timestamp <= {latest_timestamp_seconds:d}'
            )
        elif earliest_timestamp_seconds:
            filter_str = f'timestamp >= {earliest_timestamp_seconds:d}'
        elif latest_timestamp_seconds:
            filter_str = f'timestamp <= {latest_timestamp_seconds:d}'
        else:
            filter_str = ''
        response = self.context.client.list_calibrations(
            self.project_id, self.processor_id, filter_str
        )
        return [_to_calibration(c.data) for c in list(response)]

    def get_calibration(self, calibration_timestamp_seconds: int) -> calibration.Calibration:
        """Retrieve metadata about a specific calibration run.

        Params:
            calibration_timestamp_seconds: The timestamp of the calibration in
                seconds since epoch.

        Returns:
            The calibration data.
        """
        response = self.context.client.get_calibration(
            self.project_id, self.processor_id, calibration_timestamp_seconds
        )
        return _to_calibration(response.data)

    def get_current_calibration(self) -> Optional[calibration.Calibration]:
        """Returns metadata about the current calibration for a processor.

        Returns:
            The calibration data or None if there is no current calibration.
        """
        response = self.context.client.get_current_calibration(self.project_id, self.processor_id)
        if response is not None:
            return _to_calibration(response.data)
        else:
            return None

    def create_reservation(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        whitelisted_users: Optional[List[str]] = None,
    ):
        """Creates a reservation on this processor.

        Args:
            start_time: the starting date/time of the reservation.
            end_time: the ending date/time of the reservation.
            whitelisted_users: a list of emails that are allowed
              to send programs during this reservation (in addition to users
              with permission "quantum.reservations.use" on the project).
        """
        response = self.context.client.create_reservation(
            self.project_id, self.processor_id, start_time, end_time, whitelisted_users
        )
        return response

    def _delete_reservation(self, reservation_id: str):
        """Delete a reservation.

        This will only work for reservations outside the processor's
        schedule freeze window.  If you are not sure whether the reservation
        falls within this window, use remove_reservation
        """
        return self.context.client.delete_reservation(
            self.project_id, self.processor_id, reservation_id
        )

    def _cancel_reservation(self, reservation_id: str):
        """Cancel a reservation.

        This will only work for reservations inside the processor's
        schedule freeze window.  If you are not sure whether the reservation
        falls within this window, use remove_reservation
        """
        return self.context.client.cancel_reservation(
            self.project_id, self.processor_id, reservation_id
        )

    def remove_reservation(self, reservation_id: str):
        reservation = self.get_reservation(reservation_id)
        if reservation is None:
            raise ValueError(f'Reservation id {reservation_id} not found.')
        proc = self._inner_processor()
        if proc is not None:
            freeze = proc.schedule_frozen_period
        else:
            freeze = None
        if not freeze:
            raise ValueError(
                'Cannot determine freeze_schedule from processor.'
                'Call _cancel_reservation or _delete_reservation.'
            )
        secs_until = reservation.start_time.timestamp() - datetime.datetime.now().timestamp()
        if secs_until > freeze.total_seconds():
            return self._delete_reservation(reservation_id)
        else:
            return self._cancel_reservation(reservation_id)

    def get_reservation(self, reservation_id: str) -> Optional[quantum.QuantumReservation]:
        """Retrieve a reservation given its id."""
        return self.context.client.get_reservation(
            self.project_id, self.processor_id, reservation_id
        )

    def update_reservation(
        self,
        reservation_id: str,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        whitelisted_users: Optional[List[str]] = None,
    ):
        """Updates a reservation with new information.

        Updates a reservation with a new start date, end date, or
        list of additional users.  For each field, it the argument is left as
        None, it will not be updated.
        """
        return self.context.client.update_reservation(
            self.project_id,
            self.processor_id,
            reservation_id,
            start=start_time,
            end=end_time,
            whitelisted_users=whitelisted_users,
        )

    def list_reservations(
        self,
        from_time: Union[None, datetime.datetime, datetime.timedelta] = datetime.timedelta(),
        to_time: Union[None, datetime.datetime, datetime.timedelta] = datetime.timedelta(weeks=2),
    ) -> List[quantum.QuantumTimeSlot]:
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
        filters = _to_date_time_filters(from_time, to_time)
        filter_str = ' AND '.join(filters)
        return self.context.client.list_reservations(self.project_id, self.processor_id, filter_str)

    def get_schedule(
        self,
        from_time: Union[None, datetime.datetime, datetime.timedelta] = datetime.timedelta(),
        to_time: Union[None, datetime.datetime, datetime.timedelta] = datetime.timedelta(weeks=2),
        time_slot_type: Optional[quantum.QuantumTimeSlot.TimeSlotType] = None,
    ) -> List[quantum.QuantumTimeSlot]:
        """Retrieves the schedule for a processor.

        The schedule may be filtered by time.

        Time slot type will be supported in the future.

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
            Schedule time slots.
        """
        filters = _to_date_time_filters(from_time, to_time)
        if time_slot_type is not None:
            filters.append(f'time_slot_type = {time_slot_type.name}')
        filter_str = ' AND '.join(filters)
        return self.context.client.list_time_slots(self.project_id, self.processor_id, filter_str)

    def __str__(self):
        return (
            f"EngineProcessor(project_id={self.project_id!r}, "
            f"processor_id={self.processor_id!r})"
        )


def _to_calibration(calibration_any: any_pb2.Any) -> calibration.Calibration:
    metrics = v2.metrics_pb2.MetricsSnapshot.FromString(calibration_any.value)
    return calibration.Calibration(metrics)


def _to_date_time_filters(
    from_time: Union[None, datetime.datetime, datetime.timedelta],
    to_time: Union[None, datetime.datetime, datetime.timedelta],
) -> List[str]:
    now = datetime.datetime.now()

    if from_time is None:
        start_time = None
    elif isinstance(from_time, datetime.timedelta):
        start_time = now + from_time
    elif isinstance(from_time, datetime.datetime):
        start_time = from_time
    else:
        raise ValueError(f"Don't understand from_time of type {type(from_time)}.")

    if to_time is None:
        end_time = None
    elif isinstance(to_time, datetime.timedelta):
        end_time = now + to_time
    elif isinstance(to_time, datetime.datetime):
        end_time = to_time
    else:
        raise ValueError(f"Don't understand to_time of type {type(to_time)}.")

    filters = []
    if end_time is not None:
        filters.append(f'start_time < {int(end_time.timestamp())}')
    if start_time is not None:
        filters.append(f'end_time > {int(start_time.timestamp())}')
    return filters
