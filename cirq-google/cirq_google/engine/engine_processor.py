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

from typing import cast, Dict, Iterable, List, Optional, Sequence, TYPE_CHECKING, Union
from pytz import utc

import cirq
from cirq_google.engine.client.quantum import types as qtypes
from cirq_google.engine.client.quantum import enums as qenums
from cirq_google.api import v2
from cirq_google.devices import serializable_device
from cirq_google.engine import (
    abstract_processor,
    calibration,
    calibration_layer,
    engine_sampler,
)
from cirq_google.serialization import serializable_gate_set, serializer
from cirq_google.serialization import gate_sets as gs

if TYPE_CHECKING:
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


def _fix_deprecated_seconds_kwargs(kwargs):
    if 'earliest_timestamp_seconds' in kwargs:
        kwargs['earliest_timestamp'] = kwargs['earliest_timestamp_seconds']
        del kwargs['earliest_timestamp_seconds']
    if 'latest_timestamp_seconds' in kwargs:
        kwargs['latest_timestamp'] = kwargs['latest_timestamp_seconds']
        del kwargs['latest_timestamp_seconds']
    return kwargs


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
        _processor: Optional[qtypes.QuantumProcessor] = None,
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
        gate_set: Optional[serializer.Serializer] = None,
    ) -> engine_sampler.QuantumEngineSampler:
        """Returns a sampler backed by the engine.

        Args:
            gate_set: A `Serializer` that determines how to serialize circuits
            when requesting samples. If not specified, uses proto v2.5 serialization.

        Returns:
            A `cirq.Sampler` instance (specifically a `engine_sampler.QuantumEngineSampler`
            that will send circuits to the Quantum Computing Service
            when sampled.1
        """
        return engine_sampler.QuantumEngineSampler(
            engine=self.engine(),
            processor_id=self.processor_id,
            gate_set=gate_set,
        )

    def run_batch(
        self,
        programs: Sequence[cirq.AbstractCircuit],
        program_id: Optional[str] = None,
        job_id: Optional[str] = None,
        params_list: Sequence[cirq.Sweepable] = None,
        repetitions: int = 1,
        gate_set: Optional[serializer.Serializer] = None,
        program_description: Optional[str] = None,
        program_labels: Optional[Dict[str, str]] = None,
        job_description: Optional[str] = None,
        job_labels: Optional[Dict[str, str]] = None,
    ) -> 'abstract_job.AbstractJob':
        """Runs the supplied Circuits on this processor.

        This will combine each Circuit provided in `programs` into
        a BatchProgram.  Each circuit will pair with the associated
        parameter sweep provided in the `params_list`.  The number of
        programs is required to match the number of sweeps.
        This method does not block until a result is returned.  However,
        no results will be available until the entire batch is complete.
        Args:
            programs: The Circuits to execute as a batch.
            program_id: A user-provided identifier for the program. This must
                be unique within the Google Cloud project being used. If this
                parameter is not provided, a random id of the format
                'prog-################YYMMDD' will be generated, where # is
                alphanumeric and YYMMDD is the current year, month, and day.
            job_id: Job identifier to use. If this is not provided, a random id
                of the format 'job-################YYMMDD' will be generated,
                where # is alphanumeric and YYMMDD is the current year, month,
                and day.
            params_list: Parameter sweeps to use with the circuits. The number
                of sweeps should match the number of circuits and will be
                paired in order with the circuits. If this is None, it is
                assumed that the circuits are not parameterized and do not
                require sweeps.
            repetitions: Number of circuit repetitions to run.  Each sweep value
                of each circuit in the batch will run with the same repetitions.
            gate_set: The gate set used to serialize the circuit. The gate set
                must be supported by the selected processor.
            program_description: An optional description to set on the program.
            program_labels: Optional set of labels to set on the program.
            job_description: An optional description to set on the job.
            job_labels: Optional set of labels to set on the job.
        Returns:
            An `abstract_job.AbstractJob`. If this is iterated over it returns
            a list of `cirq.Result`. All Results for the first circuit are listed
            first, then the Results for the second, etc. The Results
            for a circuit are listed in the order imposed by the associated
            parameter sweep.
        """
        return self.engine().run_batch(
            programs=programs,
            processor_ids=[self.processor_id],
            program_id=program_id,
            params_list=list(params_list) if params_list is not None else None,
            repetitions=repetitions,
            gate_set=gate_set,
            program_description=program_description,
            program_labels=program_labels,
            job_description=job_description,
            job_labels=job_labels,
        )

    def run_calibration(
        self,
        layers: List[calibration_layer.CalibrationLayer],
        program_id: Optional[str] = None,
        job_id: Optional[str] = None,
        gate_set: Optional[serializer.Serializer] = None,
        program_description: Optional[str] = None,
        program_labels: Optional[Dict[str, str]] = None,
        job_description: Optional[str] = None,
        job_labels: Optional[Dict[str, str]] = None,
    ) -> 'abstract_job.AbstractJob':
        """Runs the specified calibrations on the processor.

        Each calibration will be specified by a `CalibrationLayer`
        that contains the type of the calibrations to run, a `Circuit`
        to optimize, and any arguments needed by the calibration routine.
        Arguments and circuits needed for each layer will vary based on the
        calibration type.  However, the typical calibration routine may
        require a single moment defining the gates to optimize, for example.
        Note: this is an experimental API and is not yet fully supported
        for all users.

        Args:
            layers: The layers of calibration to execute as a batch.
            program_id: A user-provided identifier for the program. This must
                be unique within the Google Cloud project being used. If this
                parameter is not provided, a random id of the format
                'calibration-################YYMMDD' will be generated,
                where # is alphanumeric and YYMMDD is the current year, month,
                and day.
            job_id: Job identifier to use. If this is not provided, a random id
                of the format 'calibration-################YYMMDD' will be
                generated, where # is alphanumeric and YYMMDD is the current
                year, month, and day.
            gate_set: The gate set used to serialize the circuit. The gate set
                must be supported by the selected processor.
            program_description: An optional description to set on the program.
            program_labels: Optional set of labels to set on the program.
            job_description: An optional description to set on the job.
            job_labels: Optional set of labels to set on the job.  By defauly,
                this will add a 'calibration' label to the job.
        Returns:
            An AbstractJob whose results can be retrieved by calling
            calibration_results().
        """
        return self.engine().run_calibration(
            layers=layers,
            processor_id=self.processor_id,
            program_id=program_id,
            job_id=job_id,
            gate_set=gate_set,
            program_description=program_description,
            program_labels=program_labels,
            job_description=job_description,
            job_labels=job_labels,
        )

    def run_sweep(
        self,
        program: cirq.Circuit,
        program_id: Optional[str] = None,
        job_id: Optional[str] = None,
        params: cirq.Sweepable = None,
        repetitions: int = 1,
        gate_set: Optional[serializer.Serializer] = None,
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
            gate_set: The gate set used to serialize the circuit. The gate set
                must be supported by the selected processor.
            program_description: An optional description to set on the program.
            program_labels: Optional set of labels to set on the program.
            job_description: An optional description to set on the job.
            job_labels: Optional set of labels to set on the job.
        Returns:
            An AbstractJob. If this is iterated over it returns a list of
            `cirq.Result`, one for each parameter sweep.
        """
        return self.engine().run_sweep(
            processor_ids=[self.processor_id],
            program=program,
            program_id=program_id,
            job_id=job_id,
            params=params,
            repetitions=repetitions,
            gate_set=gate_set,
            program_description=program_description,
            program_labels=program_labels,
            job_description=job_description,
            job_labels=job_labels,
        )

    def _inner_processor(self) -> qtypes.QuantumProcessor:
        if not self._processor:
            self._processor = self.context.client.get_processor(self.project_id, self.processor_id)
        return self._processor

    def health(self) -> str:
        """Returns the current health of processor."""
        self._processor = self.context.client.get_processor(self.project_id, self.processor_id)
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

    def get_device_specification(self) -> Optional[v2.device_pb2.DeviceSpecification]:
        """Returns a device specification proto for use in determining
        information about the device.

        Returns:
            Device specification proto if present.
        """
        if self._inner_processor().HasField('device_spec'):
            return v2.device_pb2.DeviceSpecification.FromString(
                self._inner_processor().device_spec.value
            )
        else:
            return None

    def get_device(
        self,
        gate_sets: Iterable[serializer.Serializer] = (),
    ) -> cirq.Device:
        """Returns a `Device` created from the processor's device specification.

        This method queries the processor to retrieve the device specification,
        which is then use to create a `serializable_gate_set.SerializableDevice` that will validate
        that operations are supported and use the correct qubits.
        """
        spec = self.get_device_specification()
        if not spec:
            raise ValueError('Processor does not have a device specification')
        if not gate_sets:
            # Default is to use all named gatesets in the device spec
            gate_sets = []
            for valid_gate_set in spec.valid_gate_sets:
                if valid_gate_set.name in gs.NAMED_GATESETS:
                    gate_sets.append(gs.NAMED_GATESETS[valid_gate_set.name])
        if not all(isinstance(gs, serializable_gate_set.SerializableGateSet) for gs in gate_sets):
            raise ValueError('All gate_sets must be SerializableGateSet currently.')
        return serializable_device.SerializableDevice.from_proto(
            spec, cast(Iterable[serializable_gate_set.SerializableGateSet], gate_sets)
        )

    @cirq._compat.deprecated_parameter(
        deadline='v1.0',
        fix='Change earliest_timestamp_seconds to earliest_timestamp.',
        parameter_desc='earliest_timestamp_seconds',
        match=lambda args, kwargs: 'earliest_timestamp_seconds' in kwargs,
        rewrite=lambda args, kwargs: (args, _fix_deprecated_seconds_kwargs(kwargs)),
    )
    @cirq._compat.deprecated_parameter(
        deadline='v1.0',
        fix='Change latest_timestamp_seconds to latest_timestamp.',
        parameter_desc='latest_timestamp_seconds',
        match=lambda args, kwargs: 'latest_timestamp_seconds' in kwargs,
        rewrite=lambda args, kwargs: (args, _fix_deprecated_seconds_kwargs(kwargs)),
    )
    def list_calibrations(
        self,
        earliest_timestamp: Optional[Union[datetime.datetime, datetime.date, int]] = None,
        latest_timestamp: Optional[Union[datetime.datetime, datetime.date, int]] = None,
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
        earliest_timestamp_seconds = _date_to_timestamp(earliest_timestamp)
        latest_timestamp_seconds = _date_to_timestamp(latest_timestamp)

        if earliest_timestamp_seconds and latest_timestamp_seconds:
            filter_str = 'timestamp >= %d AND timestamp <= %d' % (
                earliest_timestamp_seconds,
                latest_timestamp_seconds,
            )
        elif earliest_timestamp_seconds:
            filter_str = 'timestamp >= %d' % earliest_timestamp_seconds
        elif latest_timestamp_seconds:
            filter_str = 'timestamp <= %d' % latest_timestamp_seconds
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

    def get_current_calibration(
        self,
    ) -> Optional[calibration.Calibration]:
        """Returns metadata about the current calibration for a processor.

        Returns:
            The calibration data or None if there is no current calibration.
        """
        response = self.context.client.get_current_calibration(self.project_id, self.processor_id)
        if response:
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
        if proc:
            freeze = proc.schedule_frozen_period.seconds
        else:
            freeze = None
        if not freeze:
            raise ValueError(
                'Cannot determine freeze_schedule from processor.'
                'Call _cancel_reservation or _delete_reservation.'
            )
        secs_until = reservation.start_time.seconds - int(datetime.datetime.now(tz=utc).timestamp())
        if secs_until > freeze:
            return self._delete_reservation(reservation_id)
        else:
            return self._cancel_reservation(reservation_id)

    def get_reservation(self, reservation_id: str):
        """Retrieve a reservation given its id."""
        return self.context.client.get_reservation(
            self.project_id, self.processor_id, reservation_id
        )

    def update_reservation(
        self,
        reservation_id: str,
        start_time: datetime.datetime = None,
        end_time: datetime.datetime = None,
        whitelisted_users: List[str] = None,
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
    ) -> List[qenums.QuantumTimeSlot]:
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
        time_slot_type: Optional[qenums.QuantumTimeSlot.TimeSlotType] = None,
    ) -> List[qenums.QuantumTimeSlot]:
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


def _to_calibration(calibration_any: qtypes.any_pb2.Any) -> calibration.Calibration:
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
