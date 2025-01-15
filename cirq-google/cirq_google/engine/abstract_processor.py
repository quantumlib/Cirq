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
"""Abstract interface for a quantum processor.

This interface can run circuits, sweeps, batches, or calibration
requests.  Inheritors of this interface should implement all
methods.
"""

import abc
import datetime
from typing import Dict, List, Optional, TYPE_CHECKING, Union

import duet

import cirq
from cirq_google.api import v2
from cirq_google.cloud import quantum
from cirq_google.engine import calibration

if TYPE_CHECKING:
    import cirq_google as cg
    import cirq_google.engine.abstract_engine as abstract_engine
    import cirq_google.engine.abstract_job as abstract_job
    import cirq_google.serialization.serializer as serializer


class AbstractProcessor(abc.ABC):
    """An abstract interface for a quantum processor.

    This quantum processor has the ability to execute single circuits
    (via the run method), parameter sweeps (via run_sweep), batched
    lists of circuits (via run_batch), and calibration
    requests (via run_calibration).  Running circuits can also be
    done using the `cirq.Sampler` by calling get_sampler.

    The processor interface also includes methods to create, list,
    and remove reservations on the processor for dedicated access.
    The processor can also list calibration metrics for the processor
    given a time period.

    This is an abstract class.  Inheritors should implement abstract methods.
    """

    async def run_async(
        self,
        program: cirq.Circuit,
        *,
        device_config_name: str,
        run_name: str = "",
        snapshot_id: str = "",
        program_id: Optional[str] = None,
        job_id: Optional[str] = None,
        param_resolver: Optional[cirq.ParamResolver] = None,
        repetitions: int = 1,
        program_description: Optional[str] = None,
        program_labels: Optional[Dict[str, str]] = None,
        job_description: Optional[str] = None,
        job_labels: Optional[Dict[str, str]] = None,
    ) -> cirq.Result:
        """Runs the supplied Circuit on this processor.

        Args:
            program: The Circuit to execute. If a circuit is
                provided, a moment by moment schedule will be used.
            run_name: A unique identifier representing an automation run for the
                processor. An Automation Run contains a collection of device
                configurations for the processor. `snapshot_id` and `run_name`
                should not both be set. Choose one.
            snapshot_id: A unique identifier for an immutable snapshot reference.
                A snapshot contains a collection of device configurations for the
                processor. `snapshot_id` and `run_name` should not both be set.
                Choose one.
            device_config_name: An identifier used to select the processor configuration
                utilized to run the job. A configuration identifies the set of
                available qubits, couplers, and supported gates in the processor.
            program_id: A user-provided identifier for the program. This must
                be unique within the Google Cloud project being used. If this
                parameter is not provided, a random id of the format
                'prog-################YYMMDD' will be generated, where # is
                alphanumeric and YYMMDD is the current year, month, and day.
            job_id: Job identifier to use. If this is not provided, a random id
                of the format 'job-################YYMMDD' will be generated,
                where # is alphanumeric and YYMMDD is the current year, month,
                and day.
            param_resolver: Parameters to run with the program.
            repetitions: The number of repetitions to simulate.
            program_description: An optional description to set on the program.
            program_labels: Optional set of labels to set on the program.
            job_description: An optional description to set on the job.
            job_labels: Optional set of labels to set on the job.

        Returns:
            A single Result for this run.
        """
        job = await self.run_sweep_async(
            program=program,
            program_id=program_id,
            job_id=job_id,
            params=[param_resolver or cirq.ParamResolver({})],
            repetitions=repetitions,
            program_description=program_description,
            program_labels=program_labels,
            job_description=job_description,
            job_labels=job_labels,
            run_name=run_name,
            snapshot_id=snapshot_id,
            device_config_name=device_config_name,
        )
        return job.results()[0]

    run = duet.sync(run_async)

    @abc.abstractmethod
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
                processor. Both `snapshot_id` and `run_name` should not be set.
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
        """

    run_sweep = duet.sync(run_sweep_async)

    @abc.abstractmethod
    def get_sampler(
        self, run_name: str = "", device_config_name: str = ""
    ) -> 'cg.ProcessorSampler':
        """Returns a sampler backed by the processor.

        Args:
            run_name: A unique identifier representing an automation run for the
                processor. An Automation Run contains a collection of device
                configurations for the processor.
            device_config_name: An identifier used to select the processor configuration
                utilized to run the job. A configuration identifies the set of
                available qubits, couplers, and supported gates in the processor.
        """

    @abc.abstractmethod
    def engine(self) -> Optional['abstract_engine.AbstractEngine']:
        """Returns the parent Engine object.

        Returns:
            The program's parent Engine.
        """

    @abc.abstractmethod
    def health(self) -> str:
        """Returns the current health of processor."""

    @abc.abstractmethod
    def expected_down_time(self) -> 'Optional[datetime.datetime]':
        """Returns the start of the next expected down time of the processor, if
        set."""

    @abc.abstractmethod
    def expected_recovery_time(self) -> 'Optional[datetime.datetime]':
        """Returns the expected the processor should be available, if set."""

    @abc.abstractmethod
    def supported_languages(self) -> List[str]:
        """Returns the list of processor supported program languages."""

    @abc.abstractmethod
    def get_device_specification(self) -> Optional[v2.device_pb2.DeviceSpecification]:
        """Returns a device specification proto for use in determining
        information about the device.

        Returns:
            Device specification proto if present.
        """

    @abc.abstractmethod
    def get_device(self) -> cirq.Device:
        """Returns a `Device` created from the processor's device specification.

        This method queries the processor to retrieve the device specification,
        which is then use to create a `Device` that will validate
        that operations are supported and use the correct qubits.

        Args:
            gate_sets: An iterable of serializers that can be used in the device.

        Returns:
            A `cirq.Devive` representing the processor.
        """

    @abc.abstractmethod
    def list_calibrations(
        self,
        earliest_timestamp: Optional[Union[datetime.datetime, datetime.date, int]] = None,
        latest_timestamp: Optional[Union[datetime.datetime, datetime.date, int]] = None,
    ) -> List[calibration.Calibration]:
        """Retrieve metadata about a specific calibration run.

        Args:
            earliest_timestamp: The earliest timestamp of a calibration
                to return in UTC.
            latest_timestamp: The latest timestamp of a calibration to
                return in UTC.

        Returns:
            The list of calibration data with the most recent first.
        """

    @abc.abstractmethod
    def get_calibration(self, calibration_timestamp_seconds: int) -> calibration.Calibration:
        """Retrieve metadata about a specific calibration run.

        Args:
            calibration_timestamp_seconds: The timestamp of the calibration in
                seconds since epoch.

        Returns:
            The calibration data.
        """

    @abc.abstractmethod
    def get_current_calibration(self) -> Optional[calibration.Calibration]:
        """Returns metadata about the current calibration for a processor.

        Returns:
            The calibration data or None if there is no current calibration.
        """

    @abc.abstractmethod
    def create_reservation(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        whitelisted_users: Optional[List[str]] = None,
    ) -> quantum.QuantumReservation:
        """Creates a reservation on this processor.

        Args:
            start_time: the starting date/time of the reservation.
            end_time: the ending date/time of the reservation.
            whitelisted_users: a list of emails that are allowed
              to send programs during this reservation (in addition to users
              with permission "quantum.reservations.use" on the project).
        """

    @abc.abstractmethod
    def remove_reservation(self, reservation_id: str) -> None:
        """Removes a reservation on this processor."""

    @abc.abstractmethod
    def get_reservation(self, reservation_id: str) -> Optional[quantum.QuantumReservation]:
        """Retrieve a reservation given its id."""

    @abc.abstractmethod
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

    @abc.abstractmethod
    def list_reservations(
        self,
        from_time: Union[None, datetime.datetime, datetime.timedelta],
        to_time: Union[None, datetime.datetime, datetime.timedelta],
    ) -> List[quantum.QuantumReservation]:
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

    @abc.abstractmethod
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
