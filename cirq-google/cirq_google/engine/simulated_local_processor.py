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

from typing import Dict, List, Optional, Sequence, Union

import cirq

from cirq_google.api import v2
from cirq_google.engine import calibration, validating_sampler
from cirq_google.engine.abstract_local_processor import AbstractLocalProcessor
from cirq_google.engine.abstract_local_program import AbstractLocalProgram
from cirq_google.engine.abstract_program import AbstractProgram
from cirq_google.engine.local_simulation_type import LocalSimulationType
from cirq_google.engine.simulated_local_job import SimulatedLocalJob
from cirq_google.engine.simulated_local_program import SimulatedLocalProgram
from cirq_google.serialization.circuit_serializer import CIRCUIT_SERIALIZER
from cirq_google.engine.processor_sampler import ProcessorSampler
from cirq_google.engine import engine_validator

VALID_LANGUAGES = [
    'type.googleapis.com/cirq.google.api.v2.Program',
    'type.googleapis.com/cirq.google.api.v2.BatchProgram',
]


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


class SimulatedLocalProcessor(AbstractLocalProcessor):
    """A processor backed by a sampler and device.

    Intended for local simulation testing, this processor will
    create a `ValidationSampler` that will validate requests based on
    the provided device and an additional Callable (that can verify
    serialization constraints, for instance).  Jobs will then be
    executed using the provided sampler.

    This class also supports a list of calibration metrics that are
    stored in-memory to replicate Quantum Engine calibration metrics.

    This class can be used as a local emulator for the Quantum Engine
    API or for testing or mocking.

    Attributes:
        sampler: A `cirq.Sampler` that can execute the quantum jobs.
        device: An optional device, for validation of qubit connectivity.
        validator: A Callable that can validate additional characteristics
            beyond the device, such as serialization, repetition limits, etc.
        gate_set_validator:  A callable that can validate a circuit and sweeps
            based on the given serializer.
        simulation_type:  Whether sampler execution should be
            synchronous or asynchronous.
        calibrations: A dictionary of calibration metrics keyed by epoch seconds
            that can be returned by the processor.
        processor_id: Unique string id of the processor.
        engine: The parent `AbstractEngine` object, if available.
        expected_down_time: Optional datetime of the next expected downtime.
            For informational purpose only.
        expected_recovery_time: Optional datetime when the processor is
            expected to be available again.  For informational purpose only.
        schedule:  List of time slots that the scheduling/reservation should
            use.  All time slots must be non-overlapping.
        project_name: A project_name for resource naming.
        device_specification: a` DeviceSpecification` proto that the processor
            should return if `get_device_specification()` is queried.
    """

    def __init__(
        self,
        *args,
        sampler: cirq.Sampler = cirq.Simulator(),
        device: cirq.Device = cirq.UNCONSTRAINED_DEVICE,
        validator: Optional[validating_sampler.VALIDATOR_TYPE] = None,
        program_validator: Optional[engine_validator.PROGRAM_VALIDATOR_TYPE] = None,
        simulation_type: LocalSimulationType = LocalSimulationType.SYNCHRONOUS,
        calibrations: Optional[Dict[int, calibration.Calibration]] = None,
        device_specification: Optional[v2.device_pb2.DeviceSpecification] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._calibrations = calibrations or {}
        self._device = device
        self._simulation_type = simulation_type
        self._program_validator = program_validator or (lambda a, b, c, d: None)
        self._validator = validator
        self._sampler = validating_sampler.ValidatingSampler(
            device=self._device, validator=self._validator, sampler=sampler
        )
        self._programs: Dict[str, AbstractLocalProgram] = {}
        self._device_specification = device_specification

    def remove_program(self, program_id: str):
        """Remove reference to a child program."""
        if program_id in self._programs:
            del self._programs[program_id]

    def get_calibration(self, calibration_timestamp_seconds: int) -> calibration.Calibration:
        return self._calibrations[calibration_timestamp_seconds]

    def get_latest_calibration(self, timestamp: int) -> Optional[calibration.Calibration]:
        if not self._calibrations:
            return None
        return self._calibrations[max(self._calibrations)]

    def get_current_calibration(self) -> Optional[calibration.Calibration]:
        return self.get_latest_calibration(int(datetime.datetime.now().timestamp()))

    def get_device(self) -> cirq.Device:
        """Returns a `cirq.Device` created from the processor's device specification.

        This method queries the processor to retrieve the device specification,
        which is then use to create a `cirq.Device` that will validate
        that operations are supported and use the correct qubits.
        """
        return self._device

    def get_device_specification(self) -> Optional[v2.device_pb2.DeviceSpecification]:
        return self._device_specification

    def health(self):
        return 'OK'

    def list_calibrations(
        self,
        earliest_timestamp: Optional[Union[datetime.datetime, datetime.date, int]] = None,
        latest_timestamp: Optional[Union[datetime.datetime, datetime.date, int]] = None,
        **kwargs,
    ) -> List[calibration.Calibration]:
        earliest_timestamp_seconds = _date_to_timestamp(earliest_timestamp) or 0
        latest_timestamp_seconds = (
            _date_to_timestamp(latest_timestamp)
            or (datetime.datetime.now() + datetime.timedelta(days=10000)).timestamp()
        )
        return [
            cal[1]
            for cal in self._calibrations.items()
            if earliest_timestamp_seconds <= cal[0] <= latest_timestamp_seconds
        ]

    def get_sampler(self, run_name: str = "", device_config_name="") -> ProcessorSampler:
        return ProcessorSampler(
            processor=self, run_name=run_name, device_config_name=device_config_name
        )

    def supported_languages(self) -> List[str]:
        return VALID_LANGUAGES

    def list_programs(
        self,
        created_before: Optional[Union[datetime.datetime, datetime.date]] = None,
        created_after: Optional[Union[datetime.datetime, datetime.date]] = None,
        has_labels: Optional[Dict[str, str]] = None,
    ) -> List[AbstractLocalProgram]:
        before_limit = created_before or datetime.datetime(datetime.MAXYEAR, 1, 1)
        after_limit = created_after or datetime.datetime(datetime.MINYEAR, 1, 1)
        labels = has_labels or {}

        def _labels_match(program_labels):
            return all(program_labels.get(key) == label for key, label in labels.items())

        return [
            p
            for p in self._programs.values()
            if after_limit < p.create_time() < before_limit and _labels_match(p.labels())
        ]

    def get_program(self, program_id: str) -> AbstractProgram:
        """Returns an AbstractProgram for an existing Quantum Engine program.

        Args:
            program_id: Unique ID of the program within the parent project.

        Returns:
            An AbstractProgram for the program.

        Raises:
            KeyError: if program is not found
        """
        return self._programs[program_id]

    async def run_batch_async(
        self,
        programs: Sequence[cirq.AbstractCircuit],
        program_id: Optional[str] = None,
        job_id: Optional[str] = None,
        params_list: Optional[Sequence[cirq.Sweepable]] = None,
        repetitions: int = 1,
        program_description: Optional[str] = None,
        program_labels: Optional[Dict[str, str]] = None,
        job_description: Optional[str] = None,
        job_labels: Optional[Dict[str, str]] = None,
        run_name: str = "",
        device_config_name: str = "",
    ) -> SimulatedLocalJob:
        if program_id is None:
            program_id = self._create_id(id_type='program')
        if job_id is None:
            job_id = self._create_id(id_type='job')
        self._program_validator(programs, params_list or [{}], repetitions, CIRCUIT_SERIALIZER)
        self._programs[program_id] = SimulatedLocalProgram(
            program_id=program_id,
            simulation_type=self._simulation_type,
            circuits=programs,
            engine=self.engine(),
            processor=self,
        )
        job = SimulatedLocalJob(
            job_id=job_id,
            processor_id=self.processor_id,
            parent_program=self._programs[program_id],
            repetitions=repetitions,
            sweeps=list(params_list) if params_list is not None else None,
            sampler=self._sampler,
            simulation_type=self._simulation_type,
        )
        self._programs[program_id].add_job(job_id, job)
        return job

    async def run_sweep_async(
        self,
        program: cirq.AbstractCircuit,
        program_id: Optional[str] = None,
        job_id: Optional[str] = None,
        params: cirq.Sweepable = None,
        repetitions: int = 1,
        program_description: Optional[str] = None,
        program_labels: Optional[Dict[str, str]] = None,
        job_description: Optional[str] = None,
        job_labels: Optional[Dict[str, str]] = None,
        run_name: str = "",
        device_config_name: str = "",
    ) -> SimulatedLocalJob:
        if program_id is None:
            program_id = self._create_id(id_type='program')
        if job_id is None:
            job_id = self._create_id(id_type='job')
        self._program_validator([program], [params], repetitions, CIRCUIT_SERIALIZER)
        self._programs[program_id] = SimulatedLocalProgram(
            program_id=program_id,
            simulation_type=self._simulation_type,
            circuits=[program],
            processor=self,
            engine=self.engine(),
        )
        job = SimulatedLocalJob(
            job_id=job_id,
            processor_id=self.processor_id,
            parent_program=self._programs[program_id],
            repetitions=repetitions,
            sweeps=[params],
            sampler=self._sampler,
            simulation_type=self._simulation_type,
        )
        self._programs[program_id].add_job(job_id, job)
        return job

    async def run_calibration_async(self, *args, **kwargs):
        raise NotImplementedError
