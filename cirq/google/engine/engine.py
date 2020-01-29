# Copyright 2018 The Cirq Developers
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
"""Classes for running against Google's Quantum Cloud Service.

As an example, to run a circuit against the xmon simulator on the cloud,
    engine = cirq.google.Engine(project_id='my-project-id')
    program = engine.create_program(circuit)
    result0 = program.run(params=params0, repetitions=10)
    result1 = program.run(params=params1, repetitions=10)

In order to run on must have access to the Quantum Engine API. Access to this
API is (as of June 22, 2018) restricted to invitation only.
"""

import datetime
import enum
import random
import string
from typing import Dict, List, Optional, Sequence, TypeVar, \
    Union, TYPE_CHECKING

from cirq import circuits, study, value
from cirq.google import gate_sets, serializable_gate_set
from cirq.google.api import v1
from cirq.google.engine import calibration
from cirq.google.engine import (engine_client, engine_job, engine_processor,
                                engine_program, engine_sampler)
from cirq.google.engine.client.quantum import types as qtypes

if TYPE_CHECKING:
    import cirq

TYPE_PREFIX = 'type.googleapis.com/'

_R = TypeVar('_R')


class ProtoVersion(enum.Enum):
    """Protocol buffer version to use for requests to the quantum engine."""
    UNDEFINED = 0
    V1 = 1
    V2 = 2


def _make_random_id(prefix: str, length: int = 16):
    random_digits = [
        random.choice(string.ascii_uppercase + string.digits)
        for _ in range(length)
    ]
    suffix = ''.join(random_digits)
    suffix += datetime.date.today().strftime('%y%m%d')
    return '%s%s' % (prefix, suffix)


def _infer_job_id(job_config: 'JobConfig') -> None:
    if job_config.job_id is None:
        job_config.job_id = _make_random_id('job-')


def implied_job_config(job_config: 'Optional[JobConfig]') -> 'JobConfig':
    implied_job_config = (JobConfig()
                          if job_config is None else job_config.copy())

    # Note: inference order is important. Later ones may need earlier ones.
    _infer_job_id(implied_job_config)

    return implied_job_config


@value.value_equality
class JobConfig:
    """Configuration for a job to run on the Quantum Engine API.

    An instance of a program that has been scheduled on the Quantum Engine is
    called a Job. This object contains the configuration for a job.
    """

    def __init__(self, job_id: Optional[str] = None) -> None:
        """Configuration for a job that is run on Quantum Engine.

        Args:
            job_id: Id of the job to create, defaults to 'job-0'.
        """
        self.job_id = job_id

    def copy(self) -> 'JobConfig':
        return JobConfig(job_id=self.job_id)

    def _value_equality_values_(self):
        return (self.job_id)

    def __repr__(self):
        return ('cirq.google.JobConfig(job_id={!r})').format(self.job_id)


class Engine:
    """Runs programs via the Quantum Engine API.

    This class has methods for creating programs and jobs that execute on
    Quantum Engine:
        create_program
        run
        run_sweep

    Another set of methods return information about programs and jobs that
    have been previously created on the Quantum Engine, as well as metadata
    about available processors:
        get_calibration
        get_job
        get_job_results
        get_latest_calibration
        get_program
        list_processors

    Finally, the engine has methods to update existing programs and jobs:
        add_job_labels
        add_program_labels
        cancel_job
        remove_job_labels
        remove_program_labels
        set_job_labels
        set_program_labels
    """

    def __init__(
            self,
            project_id: str,
            proto_version: ProtoVersion = ProtoVersion.V1,
            service_args: Optional[Dict] = None,
            verbose: bool = True,
    ) -> None:
        """Engine service client.

        Args:
            project_id: A project_id string of the Google Cloud Project to use.
                API interactions will be attributed to this project and any
                resources created will be owned by the project. See
                https://cloud.google.com/resource-manager/docs/creating-managing-projects#identifying_projects
            proto_version: The version of cirq protos to use.
            service_args: A dictionary of arguments that can be used to
                configure options on the underlying apiclient. See
                https://github.com/googleapis/google-api-python-client
            verbose: Suppresses stderr messages when set to False. Default is
                true.
        """
        self.project_id = project_id
        self.proto_version = proto_version

        self.client = engine_client.EngineClient(service_args=service_args,
                                                 verbose=verbose)

    def run(
            self,
            *,  # Force keyword args.
            program: 'cirq.Circuit',
            program_id: Optional[str] = None,
            job_config: Optional[JobConfig] = None,
            param_resolver: study.ParamResolver = study.ParamResolver({}),
            repetitions: int = 1,
            priority: int = 50,
            processor_ids: Sequence[str] = ('xmonsim',),
            gate_set: serializable_gate_set.SerializableGateSet = None,
            program_description: Optional[str] = None,
            program_labels: Optional[Dict[str, str]] = None,
            job_description: Optional[str] = None,
            job_labels: Optional[Dict[str, str]] = None,
    ) -> study.TrialResult:
        """Runs the supplied Circuit via Quantum Engine.

        Args:
            program: The Circuit to execute. If a circuit is
                provided, a moment by moment schedule will be used.
            program_id: A user-provided identifier for the program. This must
                be unique within the Google Cloud project being used. If this
                parameter is not provided, a random id of the format
                'prog-################YYMMDD' will be generated, where # is
                alphanumeric and YYMMDD is the current year, month, and day.
            job_config: Configures the ids and properties of jobs.
            param_resolver: Parameters to run with the program.
            repetitions: The number of repetitions to simulate.
            priority: The priority to run at, 0-1000.
            processor_ids: The engine processors that should be candidates
                to run the program. Only one of these will be scheduled for
                execution.
            gate_set: The gate set used to serialize the circuit. The gate set
                must be supported by the selected processor.
            program_description: An optional description to set on the program.
            program_labels: Optional set of label to set on the program.
            job_description: An optional description to set on the job.
            job_labels: Optional set of label to set on the job.

        Returns:
            A single TrialResult for this run.
        """
        gate_set = gate_set or gate_sets.XMON
        return list(
            self.run_sweep(program=program,
                           program_id=program_id,
                           job_config=job_config,
                           params=[param_resolver],
                           repetitions=repetitions,
                           priority=priority,
                           processor_ids=processor_ids,
                           gate_set=gate_set,
                           program_description=program_description,
                           program_labels=program_labels,
                           job_description=job_description,
                           job_labels=job_labels))[0]

    def run_sweep(
            self,
            *,  # Force keyword args.
            program: 'cirq.Circuit',
            program_id: Optional[str] = None,
            job_config: Optional[JobConfig] = None,
            params: study.Sweepable = None,
            repetitions: int = 1,
            priority: int = 500,
            processor_ids: Sequence[str] = ('xmonsim',),
            gate_set: serializable_gate_set.SerializableGateSet = None,
            program_description: Optional[str] = None,
            program_labels: Optional[Dict[str, str]] = None,
            job_description: Optional[str] = None,
            job_labels: Optional[Dict[str, str]] = None,
    ) -> engine_job.EngineJob:
        """Runs the supplied Circuit via Quantum Engine.

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
            job_config: Configures the ids and properties of jobs.
            params: Parameters to run with the program.
            repetitions: The number of circuit repetitions to run.
            priority: The priority to run at, 0-1000.
            processor_ids: The engine processors that should be candidates
                to run the program. Only one of these will be scheduled for
                execution.
            gate_set: The gate set used to serialize the circuit. The gate set
                must be supported by the selected processor.
            program_description: An optional description to set on the program.
            program_labels: Optional set of label to set on the program.
            job_description: An optional description to set on the job.
            job_labels: Optional set of label to set on the job.

        Returns:
            An EngineJob. If this is iterated over it returns a list of
            TrialResults, one for each parameter sweep.
        """
        gate_set = gate_set or gate_sets.XMON
        engine_program = self.create_program(program, program_id, gate_set,
                                             program_description,
                                             program_labels)
        return engine_program.run_sweep(job_config=job_config,
                                        params=params,
                                        repetitions=repetitions,
                                        priority=priority,
                                        processor_ids=processor_ids,
                                        description=job_description,
                                        labels=job_labels)

    def create_program(
            self,
            program: 'cirq.Circuit',
            program_id: Optional[str] = None,
            gate_set: serializable_gate_set.SerializableGateSet = None,
            description: Optional[str] = None,
            labels: Optional[Dict[str, str]] = None,
    ) -> engine_program.EngineProgram:
        """Wraps a Circuit for use with the Quantum Engine.

        Args:
            program: The Circuit to execute.
            program_id: A user-provided identifier for the program. This must be
                unique within the Google Cloud project being used. If this
                parameter is not provided, a random id of the format
                'prog-################YYMMDD' will be generated, where # is
                alphanumeric and YYMMDD is the current year, month, and day.
            gate_set: The gate set used to serialize the circuit. The gate set
                must be supported by the selected processor
            description: An optional description to set on the program.
            labels: Optional set of label to set on the program.

        Returns:
            A EngineProgram for the newly created program.
        """
        gate_set = gate_set or gate_sets.XMON

        if not program_id:
            program_id = _make_random_id('prog-')

        created_program_id, created_program = self.client.create_program(
            self.project_id,
            program_id,
            code=self._serialize_program(program, gate_set),
            description=description,
            labels=labels)

        return engine_program.EngineProgram(created_program_id, self,
                                            created_program)

    def _serialize_program(
            self,
            program: 'cirq.Circuit',
            gate_set: serializable_gate_set.SerializableGateSet = None
    ) -> qtypes.any_pb2.Any:
        gate_set = gate_set or gate_sets.XMON
        code = qtypes.any_pb2.Any()

        if self.proto_version == ProtoVersion.V1:
            if isinstance(program, circuits.Circuit):
                program.device.validate_circuit(program)
            else:
                raise TypeError(f'Unrecognized program type: {type(program)}')
            code.Pack(
                v1.program_pb2.Program(operations=[
                    op for op in v1.circuit_as_schedule_to_protos(program)
                ]))
        elif self.proto_version == ProtoVersion.V2:
            program = gate_set.serialize(program)
            code.Pack(program)
        else:
            raise ValueError('invalid program proto version: {}'.format(
                self.proto_version))
        return code

    def get_program(self, program_id: str, project_id: Optional[str] = None
                   ) -> engine_program.EngineProgram:
        """Creates an EngineProgram for an existing Quantum Engine program.

        Args:
            program_id: Unique ID of the program within the parent project.
            project_id: The project id for the project containing the program.
                If provided will be checked against the project id of the
                engine.

        Returns:
            A EngineProgram for the program.
        """
        if project_id and project_id != self.project_id:
            raise ValueError(
                'Engine project id {} does not match given project_id {}'.
                format(self.project_id, project_id))

        program = self.client.get_program(self.project_id, program_id, False)
        return engine_program.EngineProgram(program_id, self, program)

    def get_job(self,
                program_id: str,
                job_id: str,
                project_id: Optional[str] = None) -> engine_job.EngineJob:
        """Creates an EngineJob for an existing Quantum Engine job.

        Args:
            program_id: Unique ID of the program within the parent project.
            job_id: Unique ID of the job within the parent program.
            project_id: The project id for the project containing the program.
                If provided will be checked against the project id of the
                engine.

        Returns:
            A EngineJob for the job.
        """
        return self.get_program(program_id, project_id).get_job(job_id)

    def list_processors(self) -> List[engine_processor.EngineProcessor]:
        """Returns a list of Processors that the user has visibility to in the
        current Engine project. The names of these processors are used to
        identify devices when scheduling jobs and gathering calibration metrics.

        Returns:
            A list of metadata of each processor.
        """
        response = self.client.list_processors(self.project_id)
        return [
            engine_processor.EngineProcessor(
                self.client._ids_from_processor_name(p.name)[1], self, p)
            for p in list(response)
        ]

    def get_processor(self, processor_id: str, project_id: Optional[str] = None
                     ) -> engine_processor.EngineProcessor:
        """Creates an EngineProcessor for a Quantum Engine processor.

        Args:
            processor_id: The processor unique identifier.
            project_id: The project id for the project containing the processor.
                If provided will be checked against the project id of the
                engine.

        Returns:
            A EngineProcessor for the processor.
        """
        if project_id and project_id != self.project_id:
            raise ValueError(
                'Engine project id {} does not match given project_id {}'.
                format(self.project_id, project_id))

        processor = self.client.get_processor(self.project_id, processor_id)
        return engine_processor.EngineProcessor(processor_id, self, processor)

    def get_calibration(self,
                        processor_id: str,
                        calibration_timestamp_seconds: int,
                        project_id: Optional[str] = None
                       ) -> calibration.Calibration:
        """Retrieve metadata about a specific calibration run.

        Args:
            processor_id: The processor unique identifier.
            calibration_timestamp_seconds: The timestamp of the calibration in
                seconds since epoch.
            project_id: The project id for the project containing the program.
                If provided will be checked against the project id of the
                engine.

        Returns:
            The calibration data.
        """
        return self.get_processor(
            processor_id,
            project_id).get_calibration(calibration_timestamp_seconds)

    def sampler(self, processor_id: Union[str, List[str]],
                gate_set: serializable_gate_set.SerializableGateSet
               ) -> engine_sampler.QuantumEngineSampler:
        """Returns a sampler backed by the engine.

        Args:
            processor_id: String identifier, or list of string identifiers,
                determining which processors may be used when sampling.
            gate_set: Determines how to serialize circuits when requesting
                samples.
        """
        return engine_sampler.QuantumEngineSampler(engine=self,
                                                   processor_id=processor_id,
                                                   gate_set=gate_set)
