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
    engine = cirq_google.Engine(project_id='my-project-id')
    program = engine.create_program(circuit)
    result0 = program.run(params=params0, repetitions=10)
    result1 = program.run(params=params1, repetitions=10)

In order to run on must have access to the Quantum Engine API. Access to this
API is (as of June 22, 2018) restricted to invitation only.
"""

from __future__ import annotations

import datetime
import enum
import random
import string
from typing import Dict, List, Optional, Set, TYPE_CHECKING, TypeVar, Union

import duet
import google.auth
from google.protobuf import any_pb2

import cirq
from cirq_google.api import v2
from cirq_google.engine import (
    abstract_engine,
    abstract_program,
    engine_client,
    engine_job,
    engine_processor,
    engine_program,
    util,
)
from cirq_google.serialization import CIRCUIT_SERIALIZER, Serializer

if TYPE_CHECKING:
    import google.protobuf

    import cirq_google
    from cirq_google.cloud import quantum

TYPE_PREFIX = 'type.googleapis.com/'

_R = TypeVar('_R')


class ProtoVersion(enum.Enum):
    """Protocol buffer version to use for requests to the quantum engine."""

    UNDEFINED = 0
    V1 = 1
    V2 = 2


def _make_random_id(prefix: str, length: int = 16):
    random_digits = [random.choice(string.ascii_uppercase + string.digits) for _ in range(length)]
    suffix = ''.join(random_digits)
    suffix += datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    return f'{prefix}{suffix}'


@cirq.value_equality
class EngineContext:
    """Context for running against the Quantum Engine API. Most users should
    simply create an Engine object instead of working with one of these
    directly."""

    def __init__(
        self,
        proto_version: Optional[ProtoVersion] = None,
        service_args: Optional[Dict] = None,
        verbose: Optional[bool] = None,
        client: Optional[engine_client.EngineClient] = None,
        timeout: Optional[int] = None,
        serializer: Serializer = CIRCUIT_SERIALIZER,
        # TODO(#5996) Remove enable_streaming once the feature is stable.
        enable_streaming: bool = True,
    ) -> None:
        """Context and client for using Quantum Engine.

        Args:
            proto_version: The version of cirq protos to use. If None, then
                ProtoVersion.V2 will be used.
            service_args: A dictionary of arguments that can be used to
                configure options on the underlying client.
            verbose: Suppresses stderr messages when set to False. Default is
                true.
            client: The engine client to use, if not supplied one will be
                created.
            timeout: Timeout for polling for results, in seconds.  Default is
                to never timeout.
            serializer: Used to serialize circuits when running jobs.
            enable_streaming: Feature gate for making Quantum Engine requests using the stream RPC.
                If True, the Quantum Engine streaming RPC is used for creating jobs
                and getting results. Otherwise, unary RPCs are used.

        Raises:
            ValueError: If either `service_args` and `verbose` were supplied
                or `client` was supplied, or if proto version 1 is specified.
        """
        if (service_args or verbose) and client:
            raise ValueError('either specify service_args and verbose or client')

        self.proto_version = proto_version or ProtoVersion.V2
        if self.proto_version == ProtoVersion.V1:
            raise ValueError('ProtoVersion V1 no longer supported')
        self.serializer = serializer
        self.enable_streaming = enable_streaming

        if not client:
            client = engine_client.EngineClient(service_args=service_args, verbose=verbose)
        self.client = client
        self.timeout = timeout

    def copy(self) -> EngineContext:
        return EngineContext(proto_version=self.proto_version, client=self.client)

    def _value_equality_values_(self):
        return self.proto_version, self.client

    def _serialize_program(self, program: cirq.AbstractCircuit) -> any_pb2.Any:
        if not isinstance(program, cirq.AbstractCircuit):
            raise TypeError(f'Unrecognized program type: {type(program)}')
        if self.proto_version != ProtoVersion.V2:
            raise ValueError(f'invalid program proto version: {self.proto_version}')
        return util.pack_any(self.serializer.serialize(program))

    def _serialize_run_context(self, sweeps: cirq.Sweepable, repetitions: int) -> any_pb2.Any:
        if self.proto_version != ProtoVersion.V2:
            raise ValueError(f'invalid run context proto version: {self.proto_version}')
        return util.pack_any(v2.run_context_to_proto(sweeps, repetitions))


class Engine(abstract_engine.AbstractEngine):
    """Runs programs via the Quantum Engine API.

    This class has methods for creating programs and jobs that execute on
    Quantum Engine:

    *   create_program
    *   run
    *   run_sweep

    Another set of methods return information about programs and jobs that
    have been previously created on the Quantum Engine, as well as metadata
    about available processors:

    *   get_program
    *   list_processors
    *   get_processor

    """

    def __init__(
        self,
        project_id: str,
        proto_version: Optional[ProtoVersion] = None,
        service_args: Optional[Dict] = None,
        verbose: Optional[bool] = None,
        timeout: Optional[int] = None,
        context: Optional[EngineContext] = None,
    ) -> None:
        """Supports creating and running programs against the Quantum Engine.

        Args:
            project_id: A project_id string of the Google Cloud Project to use.
                API interactions will be attributed to this project and any
                resources created will be owned by the project. See
                https://cloud.google.com/resource-manager/docs/creating-managing-projects#identifying_projects
            proto_version: The version of cirq protos to use. If None, then
                ProtoVersion.V2 will be used.
            service_args: A dictionary of arguments that can be used to
                configure options on the underlying client.
            verbose: Suppresses stderr messages when set to False. Default is
                true.
            timeout: Timeout for polling for results, in seconds.  Default is
                to never timeout.
            context: Engine configuration and context to use. For most users
                this should never be specified.

        Raises:
            ValueError: If context is provided and one of proto_version, service_args, or verbose.
        """
        if context and (proto_version or service_args or verbose):
            raise ValueError('Either provide context or proto_version, service_args and verbose.')

        self.project_id = project_id
        if not context:
            context = EngineContext(
                proto_version=proto_version,
                service_args=service_args,
                verbose=verbose,
                timeout=timeout,
            )
        self.context = context

    def __str__(self) -> str:
        return f'Engine(project_id={self.project_id!r})'

    def run(
        self,
        program: cirq.AbstractCircuit,
        processor_id: str,
        program_id: Optional[str] = None,
        job_id: Optional[str] = None,
        param_resolver: cirq.ParamResolver = cirq.ParamResolver({}),
        repetitions: int = 1,
        program_description: Optional[str] = None,
        program_labels: Optional[Dict[str, str]] = None,
        job_description: Optional[str] = None,
        job_labels: Optional[Dict[str, str]] = None,
        *,
        run_name: str = "",
        snapshot_id: str = "",
        device_config_name: str = "",
    ) -> cirq.Result:
        """Runs the supplied Circuit via Quantum Engine.

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
            param_resolver: Parameters to run with the program.
            repetitions: The number of repetitions to simulate.
            program_description: An optional description to set on the program.
            program_labels: Optional set of labels to set on the program.
            job_description: An optional description to set on the job.
            job_labels: Optional set of labels to set on the job.
            processor_id: Processor id for running the program.
            run_name: A unique identifier representing an automation run for the
                specified processor. An Automation Run contains a collection of
                device configurations for a processor. If specified, `processor_id`
                is required to be set.
            snapshot_id: A unique identifier for an immutable snapshot reference.
                A snapshot contains a collection of device configurations for the
                processor.
            device_config_name: An identifier used to select the processor configuration
                utilized to run the job. A configuration identifies the set of
                available qubits, couplers, and supported gates in the processor.
                If specified, `processor_id` is required to be set.

        Returns:
            A single Result for this run.

        Raises:
            ValueError: If no gate set is provided.
            ValueError: If only one of `run_name` and `device_config_name` are specified.
            ValueError: If either `run_name` and `device_config_name` are set but
                `processor_id` is empty.
        """
        return list(
            self.run_sweep(
                program=program,
                program_id=program_id,
                job_id=job_id,
                params=[param_resolver],
                repetitions=repetitions,
                processor_id=processor_id,
                program_description=program_description,
                program_labels=program_labels,
                job_description=job_description,
                job_labels=job_labels,
                run_name=run_name,
                snapshot_id=snapshot_id,
                device_config_name=device_config_name,
            )
        )[0]

    async def run_sweep_async(
        self,
        program: cirq.AbstractCircuit,
        processor_id: str,
        program_id: Optional[str] = None,
        job_id: Optional[str] = None,
        params: cirq.Sweepable = None,
        repetitions: int = 1,
        program_description: Optional[str] = None,
        program_labels: Optional[Dict[str, str]] = None,
        job_description: Optional[str] = None,
        job_labels: Optional[Dict[str, str]] = None,
        *,
        run_name: str = "",
        snapshot_id: str = "",
        device_config_name: str = "",
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
            processor_id: Processor id for running the program.
            run_name: A unique identifier representing an automation run for the
                specified processor. An Automation Run contains a collection of
                device configurations for a processor. If specified, `processor_id`
                is required to be set.
            snapshot_id: A unique identifier for an immutable snapshot reference.
                A snapshot contains a collection of device configurations for the
                processor.
            device_config_name: An identifier used to select the processor configuration
                utilized to run the job. A configuration identifies the set of
                available qubits, couplers, and supported gates in the processor.
                If specified, `processor_id` is required to be set.

        Returns:
            An EngineJob. If this is iterated over it returns a list of
            TrialResults, one for each parameter sweep.

        Raises:
            ValueError: If no gate set is provided.
            ValueError: If  only one of `run_name` and `device_config_name` are specified.
            ValueError: If either `run_name` and `device_config_name` are set but
                `processor_id` is empty.
        """

        if self.context.enable_streaming:
            if not program_id:
                program_id = _make_random_id('prog-')
            if not job_id:
                job_id = _make_random_id('job-')
            run_context = self.context._serialize_run_context(params, repetitions)

            job_result_future = self.context.client.run_job_over_stream(
                project_id=self.project_id,
                program_id=str(program_id),
                program_description=program_description,
                program_labels=program_labels,
                code=self.context._serialize_program(program),
                job_id=str(job_id),
                run_context=run_context,
                job_description=job_description,
                job_labels=job_labels,
                processor_id=processor_id,
                run_name=run_name,
                snapshot_id=snapshot_id,
                device_config_name=device_config_name,
            )
            return engine_job.EngineJob(
                self.project_id,
                str(program_id),
                str(job_id),
                self.context,
                job_result_future=job_result_future,
            )

        engine_program = await self.create_program_async(
            program, program_id, description=program_description, labels=program_labels
        )
        return await engine_program.run_sweep_async(
            job_id=job_id,
            params=params,
            repetitions=repetitions,
            processor_id=processor_id,
            description=job_description,
            labels=job_labels,
            run_name=run_name,
            snapshot_id=snapshot_id,
            device_config_name=device_config_name,
        )

    run_sweep = duet.sync(run_sweep_async)

    async def create_program_async(
        self,
        program: cirq.AbstractCircuit,
        program_id: Optional[str] = None,
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
            description: An optional description to set on the program.
            labels: Optional set of labels to set on the program.

        Returns:
            A EngineProgram for the newly created program.

        Raises:
            ValueError: If no gate set is provided.
        """
        if not program_id:
            program_id = _make_random_id('prog-')

        new_program_id, new_program = await self.context.client.create_program_async(
            self.project_id,
            program_id,
            code=self.context._serialize_program(program),
            description=description,
            labels=labels,
        )

        return engine_program.EngineProgram(
            self.project_id, new_program_id, self.context, new_program
        )

    create_program = duet.sync(create_program_async)

    def get_program(self, program_id: str) -> engine_program.EngineProgram:
        """Returns an EngineProgram for an existing Quantum Engine program.

        Args:
            program_id: Unique ID of the program within the parent project.

        Returns:
            A EngineProgram for the program.
        """
        return engine_program.EngineProgram(self.project_id, program_id, self.context)

    async def list_programs_async(
        self,
        created_before: Optional[Union[datetime.datetime, datetime.date]] = None,
        created_after: Optional[Union[datetime.datetime, datetime.date]] = None,
        has_labels: Optional[Dict[str, str]] = None,
    ) -> List[abstract_program.AbstractProgram]:
        """Returns a list of previously executed quantum programs.

        Args:
            created_after: retrieve programs that were created after this date
                or time.
            created_before: retrieve programs that were created after this date
                or time.
            has_labels: retrieve programs that have labels on them specified by
                this dict. If the value is set to `*`, filters having the label
                regardless of the label value will be filtered. For example, to
                query programs that have the shape label and have the color
                label with value red can be queried using
                `{'color: red', 'shape:*'}`
        """

        client = self.context.client
        response = await client.list_programs_async(
            self.project_id,
            created_before=created_before,
            created_after=created_after,
            has_labels=has_labels,
        )
        return [
            engine_program.EngineProgram(
                project_id=engine_client._ids_from_program_name(p.name)[0],
                program_id=engine_client._ids_from_program_name(p.name)[1],
                _program=p,
                context=self.context,
            )
            for p in response
        ]

    list_programs = duet.sync(list_programs_async)

    async def list_jobs_async(
        self,
        created_before: Optional[Union[datetime.datetime, datetime.date]] = None,
        created_after: Optional[Union[datetime.datetime, datetime.date]] = None,
        has_labels: Optional[Dict[str, str]] = None,
        execution_states: Optional[Set[quantum.ExecutionStatus.State]] = None,
    ):
        """Returns the list of jobs in the project.

        All historical jobs can be retrieved using this method and filtering
        options are available too, to narrow down the search baesd on:
          * creation time
          * job labels
          * execution states

        Args:
            created_after: retrieve jobs that were created after this date
                or time.
            created_before: retrieve jobs that were created after this date
                or time.
            has_labels: retrieve jobs that have labels on them specified by
                this dict. If the value is set to `*`, filters having the label
                regardless of the label value will be filtered. For example, to
                query programs that have the shape label and have the color
                label with value red can be queried using

                {'color': 'red', 'shape':'*'}

            execution_states: retrieve jobs that have an execution state  that
                 is contained in `execution_states`. See
                 `quantum.ExecutionStatus.State` enum for accepted values.
        """
        client = self.context.client
        response = await client.list_jobs_async(
            self.project_id,
            None,
            created_before=created_before,
            created_after=created_after,
            has_labels=has_labels,
            execution_states=execution_states,
        )
        return [
            engine_job.EngineJob(
                project_id=engine_client._ids_from_job_name(j.name)[0],
                program_id=engine_client._ids_from_job_name(j.name)[1],
                job_id=engine_client._ids_from_job_name(j.name)[2],
                context=self.context,
                _job=j,
            )
            for j in response
        ]

    list_jobs = duet.sync(list_jobs_async)

    async def list_processors_async(self) -> List[engine_processor.EngineProcessor]:
        """Returns a list of Processors that the user has visibility to in the
        current Engine project. The names of these processors are used to
        identify devices when scheduling jobs and gathering calibration metrics.

        Returns:
            A list of EngineProcessors to access status, device and calibration
            information.
        """
        response = await self.context.client.list_processors_async(self.project_id)
        return [
            engine_processor.EngineProcessor(
                self.project_id, engine_client._ids_from_processor_name(p.name)[1], self.context, p
            )
            for p in response
        ]

    list_processors = duet.sync(list_processors_async)

    def get_processor(self, processor_id: str) -> engine_processor.EngineProcessor:
        """Returns an EngineProcessor for a Quantum Engine processor.

        Args:
            processor_id: The processor unique identifier.

        Returns:
            A EngineProcessor for the processor.
        """
        return engine_processor.EngineProcessor(self.project_id, processor_id, self.context)

    def get_sampler(
        self,
        processor_id: Union[str, List[str]],
        run_name: str = "",
        device_config_name: str = "",
        snapshot_id: str = "",
    ) -> cirq_google.ProcessorSampler:
        """Returns a sampler backed by the engine.

        Args:
            processor_id: String identifier of which processor should be used to sample.
            run_name: A unique identifier representing an automation run for the
                processor. An Automation Run contains a collection of device
                configurations for the processor.
            device_config_name: An identifier used to select the processor configuration
                utilized to run the job. A configuration identifies the set of
                available qubits, couplers, and supported gates in the processor.
            snapshot_id: A unique identifier for an immutable snapshot reference. A
                snapshot contains a collection of device configurations for the processor.

        Returns:
            A `cirq.Sampler` instance (specifically a `engine_sampler.ProcessorSampler`
            that will send circuits to the Quantum Computing Service
            when sampled.

        Raises:
            ValueError: if a list of processors is provided.  This is no longer supported.
        """
        if not isinstance(processor_id, str):
            raise ValueError(
                f'Passing a list of processors ({processor_id}) '
                'to get_sampler() no longer supported. Use Engine.run() instead if '
                'you need to specify a list.'
            )
        return self.get_processor(processor_id).get_sampler(
            run_name=run_name, device_config_name=device_config_name, snapshot_id=snapshot_id
        )


def get_engine(project_id: Optional[str] = None) -> Engine:
    """Get an Engine instance assuming some sensible defaults.

    This uses the environment variable GOOGLE_CLOUD_PROJECT for the Engine
    project_id, unless set explicitly. By using an environment variable,
    you can avoid hard-coding the project_id in shared code.

    If the environment variables are set, but incorrect, an authentication
    failure will occur when attempting to run jobs on the engine.

    Args:
        project_id: If set overrides the project id obtained from the
            google.auth.default().

    Returns:
        The Engine instance.

    Raises:
        OSError: If the environment variable GOOGLE_CLOUD_PROJECT is not set. This is actually
            an `EnvironmentError`, which by definition is an `OsError`.
    """
    service_args = {}
    if not project_id:
        credentials, project_id = google.auth.default()
        service_args['credentials'] = credentials
    if not project_id:
        raise EnvironmentError(
            'Unable to determine project id. Please set environment variable GOOGLE_CLOUD_PROJECT '
            'or configure default project with `gcloud set project <project_id>`.'
        )

    return Engine(project_id=project_id, service_args=service_args)


def get_engine_device(processor_id: str, project_id: Optional[str] = None) -> cirq.Device:
    """Returns a `Device` object for a given processor.

    This is a short-cut for creating an engine object, getting the
    processor object, and retrieving the device.
    """
    return get_engine(project_id).get_processor(processor_id).get_device()


def get_engine_calibration(
    processor_id: str, project_id: Optional[str] = None
) -> Optional[cirq_google.Calibration]:
    """Returns calibration metrics for a given processor.

    This is a short-cut for creating an engine object, getting the
    processor object, and retrieving the current calibration.
    May return None if no calibration metrics exist for the device.
    """
    return get_engine(project_id).get_processor(processor_id).get_current_calibration()


def get_engine_sampler(
    processor_id: str, project_id: Optional[str] = None
) -> cirq_google.ProcessorSampler:
    """Get an EngineSampler assuming some sensible defaults.

    This uses the environment variable GOOGLE_CLOUD_PROJECT for the Engine
    project_id, unless set explicitly.

    Args:
        processor_id: Engine processor ID (from Cloud console or
            ``Engine.list_processors``).
        project_id: Optional explicit Google Cloud project id. Otherwise,
            this defaults to the environment variable GOOGLE_CLOUD_PROJECT.
            By using an environment variable, you can avoid hard-coding
            personal project IDs in shared code.

    Raises:
         EnvironmentError: If no project_id is specified and the environment
            variable GOOGLE_CLOUD_PROJECT is not set.
    """
    return get_engine(project_id).get_processor(processor_id).get_sampler()
