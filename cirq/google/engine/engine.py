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
from typing import Dict, List, Optional, Sequence, TypeVar, Union, TYPE_CHECKING

from cirq import circuits, study, value
from cirq.google import gate_sets, serializable_gate_set
from cirq.google.api import v1
from cirq.google.engine import (engine_client, engine_program, engine_job,
                                engine_processor, engine_sampler)
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


@value.value_equality
class EngineContext:
    """Context for running against the Quantum Engine API. Most users should
    simply create an Engine object instead of working with one of these
    directly."""

    def __init__(self,
                 proto_version: Optional[ProtoVersion] = None,
                 service_args: Optional[Dict] = None,
                 verbose: Optional[bool] = None,
                 client: 'Optional[engine_client.EngineClient]' = None,
                 timeout: Optional[int] = None) -> None:
        """Context and client for using Quantum Engine.

        Args:
            proto_version: The version of cirq protos to use. If None, then
                ProtoVersion.V2 will be used.
            service_args: A dictionary of arguments that can be used to
                configure options on the underlying client.
            verbose: Suppresses stderr messages when set to False. Default is
                true.
            timeout: Timeout for polling for results, in seconds.  Default is
                to never timeout.
        """
        if (service_args or verbose) and client:
            raise ValueError(
                'either specify service_args and verbose or client')

        self.proto_version = proto_version or ProtoVersion.V2

        if not client:
            client = engine_client.EngineClient(service_args=service_args,
                                                verbose=verbose)
        self.client = client
        self.timeout = timeout

    def copy(self) -> 'EngineContext':
        return EngineContext(proto_version=self.proto_version,
                             client=self.client)

    def _value_equality_values_(self):
        return self.proto_version, self.client


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
        get_program
        list_processors
        get_processor
    """

    def __init__(
            self,
            project_id: str,
            proto_version: Optional[ProtoVersion] = None,
            service_args: Optional[Dict] = None,
            verbose: Optional[bool] = None,
            context: Optional[EngineContext] = None,
            timeout: Optional[int] = None,
    ) -> None:
        """Supports creating and running programs against the Quantum Engine.

        Args:
            project_id: A project_id string of the Google Cloud Project to use.
                API interactions will be attributed to this project and any
                resources created will be owned by the project. See
                https://cloud.google.com/resource-manager/docs/creating-managing-projects#identifying_projects
            context: Engine configuration and context to use.
            proto_version: The version of cirq protos to use. If None, then
                ProtoVersion.V2 will be used.
            service_args: A dictionary of arguments that can be used to
                configure options on the underlying client.
            verbose: Suppresses stderr messages when set to False. Default is
                true.
            timeout: Timeout for polling for results, in seconds.  Default is
                to never timeout.
        """
        if context and (proto_version or service_args or verbose):
            raise ValueError(
                'either provide context or proto_version, service_args'
                ' and verbose')

        self.project_id = project_id
        if not context:
            context = EngineContext(proto_version=proto_version,
                                    service_args=service_args,
                                    verbose=verbose,
                                    timeout=timeout)
        self.context = context

    def run(
            self,
            program: 'cirq.Circuit',
            program_id: Optional[str] = None,
            job_id: Optional[str] = None,
            param_resolver: study.ParamResolver = study.ParamResolver({}),
            repetitions: int = 1,
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
            job_id: Job identifier to use. If this is not provided, a random id
                of the format 'job-################YYMMDD' will be generated,
                where # is alphanumeric and YYMMDD is the current year, month,
                and day.
            param_resolver: Parameters to run with the program.
            repetitions: The number of repetitions to simulate.
            processor_ids: The engine processors that should be candidates
                to run the program. Only one of these will be scheduled for
                execution.
            gate_set: The gate set used to serialize the circuit. The gate set
                must be supported by the selected processor.
            program_description: An optional description to set on the program.
            program_labels: Optional set of labels to set on the program.
            job_description: An optional description to set on the job.
            job_labels: Optional set of labels to set on the job.

        Returns:
            A single TrialResult for this run.
        """
        gate_set = gate_set or gate_sets.XMON
        return list(
            self.run_sweep(program=program,
                           program_id=program_id,
                           job_id=job_id,
                           params=[param_resolver],
                           repetitions=repetitions,
                           processor_ids=processor_ids,
                           gate_set=gate_set,
                           program_description=program_description,
                           program_labels=program_labels,
                           job_description=job_description,
                           job_labels=job_labels))[0]

    def run_sweep(
            self,
            program: 'cirq.Circuit',
            program_id: Optional[str] = None,
            job_id: Optional[str] = None,
            params: study.Sweepable = None,
            repetitions: int = 1,
            processor_ids: Sequence[str] = ('xmonsim',),
            gate_set: serializable_gate_set.SerializableGateSet = None,
            program_description: Optional[str] = None,
            program_labels: Optional[Dict[str, str]] = None,
            job_description: Optional[str] = None,
            job_labels: Optional[Dict[str, str]] = None,
    ) -> engine_job.EngineJob:
        """Runs the supplied Circuit via Quantum Engine.Creates

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
            processor_ids: The engine processors that should be candidates
                to run the program. Only one of these will be scheduled for
                execution.
            gate_set: The gate set used to serialize the circuit. The gate set
                must be supported by the selected processor.
            program_description: An optional description to set on the program.
            program_labels: Optional set of labels to set on the program.
            job_description: An optional description to set on the job.
            job_labels: Optional set of labels to set on the job.

        Returns:
            An EngineJob. If this is iterated over it returns a list of
            TrialResults, one for each parameter sweep.
        """
        gate_set = gate_set or gate_sets.XMON
        engine_program = self.create_program(program, program_id, gate_set,
                                             program_description,
                                             program_labels)
        return engine_program.run_sweep(job_id=job_id,
                                        params=params,
                                        repetitions=repetitions,
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
            labels: Optional set of labels to set on the program.

        Returns:
            A EngineProgram for the newly created program.
        """
        gate_set = gate_set or gate_sets.XMON

        if not program_id:
            program_id = _make_random_id('prog-')

        new_program_id, new_program = self.context.client.create_program(
            self.project_id,
            program_id,
            code=self._serialize_program(program, gate_set),
            description=description,
            labels=labels)

        return engine_program.EngineProgram(self.project_id, new_program_id,
                                            self.context, new_program)

    def _serialize_program(
            self,
            program: 'cirq.Circuit',
            gate_set: serializable_gate_set.SerializableGateSet = None
    ) -> qtypes.any_pb2.Any:
        gate_set = gate_set or gate_sets.XMON
        code = qtypes.any_pb2.Any()

        if not isinstance(program, circuits.Circuit):
            raise TypeError(f'Unrecognized program type: {type(program)}')
        program.device.validate_circuit(program)

        if self.context.proto_version == ProtoVersion.V1:
            code.Pack(
                v1.program_pb2.Program(operations=[
                    op for op in v1.circuit_as_schedule_to_protos(program)
                ]))
        elif self.context.proto_version == ProtoVersion.V2:
            program = gate_set.serialize(program)
            code.Pack(program)
        else:
            raise ValueError('invalid program proto version: {}'.format(
                self.context.proto_version))
        return code

    def get_program(self, program_id: str) -> engine_program.EngineProgram:
        """Returns an EngineProgram for an existing Quantum Engine program.

        Args:
            program_id: Unique ID of the program within the parent project.

        Returns:
            A EngineProgram for the program.
        """
        return engine_program.EngineProgram(self.project_id, program_id,
                                            self.context)

    def list_processors(self) -> List[engine_processor.EngineProcessor]:
        """Returns a list of Processors that the user has visibility to in the
        current Engine project. The names of these processors are used to
        identify devices when scheduling jobs and gathering calibration metrics.

        Returns:
            A list of EngineProcessors to access status, device and calibration
            information.
        """
        response = self.context.client.list_processors(self.project_id)
        return [
            engine_processor.EngineProcessor(
                self.project_id,
                self.context.client._ids_from_processor_name(p.name)[1],
                self.context, p) for p in response
        ]

    def get_processor(self,
                      processor_id: str) -> engine_processor.EngineProcessor:
        """Returns an EngineProcessor for a Quantum Engine processor.

        Args:
            processor_id: The processor unique identifier.

        Returns:
            A EngineProcessor for the processor.
        """
        return engine_processor.EngineProcessor(self.project_id, processor_id,
                                                self.context)

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
