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
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar, \
    Union, TYPE_CHECKING
import warnings

from google.api_core.exceptions import GoogleAPICallError, NotFound

from cirq import circuits, study, value
from cirq.google import gate_sets, serializable_gate_set
from cirq.google.api import v1, v2
from cirq.google.engine import (calibration, engine_job, engine_program,
                                engine_sampler)
from cirq.google.engine.client import quantum
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


class EngineException(Exception):

    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)


def _make_random_id(prefix: str, length: int = 16):
    random_digits = [
        random.choice(string.ascii_uppercase + string.digits)
        for _ in range(length)
    ]
    suffix = ''.join(random_digits)
    suffix += datetime.date.today().strftime('%y%m%d')
    return '%s%s' % (prefix, suffix)


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

    def __init__(self,
                 project_id: str,
                 proto_version: ProtoVersion = ProtoVersion.V1,
                 service_args: Optional[Dict] = None,
                 verbose: bool = True) -> None:
        """Engine service client.

        Args:
            project_id: A project_id string of the Google Cloud Project to use.
                API interactions will be attributed to this project and any
                resources created will be owned by the project. See
                https://cloud.google.com/resource-manager/docs/creating-managing-projects#identifying_projects
            service_args: A dictionary of arguments that can be used to
                configure options on the underlying apiclient. See
                https://github.com/googleapis/google-api-python-client
            verbose: Supresses stderr messages when set to False. Default is
                true.
        """
        self.project_id = project_id
        self.max_retry_delay = 3600  # 1 hour
        self.proto_version = proto_version
        self.verbose = verbose

        if not service_args:
            service_args = {}

        # Suppress warnings about using Application Default Credentials.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.client = quantum.QuantumEngineServiceClient(**service_args)

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
            gate_set: serializable_gate_set.SerializableGateSet = None
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
            job_config: Configures the names and properties of jobs.
            param_resolver: Parameters to run with the program.
            repetitions: The number of repetitions to simulate.
            priority: The priority to run at, 0-100.
            processor_ids: The engine processors that should be candidates
                to run the program. Only one of these will be scheduled for
                execution.
            gate_set: The gate set used to serialize the circuit. The gate set
                must be supported by the selected processor.

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
                           gate_set=gate_set))[0]

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
            gate_set: serializable_gate_set.SerializableGateSet = None
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
            job_config: Configures the names and properties of jobs.
            params: Parameters to run with the program.
            repetitions: The number of circuit repetitions to run.
            priority: The priority to run at, 0-100.
            processor_ids: The engine processors that should be candidates
                to run the program. Only one of these will be scheduled for
                execution.

        Returns:
            An EngineJob. If this is iterated over it returns a list of
            TrialResults, one for each parameter sweep.
        """
        gate_set = gate_set or gate_sets.XMON
        engine_program = self.create_program(program, program_id, gate_set)
        return engine_program.run_sweep(job_config=job_config,
                                        params=params,
                                        repetitions=repetitions,
                                        priority=priority,
                                        processor_ids=processor_ids)

    def create_job(
            self,
            *,  # Force keyword args.
            program_name: str,
            job_config: Optional[JobConfig] = None,
            params: study.Sweepable = None,
            repetitions: int = 1,
            priority: int = 500,
            processor_ids: Sequence[str] = ('xmonsim',),
            gate_set: serializable_gate_set.SerializableGateSet = None
    ) -> engine_job.EngineJob:
        gate_set = gate_set or gate_sets.XMON

        # Check program to run and program parameters.
        if not 0 <= priority < 1000:
            raise ValueError('priority must be between 0 and 1000')

        job_config = self.implied_job_config(job_config)
        sweeps = study.to_sweeps(params or study.ParamResolver({}))
        run_context = self._serialize_run_context(sweeps, repetitions)

        # Create job.
        request = qtypes.QuantumJob(
            name='%s/jobs/%s' % (program_name, job_config.job_id),
            scheduling_config=qtypes.SchedulingConfig(
                priority=priority,
                processor_selector=qtypes.SchedulingConfig.ProcessorSelector(
                    processor_names=[
                        'projects/%s/processors/%s' %
                        (self.project_id, processor_id)
                        for processor_id in processor_ids
                    ])),
            run_context=run_context)
        response = self._make_request(lambda: self.client.create_quantum_job(
            program_name, request, False))

        return engine_job.EngineJob(job_config, response, self)

    def implied_job_config(self, job_config: Optional[JobConfig]) -> JobConfig:
        implied_job_config = (JobConfig()
                              if job_config is None else job_config.copy())

        # Note: inference order is important. Later ones may need earlier ones.
        self._infer_job_id(implied_job_config)

        return implied_job_config

    def _infer_job_id(self, job_config: JobConfig) -> None:
        if job_config.job_id is None:
            job_config.job_id = _make_random_id('job-')

    def _make_request(self, request: Callable[[], _R]) -> _R:
        retryable_error_codes = [500, 503]
        current_delay = 0.1  #100ms

        while True:
            try:
                return request()
            except GoogleAPICallError as err:
                message = err.message
                # Raise RuntimeError for exceptions that are not retryable.
                # Otherwise, pass through to retry.
                if not err.code.value in retryable_error_codes:
                    raise EngineException(message) from err

            current_delay *= 2
            if current_delay > self.max_retry_delay:
                raise TimeoutError(
                    'Reached max retry attempts for error: {}'.format(message))
            if (self.verbose):
                print(message, file=sys.stderr)
                print('Waiting ',
                      current_delay,
                      'seconds before retrying.',
                      file=sys.stderr)
            time.sleep(current_delay)

    def _serialize_run_context(
            self,
            sweeps: List[study.Sweep],
            repetitions: int,
    ) -> qtypes.any_pb2.Any:
        context = qtypes.any_pb2.Any()
        proto_version = self.proto_version
        if proto_version == ProtoVersion.V1:
            context.Pack(
                v1.program_pb2.RunContext(parameter_sweeps=[
                    v1.sweep_to_proto(sweep, repetitions) for sweep in sweeps
                ]))
        elif proto_version == ProtoVersion.V2:
            run_context = v2.run_context_pb2.RunContext()
            for sweep in sweeps:
                sweep_proto = run_context.parameter_sweeps.add()
                sweep_proto.repetitions = repetitions
                v2.sweep_to_proto(sweep, out=sweep_proto.sweep)

            context.Pack(run_context)
        else:
            raise ValueError(
                'invalid run context proto version: {}'.format(proto_version))
        return context

    def create_program(
            self,
            program: 'cirq.Circuit',
            program_id: Optional[str] = None,
            gate_set: serializable_gate_set.SerializableGateSet = None
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
        """
        gate_set = gate_set or gate_sets.XMON

        if not program_id:
            program_id = _make_random_id('prog-')

        parent_name = 'projects/%s' % self.project_id
        program_name = '%s/programs/%s' % (parent_name, program_id)
        # Create program.
        request = qtypes.QuantumProgram(name=program_name,
                                        code=self._serialize_program(
                                            program, gate_set))
        result = self._make_request(lambda: self.client.create_quantum_program(
            parent_name, request, False))

        return engine_program.EngineProgram(result.name, self)

    def _serialize_program(
            self,
            program: 'cirq.Circuit',
            gate_set: serializable_gate_set.SerializableGateSet = None
    ) -> Dict[str, Any]:
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

    def get_program(self, program_id: str) -> qtypes.QuantumProgram:
        """Returns a previously created quantum program.

        Params:
            program_id: A string containing the unique ID of a program within
              the project specified for the Engine.

        Returns:
            A quantum program.
        """
        program_resource_name = self._program_name_from_id(program_id)
        return self._make_request(lambda: self.client.get_quantum_program(
            program_resource_name, False))

    def get_job(self, job_resource_name: str) -> qtypes.QuantumJob:
        """Returns a previously created job.

        See get_job_result if you want the results of the job and not just
        metadata about the job.

        Params:
            job_resource_name: A string of the form
                `projects/project_id/programs/program_id/jobs/job_id`.

        Returns:
            A quantum job.
        """
        return self._make_request(lambda: self.client.get_quantum_job(
            job_resource_name, False))

    def get_job_results(self,
                        job_resource_name: str) -> List[study.TrialResult]:
        """Returns the actual results (not metadata) of a completed job.

        Params:
            job_resource_name: A string of the form
                `projects/project_id/programs/program_id/jobs/job_id`.

        Returns:
            An iterable over the TrialResult, one per parameter in the
            parameter sweep.
        """
        response = self._make_request(lambda: self.client.get_quantum_result(
            job_resource_name))
        result = response.result
        result_type = result.type_url[len(TYPE_PREFIX):]
        if (result_type == 'cirq.google.api.v1.Result' or
                result_type == 'cirq.api.google.v1.Result'):
            v1_parsed_result = v1.program_pb2.Result()
            v1_parsed_result.ParseFromString(result.value)
            return self._get_job_results_v1(v1_parsed_result)
        if (result_type == 'cirq.google.api.v2.Result' or
                result_type == 'cirq.api.google.v2.Result'):
            v2_parsed_result = v2.result_pb2.Result()
            v2_parsed_result.ParseFromString(result.value)
            return self._get_job_results_v2(v2_parsed_result)
        raise ValueError('invalid result proto version: {}'.format(
            self.proto_version))

    def _get_job_results_v1(self, result: v1.program_pb2.Result
                           ) -> List[study.TrialResult]:
        trial_results = []
        for sweep_result in result.sweep_results:
            sweep_repetitions = sweep_result.repetitions
            key_sizes = [
                (m.key, len(m.qubits)) for m in sweep_result.measurement_keys
            ]
            for result in sweep_result.parameterized_results:
                data = result.measurement_results
                measurements = v1.unpack_results(data, sweep_repetitions,
                                                 key_sizes)

                trial_results.append(
                    study.TrialResult.from_single_parameter_set(
                        params=study.ParamResolver(result.params.assignments),
                        measurements=measurements))
        return trial_results

    def _get_job_results_v2(self, result: v2.result_pb2.Result
                           ) -> List[study.TrialResult]:
        sweep_results = v2.results_from_proto(result)
        # Flatten to single list to match to sampler api.
        return [
            trial_result for sweep_result in sweep_results
            for trial_result in sweep_result
        ]

    def cancel_job(self, job_resource_name: str):
        """Cancels the given job.

        See also the cancel method on EngineJob.

        Params:
            job_resource_name: A string of the form
                `projects/project_id/programs/program_id/jobs/job_id`.
        """
        self._make_request(lambda: self.client.cancel_quantum_job(
            job_resource_name))

    def _program_name_from_id(self, program_id: str) -> str:
        return 'projects/%s/programs/%s' % (self.project_id, program_id)

    def _set_program_labels(self, program_id: str, labels: Dict[str, str],
                            fingerprint: str):
        program_resource_name = self._program_name_from_id(program_id)
        return self._make_request(lambda: self.client.update_quantum_program(
            program_resource_name,
            qtypes.QuantumProgram(name=program_resource_name,
                                  labels=labels,
                                  label_fingerprint=fingerprint),
            qtypes.field_mask_pb2.FieldMask(paths=['labels'])))

    def set_program_labels(self, program_id: str, labels: Dict[str, str]):
        program = self.get_program(program_id)
        return self._set_program_labels(program_id, labels,
                                        program.label_fingerprint)

    def add_program_labels(self, program_id: str, labels: Dict[str, str]):
        program = self.get_program(program_id)
        old_labels = program.labels
        new_labels = dict(old_labels)
        new_labels.update(labels)
        if new_labels != old_labels:
            fingerprint = program.label_fingerprint
            return self._set_program_labels(program_id, new_labels, fingerprint)
        return program

    def remove_program_labels(self, program_id: str, label_keys: List[str]):
        program = self.get_program(program_id)
        old_labels = program.labels
        new_labels = dict(old_labels)
        for key in label_keys:
            new_labels.pop(key, None)
        if new_labels != old_labels:
            fingerprint = program.label_fingerprint
            return self._set_program_labels(program_id, new_labels, fingerprint)
        return program

    def _set_job_labels(self, job_resource_name: str, labels: Dict[str, str],
                        fingerprint: str):
        return self._make_request(lambda: self.client.update_quantum_job(
            job_resource_name,
            qtypes.QuantumJob(name=job_resource_name,
                              labels=labels,
                              label_fingerprint=fingerprint),
            qtypes.field_mask_pb2.FieldMask(paths=['labels'])))

    def set_job_labels(self, job_resource_name: str, labels: Dict[str, str]):
        job = self.get_job(job_resource_name)
        return self._set_job_labels(job_resource_name, labels,
                                    job.label_fingerprint)

    def add_job_labels(self, job_resource_name: str, labels: Dict[str, str]):
        job = self.get_job(job_resource_name)
        old_labels = job.labels
        new_labels = dict(old_labels)
        new_labels.update(labels)
        if new_labels != old_labels:
            fingerprint = job.label_fingerprint
            return self._set_job_labels(job_resource_name, new_labels,
                                        fingerprint)
        return job

    def remove_job_labels(self, job_resource_name: str, label_keys: List[str]):
        job = self.get_job(job_resource_name)
        old_labels = job.labels
        new_labels = dict(old_labels)
        for key in label_keys:
            new_labels.pop(key, None)
        if new_labels != old_labels:
            fingerprint = job.label_fingerprint
            return self._set_job_labels(job_resource_name, new_labels,
                                        fingerprint)
        return job

    def list_processors(self) -> List[qtypes.QuantumProcessor]:
        """Returns a list of Processors that the user has visibility to in the
        current Engine project. The names of these processors are used to
        identify devices when scheduling jobs and gathering calibration metrics.

        Returns:
            A list of dictionaries containing the metadata of each processor.
        """
        response = self._make_request(lambda: self.client.
                                      list_quantum_processors(
                                          'projects/%s' % self.project_id, ''))
        return list(response)

    def get_device_specification(
            self,
            processor_id: str) -> Optional[v2.device_pb2.DeviceSpecification]:
        """Returns a device specification proto for use in determining
        information about the device.

        Params:
            processor_id: The processor identifier within the resource name,
                where name has the format:
                `projects/<project_id>/processors/<processor_id>`.

        Returns:
            Device specification proto.
        """
        processor_name = 'projects/%s/processors/%s' % (self.project_id,
                                                        processor_id)
        response = self._make_request(lambda: self.client.get_quantum_processor(
            processor_name))

        device_spec = v2.device_pb2.DeviceSpecification()
        device_spec.ParseFromString(response.device_spec.value)

        return device_spec

    def get_latest_calibration(self, processor_id: str
                              ) -> Optional[calibration.Calibration]:
        """Returns metadata about the latest known calibration for a processor.

        Params:
            processor_id: The processor identifier within the resource name,
                where name has the format:
                `projects/<project_id>/processors/<processor_id>`.

        Returns:
            The calibration data or None if there is no current calibration.
        """
        calibration_name = 'projects/%s/processors/%s/calibrations/%s' % (
            self.project_id, processor_id, 'current')
        try:
            return self.get_calibration(calibration_name)
        except EngineException as err:
            if isinstance(err.__cause__, NotFound):
                return None
            raise

    def get_calibration(self, calibration_name: str) -> calibration.Calibration:
        """Retrieve metadata about a specific calibration run.

        Params:
            calibration_name: A string of the form
                `projects/<project_id>/processors/<processor id>`
                `/calibrations/<timestamp in seconds since epoch>`

        Returns:
            A dictionary containing the metadata.
        """
        response = self._make_request(lambda: self.client.
                                      get_quantum_calibration(calibration_name))
        metrics = v2.metrics_pb2.MetricsSnapshot()
        metrics.ParseFromString(response.data.value)
        return calibration.Calibration(metrics)

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
