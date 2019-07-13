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
    engine = cirq.google.Engine(api_key='mysecretapikey')
    options = cirq.google.JobConfig(project_id='my-project-id')
    result = engine.run(options, circuit, repetitions=10)

In order to run on must have access to the Quantum Engine API. Access to this
API is (as of June 22, 2018) restricted to invitation only.
"""

import base64
import enum
import random
import re
import string
import time
from collections import Iterable
from typing import Any, cast, Dict, List, Optional, Sequence, Tuple, Union

from apiclient import discovery
import google.protobuf as gp
from google.protobuf import any_pb2

from cirq import optimizers, circuits
from cirq.google import gate_sets
from cirq.api.google import v1, v2
from cirq.google.api import v2 as api_v2
from cirq.google.convert_to_xmon_gates import ConvertToXmonGates
from cirq.google.params import sweep_to_proto_dict
from cirq.google.programs import schedule_to_proto_dicts, unpack_results
from cirq.google.serializable_gate_set import SerializableGateSet
from cirq.schedules import Schedule, moment_by_moment_schedule
from cirq.study import ParamResolver, Sweep, Sweepable, TrialResult
from cirq.study.sweeps import Points, UnitSweep, Zip

gcs_prefix_pattern = re.compile('gs://[a-z0-9._/-]+')
TERMINAL_STATES = ['SUCCESS', 'FAILURE', 'CANCELLED']
TYPE_PREFIX = 'type.googleapis.com/'


class ProtoVersion(enum.Enum):
    """Protocol buffer version to use for requests to the quantum engine."""
    UNDEFINED = 0
    V1 = 1
    V2 = 2


# Quantum programs to run can be specified as circuits or schedules.
Program = Union[circuits.Circuit, Schedule]


class JobConfig:
    """Configuration for a program and job to run on the Quantum Engine API.

    Quantum engine has two resources: programs and jobs. Programs live
    under cloud projects. Every program may have many jobs, which represent
    scheduled or terminated programs executions. Program and job resources have
    string names. This object contains the information necessary to create a
    program and then create a job on Quantum Engine, hence running the program.
    Program ids are of the form
        `projects/project_id/programs/program_id`
    while job ids are of the form
        `projects/project_id/programs/program_id/jobs/job_id`
    """

    def __init__(self,
                 project_id: Optional[str] = None,
                 program_id: Optional[str] = None,
                 job_id: Optional[str] = None,
                 gcs_prefix: Optional[str] = None,
                 gcs_program: Optional[str] = None,
                 gcs_results: Optional[str] = None) -> None:
        """Configuration for a job that is run on Quantum Engine.

        Requires project_id.

        Args:
            project_id: The project id string of the Google Cloud Project to
                use. Programs and Jobs will be created under this project id.
                If this is set to None, the engine's default project id will be
                used instead. If that also isn't set, calls will fail.
            program_id: Id of the program to create, defaults to a random
                version of 'prog-ABCD'.
            job_id: Id of the job to create, defaults to 'job-0'.
            gcs_prefix: Google Cloud Storage bucket and object prefix to use
                for storing programs and results. The bucket will be created if
                needed. Must be in the form "gs://bucket-name/object-prefix/".
            gcs_program: Explicit override for the program storage location.
            gcs_results: Explicit override for the results storage location.
        """
        self.project_id = project_id
        self.program_id = program_id
        self.job_id = job_id
        self.gcs_prefix = gcs_prefix
        self.gcs_program = gcs_program
        self.gcs_results = gcs_results

    def copy(self):
        return JobConfig(
            project_id=self.project_id,
            program_id=self.program_id,
            job_id=self.job_id,
            gcs_prefix=self.gcs_prefix,
            gcs_program=self.gcs_program,
            gcs_results=self.gcs_results)

    def __repr__(self):
        return ('JobConfig(project_id={!r}, '
                'program_id={!r}, '
                'job_id={!r}, '
                'gcs_prefix={!r}, '
                'gcs_program={!r}, '
                'gcs_results={!r})').format(self.project_id, self.program_id,
                                            self.job_id, self.gcs_prefix,
                                            self.gcs_program, self.gcs_results)


class Engine:
    """Runs programs via the Quantum Engine API.

    This class has methods for creating programs and jobs that execute on
    Quantum Engine:
        run
        run_sweep

    Another set of methods return information about programs and jobs that
    have been previously created on the Quantum Engine:
        get_program
        get_job
        get_job_results

    Finally, the engine has methods to update existing programs and jobs:
        cancel_job
        set_program_labels
        add_program_labels
        remove_program_labels
        set_job_labels
        add_job_labels
        remove_job_labels
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 api: str = 'quantum',
                 version: str = 'v1alpha1',
                 default_project_id: Optional[str] = None,
                 discovery_url: Optional[str] = None,
                 default_gcs_prefix: Optional[str] = None,
                 proto_version: ProtoVersion = ProtoVersion.V1,
                 **kwargs) -> None:
        """Engine service client.

        Args:
            api_key: API key to use to retrieve discovery doc.
            api: API name.
            version: API version.
            default_project_id: A fallback project_id to use when one isn't
                specified in the JobConfig given to 'run' methods.
                See JobConfig for more information on project_id.
            discovery_url: Discovery url for the API. If not supplied, uses
                Google's default api.googleapis.com endpoint.
            default_gcs_prefix: A fallback gcs_prefix to use when one isn't
                specified in the JobConfig given to 'run' methods.
                See JobConfig for more information on gcs_prefix.
        """
        self.api_key = api_key
        self.api = api
        self.default_project_id = default_project_id
        self.version = version
        self.discovery_url = discovery_url or ('https://{api}.googleapis.com/'
                                               '$discovery/rest'
                                               '?version={apiVersion}')
        self.default_gcs_prefix = default_gcs_prefix
        self.proto_version = proto_version

        discovery_service_url = self.discovery_url
        if self.api_key is not None:
            discovery_service_url += '&key={}'.format(self.api_key)
        self.service = discovery.build(
            self.api,
            self.version,
            discoveryServiceUrl=discovery_service_url,
            **kwargs)

    def run(
            self,
            *,  # Force keyword args.
            program: Program,
            job_config: Optional[JobConfig] = None,
            param_resolver: ParamResolver = ParamResolver({}),
            repetitions: int = 1,
            priority: int = 50,
            processor_ids: Sequence[str] = ('xmonsim',),
            gate_set: SerializableGateSet = gate_sets.XMON) -> TrialResult:
        """Runs the supplied Circuit or Schedule via Quantum Engine.

        Args:
            program: The Circuit or Schedule to execute. If a circuit is
                provided, a moment by moment schedule will be used.
            job_config: Configures the names of programs and jobs.
            param_resolver: Parameters to run with the program.
            repetitions: The number of repetitions to simulate.
            priority: The priority to run at, 0-100.
            processor_ids: The engine processors to run against.
            gate_set: The gate set used to serialize the circuit. The gate set
                must be supported by the selected processor.

        Returns:
            A single TrialResult for this run.
        """
        return list(
            self.run_sweep(program=program,
                           job_config=job_config,
                           params=[param_resolver],
                           repetitions=repetitions,
                           priority=priority,
                           processor_ids=processor_ids,
                           gate_set=gate_set))[0]

    def _infer_project_id(self, job_config) -> None:
        if job_config.project_id is not None:
            return

        if self.default_project_id is not None:
            job_config.project_id = self.default_project_id
            return

        raise ValueError(
            "Need a cloud project id. "
            "This engine has default_project_id=None and "
            "the given JobConfig has project_id=None. "
            "One or the other must be set.")

    def _infer_gcs_prefix(self, job_config: JobConfig) -> None:
        if job_config.project_id is None:
            raise ValueError("Must infer project_id before gcs_prefix.")
        project_id = cast(str, job_config.project_id)

        gcs_prefix = (
            job_config.gcs_prefix or
            self.default_gcs_prefix or
            'gs://gqe-' + project_id[project_id.rfind(':') + 1:]
        )
        if gcs_prefix and not gcs_prefix.endswith('/'):
            gcs_prefix += '/'

        if not gcs_prefix_pattern.match(gcs_prefix):
            raise ValueError('gcs_prefix must be of the form "gs://'
                             '<bucket name and optional object prefix>/"')

        job_config.gcs_prefix = gcs_prefix

    def _infer_program_id(self, job_config: JobConfig) -> None:
        if job_config.program_id is not None:
            return

        random_digits = [
            random.choice(string.ascii_uppercase + string.digits)
            for _ in range(6)
        ]
        job_config.program_id = 'prog-' + ''.join(random_digits)

    def _infer_job_id(self, job_config: JobConfig) -> None:
        if job_config.job_id is None:
            job_config.job_id = 'job-0'

    def _infer_gcs_program(self, job_config: JobConfig) -> None:
        if job_config.gcs_prefix is None:
            raise ValueError("Must infer gcs_prefix before gcs_program.")

        if job_config.gcs_program is None:
            job_config.gcs_program = '{}programs/{}/{}'.format(
                job_config.gcs_prefix,
                job_config.program_id,
                job_config.program_id)

    def _infer_gcs_results(self, job_config: JobConfig) -> None:
        if job_config.gcs_prefix is None:
            raise ValueError("Must infer gcs_prefix before gcs_results.")

        if job_config.gcs_results is None:
            job_config.gcs_results = '{}programs/{}/jobs/{}'.format(
                job_config.gcs_prefix,
                job_config.program_id,
                job_config.job_id)

    def implied_job_config(self, job_config: Optional[JobConfig]) -> JobConfig:
        implied_job_config = (JobConfig()
                              if job_config is None
                              else job_config.copy())

        # Note: inference order is important. Later ones may need earlier ones.
        self._infer_project_id(implied_job_config)
        self._infer_gcs_prefix(implied_job_config)
        self._infer_program_id(implied_job_config)
        self._infer_job_id(implied_job_config)
        self._infer_gcs_program(implied_job_config)
        self._infer_gcs_results(implied_job_config)

        return implied_job_config

    def program_as_schedule(self, program: Program) -> Schedule:
        if isinstance(program, circuits.Circuit):
            device = program.device
            circuit_copy = program.copy()
            ConvertToXmonGates().optimize_circuit(circuit_copy)
            optimizers.DropEmptyMoments().optimize_circuit(circuit_copy)
            device.validate_circuit(circuit_copy)
            return moment_by_moment_schedule(device, circuit_copy)

        if isinstance(program, Schedule):
            return program

        raise TypeError('Unexpected program type.')

    def run_sweep(
            self,
            *,  # Force keyword args.
            program: Program,
            job_config: Optional[JobConfig] = None,
            params: Sweepable = None,
            repetitions: int = 1,
            priority: int = 500,
            processor_ids: Sequence[str] = ('xmonsim',),
            gate_set: SerializableGateSet = gate_sets.XMON) -> 'EngineJob':
        """Runs the supplied Circuit or Schedule via Quantum Engine.

        In contrast to run, this runs across multiple parameter sweeps, and
        does not block until a result is returned.

        Args:
            program: The Circuit or Schedule to execute. If a circuit is
                provided, a moment by moment schedule will be used.
            job_config: Configures the names of programs and jobs.
            params: Parameters to run with the program.
            repetitions: The number of circuit repetitions to run.
            priority: The priority to run at, 0-100.
            processor_ids: The engine processors to run against.
            gate_set: The gate set used to serialize the circuit. The gate set
                must be supported by the selected processor

        Returns:
            An EngineJob. If this is iterated over it returns a list of
            TrialResults, one for each parameter sweep.
        """

        job_config = self.implied_job_config(job_config)

        # Check program to run and program parameters.
        if not 0 <= priority < 1000:
            raise ValueError('priority must be between 0 and 1000')

        sweeps = _sweepable_to_sweeps(params or ParamResolver({}))
        if self.proto_version == ProtoVersion.V1:
            code, run_context = self._serialize_program_v1(
                program, sweeps, repetitions)
        elif self.proto_version == ProtoVersion.V2:
            code, run_context = self._serialize_program_v2(
                program, sweeps, repetitions, gate_set)
        else:
            raise ValueError('invalid proto version: {}'.format(
                self.proto_version))

        # Create program.
        request = {
            'name': 'projects/%s/programs/%s' % (job_config.project_id,
                                                 job_config.program_id,),
            'gcs_code_location': {'uri': job_config.gcs_program},
            'code': code,
        }
        response = self.service.projects().programs().create(
            parent='projects/%s' % job_config.project_id,
            body=request).execute()

        # Create job.
        request = {
            'name': '%s/jobs/%s' % (response['name'], job_config.job_id),
            'output_config': {
                'gcs_results_location': {
                    'uri': job_config.gcs_results
                }
            },
            'scheduling_config': {
                'priority': priority,
                'processor_selector': {
                    'processor_names': [
                        'projects/%s/processors/%s' %
                        (job_config.project_id, processor_id)
                        for processor_id in processor_ids
                    ]
                }
            },
            'run_context': run_context
        }
        response = self.service.projects().programs().jobs().create(
            parent=response['name'], body=request).execute()

        return EngineJob(job_config, response, self)

    def _serialize_program_v1(self, program: Program, sweeps: List[Sweep],
                              repetitions: int
                             ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        schedule = self.program_as_schedule(program)
        schedule.device.validate_schedule(schedule)

        program_descriptor = v1.program_pb2.Program.DESCRIPTOR
        program_dict = {}  # type: Dict[str, Any]
        program_dict['@type'] = TYPE_PREFIX + program_descriptor.full_name
        program_dict['operations'] = [
            op for op in schedule_to_proto_dicts(schedule)
        ]

        context_descriptor = v1.program_pb2.RunContext.DESCRIPTOR
        context_dict = {}  # type: Dict[str, Any]
        context_dict['@type'] = TYPE_PREFIX + context_descriptor.full_name
        context_dict['parameter_sweeps'] = [
            sweep_to_proto_dict(sweep, repetitions) for sweep in sweeps
        ]
        return program_dict, context_dict

    def _serialize_program_v2(self, program: Program, sweeps: List[Sweep],
                              repetitions: int, gate_set: SerializableGateSet
                             ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if isinstance(program, Schedule):
            program.device.validate_schedule(program)

        program = gate_set.serialize(program)

        run_context = v2.run_context_pb2.RunContext()
        for sweep in sweeps:
            sweep_proto = run_context.parameter_sweeps.add()
            sweep_proto.repetitions = repetitions
            api_v2.sweep_to_proto(sweep, out=sweep_proto.sweep)

        def any_dict(message: gp.message.Message) -> Dict[str, Any]:
            any_message = any_pb2.Any()
            any_message.Pack(message)
            return gp.json_format.MessageToDict(any_message)

        return any_dict(program), any_dict(run_context)

    def get_program(self, program_resource_name: str) -> Dict:
        """Returns the previously created quantum program.

        Params:
            program_resource_name: A string of the form
                `projects/project_id/programs/program_id`.

        Returns:
            A dictionary containing the metadata and the program.
        """
        return self.service.projects().programs().get(
            name=program_resource_name).execute()

    def get_job(self, job_resource_name: str) -> Dict:
        """Returns metadata about a previously created job.

        See get_job_result if you want the results of the job and not just
        metadata about the job.

        Params:
            job_resource_name: A string of the form
                `projects/project_id/programs/program_id/jobs/job_id`.

        Returns:
            A dictionary containing the metadata.
        """
        return self.service.projects().programs().jobs().get(
            name=job_resource_name).execute()

    def get_job_results(self, job_resource_name: str) -> List[TrialResult]:
        """Returns the actual results (not metadata) of a completed job.

        Params:
            job_resource_name: A string of the form
                `projects/project_id/programs/program_id/jobs/job_id`.

        Returns:
            An iterable over the TrialResult, one per parameter in the
            parameter sweep.
        """
        response = self.service.projects().programs().jobs().getResult(
            parent=job_resource_name).execute()
        result = response['result']
        result_type = result['@type'][len(TYPE_PREFIX):]

        if result_type == 'cirq.api.google.v1.Result':
            return self._get_job_results_v1(response['result'])
        if result_type == 'cirq.api.google.v2.Result':
            return self._get_job_results_v2(response['result'])
        raise ValueError('invalid result proto version: {}'.format(
            self.proto_version))

    def _get_job_results_v1(self, result: Dict[str, Any]) -> List[TrialResult]:
        trial_results = []
        for sweep_result in result['sweepResults']:
            sweep_repetitions = sweep_result['repetitions']
            key_sizes = [(m['key'], len(m['qubits']))
                         for m in sweep_result['measurementKeys']]
            for result in sweep_result['parameterizedResults']:
                data = base64.standard_b64decode(result['measurementResults'])
                measurements = unpack_results(data, sweep_repetitions,
                                              key_sizes)

                trial_results.append(
                    TrialResult.from_single_parameter_set(
                        params=ParamResolver(
                            result.get('params', {}).get('assignments', {})),
                        measurements=measurements))
        return trial_results

    def _get_job_results_v2(self,
                            result_dict: Dict[str, Any]) -> List[TrialResult]:
        result_any = any_pb2.Any()
        gp.json_format.ParseDict(result_dict, result_any)
        result = v2.result_pb2.Result()
        result_any.Unpack(result)

        sweep_results = api_v2.results_from_proto(result)
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
        self.service.projects().programs().jobs().cancel(
            name=job_resource_name, body={}).execute()

    def _set_program_labels(self, program_resource_name: str,
                            labels: Dict[str, str], fingerprint: str):
        self.service.projects().programs().patch(
            name=program_resource_name,
            body={'name': program_resource_name, 'labels': labels,
                  'labelFingerprint': fingerprint},
            updateMask='labels').execute()

    def set_program_labels(self, program_resource_name: str,
                           labels: Dict[str, str]):
        job = self.get_program(program_resource_name)
        self._set_program_labels(program_resource_name, labels,
                                 job.get('labelFingerprint', ''))

    def add_program_labels(self, program_resource_name: str,
                           labels: Dict[str, str]):
        job = self.get_program(program_resource_name)
        old_labels = job.get('labels', {})
        new_labels = old_labels.copy()
        new_labels.update(labels)
        if new_labels != old_labels:
            fingerprint = job.get('labelFingerprint', '')
            self._set_program_labels(program_resource_name, new_labels,
                                     fingerprint)

    def remove_program_labels(self, program_resource_name: str,
                              label_keys: List[str]):
        job = self.get_program(program_resource_name)
        old_labels = job.get('labels', {})
        new_labels = old_labels.copy()
        for key in label_keys:
            new_labels.pop(key, None)
        if new_labels != old_labels:
            fingerprint = job.get('labelFingerprint', '')
            self._set_program_labels(program_resource_name, new_labels,
                                     fingerprint)

    def _set_job_labels(self, job_resource_name: str, labels: Dict[str, str],
                        fingerprint: str):
        self.service.projects().programs().jobs().patch(
            name=job_resource_name,
            body={'name': job_resource_name, 'labels': labels,
                  'labelFingerprint': fingerprint},
            updateMask='labels').execute()

    def set_job_labels(self, job_resource_name: str, labels: Dict[str, str]):
        job = self.get_job(job_resource_name)
        self._set_job_labels(job_resource_name, labels,
                             job.get('labelFingerprint', ''))

    def add_job_labels(self, job_resource_name: str, labels: Dict[str, str]):
        job = self.get_job(job_resource_name)
        old_labels = job.get('labels', {})
        new_labels = old_labels.copy()
        new_labels.update(labels)
        if new_labels != old_labels:
            fingerprint = job.get('labelFingerprint', '')
            self._set_job_labels(job_resource_name, new_labels, fingerprint)

    def remove_job_labels(self, job_resource_name: str, label_keys: List[str]):
        job = self.get_job(job_resource_name)
        old_labels = job.get('labels', {})
        new_labels = old_labels.copy()
        for key in label_keys:
            new_labels.pop(key, None)
        if new_labels != old_labels:
            fingerprint = job.get('labelFingerprint', '')
            self._set_job_labels(job_resource_name, new_labels, fingerprint)


class EngineJob:
    """A job created via the Quantum Engine API.

    This job may be in a variety of states. It may be scheduling, it may be
    executing on a machine, or it may have entered a terminal state
    (either succeeding or failing).

    Attributes:
      job_config: The JobConfig used to create the job.
      job_resource_name: The full resource name of the engine job.
    """

    def __init__(self,
                 job_config: JobConfig,
                 job: Dict,
                 engine: Engine) -> None:
        """A job submitted to the engine.

        Args:
            job_config: The JobConfig used to create the job.
            job: A full Job Dict.
            engine: Engine connected to the job.
        """
        self.job_config = job_config
        self._job = job
        self._engine = engine
        self.job_resource_name = job['name']
        self.program_resource_name = self.job_resource_name.split('/jobs')[0]
        self._results = None  # type: Optional[List[TrialResult]]

    def _update_job(self):
        if self._job['executionStatus']['state'] not in TERMINAL_STATES:
            self._job = self._engine.get_job(self.job_resource_name)
        return self._job

    def status(self):
        """Return the execution status of the job."""
        return self._update_job()['executionStatus']['state']

    def cancel(self):
        """Cancel the job."""
        self._engine.cancel_job(self.job_resource_name)

    def results(self) -> List[TrialResult]:
        """Returns the job results, blocking until the job is complete.
        """
        if not self._results:
            job = self._update_job()
            for _ in range(1000):
                if job['executionStatus']['state'] in TERMINAL_STATES:
                    break
                time.sleep(0.5)
                job = self._update_job()
            if job['executionStatus']['state'] != 'SUCCESS':
                raise RuntimeError(
                    'Job %s did not succeed. It is in state %s.' % (
                        job['name'], job['executionStatus']['state']))
            self._results = self._engine.get_job_results(
                self.job_resource_name)
        return self._results

    def __iter__(self):
        return self.results().__iter__()


def _sweepable_to_sweeps(sweepable: Sweepable) -> List[Sweep]:
    if isinstance(sweepable, ParamResolver):
        return [_resolver_to_sweep(sweepable)]
    if isinstance(sweepable, Sweep):
        return [sweepable]
    if isinstance(sweepable, Iterable):
        iterable = cast(Iterable, sweepable)
        if isinstance(next(iter(iterable)), Sweep):
            sweeps = iterable
            return list(sweeps)

        resolvers = iterable
        return [_resolver_to_sweep(p) for p in resolvers]

    raise TypeError('Unexpected Sweepable.')  # coverage: ignore


def _resolver_to_sweep(resolver: ParamResolver) -> Sweep:
    params = resolver.param_dict
    if not params:
        return UnitSweep
    return Zip(*[Points(key, [value]) for key, value in params.items()])
