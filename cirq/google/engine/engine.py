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
    program = engine.create_program(ciruit)
    result0 = program.run(params=params0, repetitions=10)
    result1 = program.run(params=params1, repetitions=10)

In order to run on must have access to the Quantum Engine API. Access to this
API is (as of June 22, 2018) restricted to invitation only.
"""

import base64
import enum
import random
import re
import string
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import warnings

from apiclient import discovery, http as apiclient_http
import google.protobuf as gp
from google.protobuf import any_pb2

from cirq import circuits, optimizers, study, value
from cirq.google import gate_sets
from cirq.api.google import v1, v2
from cirq.google.api import v1 as api_v1
from cirq.google.api import v2 as api_v2
from cirq.google.convert_to_xmon_gates import ConvertToXmonGates
from cirq.google.serializable_gate_set import SerializableGateSet
from cirq.schedules import Schedule, moment_by_moment_schedule
from cirq.study import ParamResolver, Sweep, Sweepable, TrialResult

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


def _any_dict_from_msg(message: gp.message.Message) -> Dict[str, Any]:
    any_message = any_pb2.Any()
    any_message.Pack(message)
    return gp.json_format.MessageToDict(any_message)


def _user_project_header_request_builder(project_id: str):
    """Provides a request builder that sets a user project header on engine
    requests to allow using standard OAuth credentials.
    """

    def request_builder(*args, **kwargs):
        request = apiclient_http.HttpRequest(*args, **kwargs)
        request.headers['X-Goog-User-Project'] = project_id
        return request

    return request_builder


def _make_random_id(prefix: str, length: int = 6):
    random_digits = [
        random.choice(string.ascii_uppercase + string.digits)
        for _ in range(length)
    ]
    suffix = ''.join(random_digits)
    return '%s%s' % (prefix, suffix)


@value.value_equality
class JobConfig:
    """Configuration for a job to run on the Quantum Engine API.

    An instance of a program that has been scheduled on the Quantum Engine is
    called a Job. This object contains the configuration for a job.
    """

    def __init__(self,
                 job_id: Optional[str] = None,
                 gcs_prefix: Optional[str] = None,
                 gcs_results: Optional[str] = None) -> None:
        """Configuration for a job that is run on Quantum Engine.

        Args:
            job_id: Id of the job to create, defaults to 'job-0'.
            gcs_prefix: Google Cloud Storage bucket and object prefix to use
                for storing programs and results. The bucket will be created if
                needed. Must be in the form "gs://bucket-name/object-prefix/".
            gcs_results: Explicit override for the results storage location.
        """
        self.job_id = job_id
        self.gcs_prefix = gcs_prefix
        self.gcs_results = gcs_results

    def copy(self):
        return JobConfig(job_id=self.job_id,
                         gcs_prefix=self.gcs_prefix,
                         gcs_results=self.gcs_results)

    def _value_equality_values_(self):
        return (self.job_id, self.gcs_prefix, self.gcs_results)

    def __repr__(self):
        return ('cirq.google.JobConfig(job_id={!r}, '
                'gcs_prefix={!r}, '
                'gcs_results={!r})').format(self.job_id, self.gcs_prefix,
                                            self.gcs_results)


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
                 version: Optional[str] = None,
                 discovery_url: Optional[str] = None,
                 default_gcs_prefix: Optional[str] = None,
                 proto_version: ProtoVersion = ProtoVersion.V1,
                 service_args: Optional[Dict] = None) -> None:
        """Engine service client.

        Args:
            project_id: A project_id string of the Google Cloud Project to use.
                API interactions will be attributed to this project and any
                resources created will be owned by the project. See
                https://cloud.google.com/resource-manager/docs/creating-managing-projects#identifying_projects
            version: API version.
            discovery_url: Discovery url for the API to select a non-default
                backend for the Engine. Incompatible with `version` argument.
            default_gcs_prefix: A fallback gcs_prefix to use when one isn't
                specified in the JobConfig given to 'run' methods.
                See JobConfig for more information on gcs_prefix.
            service_args: A dictionary of arguments that can be used to
                configure options on the underlying apiclient. See
                https://github.com/googleapis/google-api-python-client
        """
        if discovery_url and version:
            raise ValueError("`version` and `discovery_url` are both "
                             "specified, but are incompatible. Please use "
                             "only one of these arguments")
        if not (discovery_url or version):
            version = 'v1alpha1'

        self.project_id = project_id
        self.discovery_url = discovery_url or discovery.V2_DISCOVERY_URI
        self.default_gcs_prefix = default_gcs_prefix
        self.proto_version = proto_version

        if not service_args:
            service_args = {}
        if not 'requestBuilder' in service_args:
            request_builder = _user_project_header_request_builder(project_id)
            service_args['requestBuilder'] = request_builder

        # Suppress warnings about using Application Default Credentials.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.service = discovery.build(
                '' if discovery_url else 'quantum',
                '' if discovery_url else version,
                discoveryServiceUrl=self.discovery_url,
                **service_args)

    def run(
            self,
            *,  # Force keyword args.
            program: Program,
            program_id: Optional[str] = None,
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
            program_id: A user-provided identifier for the program. This must
                be unique within the Google Cloud project being used. If this
                parameter is not provided, a random id of the format
                'prog-######' will be generated.
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
        return list(
            self.run_sweep(program=program,
                           program_id=program_id,
                           job_config=job_config,
                           params=[param_resolver],
                           repetitions=repetitions,
                           priority=priority,
                           processor_ids=processor_ids,
                           gate_set=gate_set))[0]

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
            program_id: Optional[str] = None,
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
            program_id: A user-provided identifier for the program. This must
                be unique within the Google Cloud project being used. If this
                parameter is not provided, a random id of the format
                'prog-######' will be generated.
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
            params: Sweepable = None,
            repetitions: int = 1,
            priority: int = 500,
            processor_ids: Sequence[str] = ('xmonsim',),
            gate_set: SerializableGateSet = gate_sets.XMON) -> 'EngineJob':

        # Check program to run and program parameters.
        if not 0 <= priority < 1000:
            raise ValueError('priority must be between 0 and 1000')

        job_config = self.implied_job_config(job_config)
        sweeps = study.to_sweeps(params or ParamResolver({}))
        run_context = self._serialize_run_context(sweeps, repetitions)

        # Create job.
        request = {
            'name': '%s/jobs/%s' % (program_name, job_config.job_id),
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
                        (self.project_id, processor_id)
                        for processor_id in processor_ids
                    ]
                }
            },
            'run_context': run_context
        }
        response = self.service.projects().programs().jobs().create(
            parent=program_name, body=request).execute()

        return EngineJob(job_config, response, self)

    def implied_job_config(self, job_config: Optional[JobConfig]) -> JobConfig:
        implied_job_config = (JobConfig()
                              if job_config is None else job_config.copy())

        # Note: inference order is important. Later ones may need earlier ones.
        self._infer_gcs_prefix(implied_job_config)
        self._infer_job_id(implied_job_config)
        self._infer_gcs_results(implied_job_config)

        return implied_job_config

    def _infer_gcs_prefix(self, job_config: JobConfig) -> None:
        project_id = self.project_id
        gcs_prefix = (job_config.gcs_prefix or self.default_gcs_prefix or
                      'gs://gqe-' + project_id[project_id.rfind(':') + 1:])
        if gcs_prefix and not gcs_prefix.endswith('/'):
            gcs_prefix += '/'

        if not gcs_prefix_pattern.match(gcs_prefix):
            raise ValueError('gcs_prefix must be of the form "gs://'
                             '<bucket name and optional object prefix>/"')

        job_config.gcs_prefix = gcs_prefix

    def _infer_job_id(self, job_config: JobConfig) -> None:
        if job_config.job_id is None:
            job_config.job_id = _make_random_id('job-')

    def _infer_gcs_results(self, job_config: JobConfig) -> None:
        if job_config.gcs_prefix is None:
            raise ValueError("Must infer gcs_prefix before gcs_results.")

        if job_config.gcs_results is None:
            job_config.gcs_results = '{}jobs/{}'.format(job_config.gcs_prefix,
                                                        job_config.job_id)

    def _serialize_run_context(
            self,
            sweeps: List[Sweep],
            repetitions: int,
    ) -> Dict[str, Any]:
        proto_version = self.proto_version
        if proto_version == ProtoVersion.V1:
            context_descriptor = v1.program_pb2.RunContext.DESCRIPTOR
            context_dict = {}  # type: Dict[str, Any]
            context_dict['@type'] = TYPE_PREFIX + context_descriptor.full_name
            context_dict['parameter_sweeps'] = [
                api_v1.sweep_to_proto_dict(sweep, repetitions)
                for sweep in sweeps
            ]
            return context_dict
        elif proto_version == ProtoVersion.V2:
            run_context = v2.run_context_pb2.RunContext()
            for sweep in sweeps:
                sweep_proto = run_context.parameter_sweeps.add()
                sweep_proto.repetitions = repetitions
                api_v2.sweep_to_proto(sweep, out=sweep_proto.sweep)

            return _any_dict_from_msg(run_context)
        else:
            raise ValueError(
                'invalid run context proto version: {}'.format(proto_version))

    def create_program(self,
                       program: Program,
                       program_id: Optional[str] = None,
                       gate_set: SerializableGateSet = gate_sets.XMON
                      ) -> 'EngineProgram':
        """Wraps a Circuit or Scheduler for use with the Quantum Engine.

        Args:
            program: The Circuit or Schedule to execute. If a circuit is
                provided, a moment by moment schedule will be used.
            program_id: A user-provided identifier for the program. This must be
                unique within the Google Cloud project being used. If this
                parameter is not provided, a random id of the format
                'prog-######' will be generated.
            gate_set: The gate set used to serialize the circuit. The gate set
                must be supported by the selected processor
        """
        if not program_id:
            program_id = _make_random_id('prog-')

        parent_name = 'projects/%s' % self.project_id
        program_name = '%s/programs/%s' % (parent_name, program_id)
        # Create program.
        request = {
            'name': program_name,
            'code': self._serialize_program(program, gate_set),
        }
        result = self.service.projects().programs().create(
            parent=parent_name, body=request).execute()

        return EngineProgram(result['name'], self)

    def _serialize_program(self,
                           program: Program,
                           gate_set: SerializableGateSet = gate_sets.XMON
                          ) -> Dict[str, Any]:
        if self.proto_version == ProtoVersion.V1:
            schedule = self.program_as_schedule(program)
            schedule.device.validate_schedule(schedule)

            program_descriptor = v1.program_pb2.Program.DESCRIPTOR
            program_dict = {}  # type: Dict[str, Any]
            program_dict['@type'] = TYPE_PREFIX + program_descriptor.full_name
            program_dict['operations'] = [
                op for op in api_v1.schedule_to_proto_dicts(schedule)
            ]
            return program_dict
        elif self.proto_version == ProtoVersion.V2:
            if isinstance(program, Schedule):
                program.device.validate_schedule(program)
            program = gate_set.serialize(program)
            return _any_dict_from_msg(program)
        else:
            raise ValueError('invalid program proto version: {}'.format(
                self.proto_version))

    def get_program(self, program_id: str) -> Dict:
        """Returns the previously created quantum program.

        Params:
            program_id: A string containing the unique ID of a program within
              the project specified for the Engine.

        Returns:
            A dictionary containing the metadata and the program.
        """
        program_resource_name = self._program_name_from_id(program_id)
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
                measurements = api_v1.unpack_results(data, sweep_repetitions,
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

    def _program_name_from_id(self, program_id: str) -> str:
        return 'projects/%s/programs/%s' % (
            self.project_id,
            program_id,
        )

    def _set_program_labels(self, program_id: str, labels: Dict[str, str],
                            fingerprint: str):
        program_resource_name = self._program_name_from_id(program_id)
        self.service.projects().programs().patch(
            name=program_resource_name,
            body={'name': program_resource_name, 'labels': labels,
                  'labelFingerprint': fingerprint},
            updateMask='labels').execute()

    def set_program_labels(self, program_id: str, labels: Dict[str, str]):
        program = self.get_program(program_id)
        self._set_program_labels(program_id, labels,
                                 program.get('labelFingerprint', ''))

    def add_program_labels(self, program_id: str, labels: Dict[str, str]):
        program = self.get_program(program_id)
        old_labels = program.get('labels', {})
        new_labels = old_labels.copy()
        new_labels.update(labels)
        if new_labels != old_labels:
            fingerprint = program.get('labelFingerprint', '')
            self._set_program_labels(program_id, new_labels, fingerprint)

    def remove_program_labels(self, program_id: str, label_keys: List[str]):
        program = self.get_program(program_id)
        old_labels = program.get('labels', {})
        new_labels = old_labels.copy()
        for key in label_keys:
            new_labels.pop(key, None)
        if new_labels != old_labels:
            fingerprint = program.get('labelFingerprint', '')
            self._set_program_labels(program_id, new_labels, fingerprint)

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


    def list_processors(self) -> List[Dict]:
        """Returns a list of Processors that the user has visibility to in the
        current Engine project. The names of these processors are used to
        identify devices when scheduling jobs and gathering calibration metrics.

        Returns:
            A list of dictionaries containing the metadata of each processor.
        """
        parent = 'projects/%s' % (self.project_id)
        response = self.service.projects().processors().list(
            parent=parent).execute()
        return response['processors']


    def get_latest_calibration(self,
                               processor_id: str) -> Optional['Calibration']:
        """ Returns metadata about the latest known calibration run for a
        processor, or None if there is no calibration available.

        Params:
            processor_id: The processor identifier within the resource name,
                where name has the format:
                `projects/<project_id>/processors/<processor_id>`.

        Returns:
            A dictionary containing the calibration data.
        """
        processor_name = 'projects/{}/processors/{}'.format(
            self.project_id, processor_id)
        response = self.service.projects().processors().calibrations().list(
            parent=processor_name).execute()
        if (not 'calibrations' in response or
                len(response['calibrations']) < 1):
            return None
        return Calibration(response['calibrations'][0]['data'])

    def get_calibration(self, calibration_name: str) -> 'Calibration':
        """Retrieve metadata about a specific calibration run.

        Params:
            calibration_name: A string of the form
                `<processor name>/calibrations/<ms since epoch>`

        Returns:
            A dictionary containing the metadata.
        """
        response = self.service.projects().processors().calibrations().get(
            name=calibration_name).execute()
        return Calibration(response['data']['data'])


class Calibration:
    """A convenience wrapper for calibrations

    Attributes:
        timestamp: The time that this calibration was run, in milliseconds since
            the epoch.
    """

    def __init__(self, calibration: Dict) -> None:
        self.timestamp = int(calibration['timestampMs'])
        self._metrics = calibration['metrics']

    def get_metric_names(self) -> List[str]:
        """ Returns a list of known metrics in this calibration. """
        return list(set(m['name'] for m in self._metrics))

    def get_metrics_by_name(self, name: str) -> List[Dict]:
        """ Get a filtered list of metrics matching the provided name.

        Values are grouped into a flat list, grouped by target.

        Params:
            name: the name of a metric referred to in the calibration. Valid
            names can be found with the get_metric_names() method.

        Returns:
            A list of dictionaries containing pairs of targets and values for
            the requested metric.
        """
        result = []
        matching_metrics = [m for m in self._metrics if m['name'] == name]
        for metric in matching_metrics:
            flat_values: List = []
            # Flatten the values a list, removing keys containing type names
            # (e.g. proto version of each value is {<type>: value}).
            for value in metric['values']:
                flat_values += [value[type] for type in value]
            result.append({'targets': metric['targets'], 'values': flat_values})
        return result


class EngineProgram:
    """A program created via the Quantum Engine API.

    This program wraps a Circuit or Schedule with additional metadata used to
    schedule againt devices maintained by Google.

    Attributes:
      name: The full resource name of the engine program.
      code: A serialized version of the Circuit or Schedule
    """

    def __init__(self, resource_name: str, engine: Engine) -> None:
        """A job submitted to the engine.

        Args:
            resource_name: The globally unique identifier for the program:
                `projects/project_id/programs/program_id`.
            engine: An Engine object associated with the same project as the
                program.
        """
        self.resource_name = resource_name
        self._engine = engine

    def run_sweep(
            self,
            *,  # Force keyword args.
            job_config: Optional[JobConfig] = None,
            params: Sweepable = None,
            repetitions: int = 1,
            priority: int = 500,
            processor_ids: Sequence[str] = ('xmonsim',)) -> 'EngineJob':
        """Runs the program on the QuantumEngine.

        In contrast to run, this runs across multiple parameter sweeps, and
        does not block until a result is returned.

        Args:
            job_config: Configures optional job parameters.
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
        return self._engine.create_job(program_name=self.resource_name,
                                       job_config=job_config,
                                       params=params,
                                       repetitions=repetitions,
                                       priority=priority,
                                       processor_ids=processor_ids)

    def run(
            self,
            *,  # Force keyword args.
            job_config: Optional[JobConfig] = None,
            param_resolver: ParamResolver = ParamResolver({}),
            repetitions: int = 1,
            priority: int = 50,
            processor_ids: Sequence[str] = ('xmonsim',)) -> TrialResult:
        """Runs the supplied Circuit or Schedule via Quantum Engine.

        Args:
            program: A Quantum Engine-wrapped Circuit or Schedule object. This
              may be generated with create_program() or get_program().
            program_id: A user-provided identifier for the program. This must be
                unique within the Google Cloud project being used. If this
                parameter is not provided, a random id of the format
                'prog-######' will be generated.
            program_id: A user-defined identifer for the program. This must be
              unique within the project specified on the Engine instance.
            job_config: Configures the names of programs and jobs.
            param_resolver: Parameters to run with the program.
            repetitions: The number of repetitions to simulate.
            priority: The priority to run at, 0-100.
            processor_ids: The engine processors that should be candidates
                to run the program. Only one of these will be scheduled for
                execution.

        Returns:
            A single TrialResult for this run.
        """
        return list(
            self.run_sweep(job_config=job_config,
                           params=[param_resolver],
                           repetitions=repetitions,
                           priority=priority,
                           processor_ids=processor_ids))[0]


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
            engine: An Engine instance associated with the same projct as the
                job.
        """
        self.job_config = job_config
        self._job = job
        self._engine = engine
        self.job_resource_name = job['name']
        self.program_id = self.job_resource_name.split('/jobs')[0]
        self._results = None  # type: Optional[List[TrialResult]]

    def _update_job(self):
        if self._job['executionStatus']['state'] not in TERMINAL_STATES:
            self._job = self._engine.get_job(self.job_resource_name)
        return self._job

    def status(self):
        """Return the execution status of the job."""
        return self._update_job()['executionStatus']['state']

    def get_calibration(self) -> Optional[Calibration]:
        """Returns the recorded calibration at the time when the job was run, if
        one was captured, else None."""
        status = self._job['executionStatus']
        if (not 'calibrationName' in status): return None
        return self._engine.get_calibration(status['calibrationName'])

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
