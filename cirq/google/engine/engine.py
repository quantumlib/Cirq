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

"""Client executor for the Google's Quantum Engine.
"""

import base64
import random
import re
import string
import time
import urllib.parse
from collections import Iterable
from typing import Dict, List, Optional, Union, cast

import numpy as np
from apiclient import discovery
from google.protobuf.json_format import MessageToDict

from cirq.api.google.v1 import program_pb2
from cirq.circuits import Circuit
from cirq.circuits.drop_empty_moments import DropEmptyMoments
from cirq.devices import Device
from cirq.google.convert_to_xmon_gates import ConvertToXmonGates
from cirq.google.params import sweep_to_proto
from cirq.google.programs import schedule_to_proto, unpack_results
from cirq.schedules import Schedule, moment_by_moment_schedule
from cirq.study import ParamResolver, Sweep, Sweepable, TrialResult
from cirq.study.sweeps import Points, Unit, Zip

gcs_prefix_pattern = re.compile('gs://[a-z0-9._/-]+')
TERMINAL_STATES = ['SUCCESS', 'FAILURE', 'CANCELLED']


class EngineTrialResult(TrialResult):
    """Results of a single run of an executor.

    Attributes:
        measurements: A dictionary from measurement gate key to measurement
            results ordered by the qubits acted upon by the measurement gate.
    """

    def __init__(self,
                 params: ParamResolver,
                 repetitions: int,
                 measurements: Dict[str, np.ndarray]) -> None:
        self.params = params
        self.repetitions = repetitions
        self.measurements = measurements

    def __str__(self):
        def bitstring(vals):
            return ''.join('1' if v else '0' for v in vals)

        keyed_bitstrings = [
            (key, bitstring(val)) for key, val in self.measurements.items()
        ]
        return ' '.join('{}={}'.format(key, val)
                        for key, val in sorted(keyed_bitstrings))


class EngineOptions:
    """Options for the Engine.
    """

    def __init__(self, project_id: str,
                 program_id: Optional[str] = None,
                 job_id: Optional[str] = None,
                 gcs_prefix: Optional[str] = None,
                 gcs_program: Optional[str] = None,
                 gcs_results: Optional[str] = None) -> None:
        """Engine options for running the Engine. At a minimum requires
        project_id and either gcs_prefix or gcs_program and gcs_results.

        Args:
            project_id: The project id string of the Google Cloud Project to
                use.
            program_id: Id of the program to create, defaults to a random
                version of 'prog-ABCD'.
            job_id: Id of the job to create, defaults to 'job-0'.
            gcs_prefix: Google Cloud Storage bucket and object prefix to use for
                storing programs and results. The bucket will be created if
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


class Engine:
    """Executor for Google Quantum Engine.
    """

    def __init__(self, api_key: str, api: str = 'quantum',
                 version: str = 'v1alpha1',
                 discovery_url: Optional[str] = None,
                 gcs_prefix: Optional[str] = None,
                 **kwargs
    ) -> None:
        """Engine service client.

        Args:
            api_key: API key to use to retrieve discovery doc.
            api: API name.
            version: API version.
            discovery_url: Discovery url to use.
            credentials: Credentials to use.
        """
        self.api_key = api_key
        self.api = api
        self.version = version
        self.discovery_url = discovery_url or ('https://{api}.googleapis.com/'
                                               '$discovery/rest'
                                               '?version={apiVersion}&key=%s')
        self.gcs_prefix = gcs_prefix
        self.service = discovery.build(
            self.api,
            self.version,
            discoveryServiceUrl=self.discovery_url % urllib.parse.quote_plus(
                self.api_key),
            **kwargs)

    def run(self,
            options: EngineOptions,
            circuit: Circuit,
            device: Device,
            param_resolver: ParamResolver = ParamResolver({}),
            repetitions: int = 1,
            priority: int = 50,
            target_route: str = '/xmonsim',
    ) -> EngineTrialResult:
        """Simulates the entire supplied Circuit.

        Args:
            circuit: The circuit to simulate.
            device: The device on which to run the circuit.
            param_resolver: Parameters to run with the program.
            repetitions: The number of repetitions to simulate.
            options: Options configuring the simulation.
            priority: The priority to run at, 0-100.
            target_route: The engine route to run against.

        Returns:
            Results for this run.
        """
        return list(self.run_sweep(options, circuit, device, [param_resolver],
                                   repetitions, priority, target_route))[0]

    def run_sweep(self,
                  options: EngineOptions,
                  program: Union[Circuit, Schedule],
                  device: Device = None,
                  params: Sweepable = None,
                  repetitions: int = 1,
                  priority: int = 500,
                  target_route: str = '/xmonsim',
    ) -> 'EngineJob':
        """Runs the entire supplied Circuit or Schedule via Google Quantum
         Engine.

        Args:
            program: The circuit or schedule to execute.
            device: The device on which to run a circuit. Required only if
                program is a Circuit.
            params: Parameters to run with the program.
            repetitions: The number of circuit repetitions to run.
            options: Options configuring the engine.
            priority: The priority to run at, 0-100.
            target_route: The engine route to run against.

        Returns:
            Results for this run.
        """
        # Check and compute engine options.
        gcs_prefix = options.gcs_prefix or self.gcs_prefix or (
                'gs://gqe-%s/' % options.project_id[
                                 options.project_id.rfind(
                                     ':') + 1:])
        gcs_prefix = gcs_prefix if not gcs_prefix or gcs_prefix.endswith(
            '/') else gcs_prefix + '/'
        if gcs_prefix and not gcs_prefix_pattern.match(gcs_prefix):
            raise TypeError('gcs_prefix must be of the form "gs://'
                            '<bucket name and optional object prefix>/"')
        if not gcs_prefix and (not options.gcs_program or
                               not options.gcs_results):
            raise TypeError('Either gcs_prefix must be provided or both'
                            ' gcs_program and gcs_results are required.')

        project_id = options.project_id
        program_id = options.program_id or 'prog-%s' % ''.join(
            random.choice(string.ascii_uppercase + string.digits) for _ in
            range(6))
        job_id = options.job_id or 'job-0'
        gcs_program = options.gcs_program or '%sprograms/%s/%s' % (
            gcs_prefix, program_id, program_id)
        gcs_results = options.gcs_results or '%sprograms/%s/jobs/%s' % (
            gcs_prefix, program_id, job_id)

        # Check program to run and program parameters.
        if not 0 <= priority < 1000:
            raise TypeError('priority must be between 0 and 1000')

        if isinstance(program, Circuit):
            if not device:
                raise TypeError('device is required when running a circuit')
            # Convert to a schedule.
            circuit_copy = Circuit(program.moments)
            ConvertToXmonGates().optimize_circuit(circuit_copy)
            DropEmptyMoments().optimize_circuit(circuit_copy)

            schedule = moment_by_moment_schedule(device, circuit_copy)
        elif isinstance(program, Schedule):
            if device:
                raise TypeError(
                    'device can not be provided when running a schedule')
            schedule = program
        else:
            raise TypeError('Unexpected execution type')

        # Create program.
        sweeps = _sweepable_to_sweeps(params or ParamResolver({}))
        proto_program = program_pb2.Program()
        for sweep in sweeps:
            sweep_proto = proto_program.parameter_sweeps.add()
            sweep_to_proto(sweep, sweep_proto)
            sweep_proto.repetitions = repetitions
        program_dict = MessageToDict(proto_program)
        program_dict['operations'] = [MessageToDict(op) for op in
                                      schedule_to_proto(schedule)]
        code = {
            '@type': 'type.googleapis.com/cirq.api.google.v1.Program'}
        code.update(program_dict)
        request = {
            'name': 'projects/%s/programs/%s' % (project_id,
                                                 program_id,),
            'gcs_code_location': {'uri': gcs_program, },
            'code': code,
        }
        response = self.service.projects().programs().create(
            parent='projects/%s' % project_id, body=request).execute()

        # Create job.
        request = {
            'name': '%s/jobs/%s' % (response['name'], job_id),
            'output_config': {
                'gcs_results_location': {
                    'uri': gcs_results
                }
            },
            'scheduling_config': {
                'priority': priority,
                'target_route': target_route
            },
        }
        response = self.service.projects().programs().jobs().create(
            parent=response['name'], body=request).execute()

        return EngineJob(
            EngineOptions(project_id, program_id, job_id, gcs_prefix,
                          gcs_program, gcs_results), response, self)

    def get_program(self, program_resource_name) -> Dict:
        return self.service.projects().programs().get(
            name=program_resource_name).execute()

    def get_job(self, job_resource_name) -> Dict:
        return self.service.projects().programs().jobs().get(
            name=job_resource_name).execute()

    def get_job_results(self, job_resource_name) -> List[EngineTrialResult]:
        response = self.service.projects().programs().jobs().getResult(
            parent=job_resource_name).execute()
        trial_results = []
        for sweep_result in response['result']['sweepResults']:
            sweep_repetitions = sweep_result['repetitions']
            key_sizes = [(m['key'], len(m['qubits']))
                         for m in sweep_result['measurementKeys']]
            for result in sweep_result['parameterizedResults']:
                data = base64.standard_b64decode(result['measurementResults'])
                measurements = unpack_results(data, sweep_repetitions,
                                              key_sizes)

                trial_results.append(EngineTrialResult(
                    params=ParamResolver(
                        result.get('params', {}).get('assignments', {})),
                    repetitions=sweep_repetitions,
                    measurements=measurements))
        return trial_results

    def cancel_job(self, job_resource_name: str):
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
    """A job running on the engine.

    Attributes:
      engine_options: The engine options used for the job.
      job_resource_name: The full resource name of the engine job.
    """

    def __init__(self,
        engine_options: EngineOptions,
        job: Dict,
        engine: Engine) -> None:
        """A job submitted to the engine.

        Args:
            engine_options: The EngineOptions used to create the job.
            job: A full Job Dict.
            engine: Engine connected to the job.
        """
        self.engine_options = engine_options
        self._job = job
        self._engine = engine
        self.job_resource_name = job['name']
        self.program_resource_name = self.job_resource_name.split('/jobs')[0]
        self._results = None  # type: Optional[List[EngineTrialResult]]

    def _update_job(self):
        if self._job['executionStatus']['state'] not in TERMINAL_STATES:
            self._job = self._engine.get_job(self.job_resource_name)
        return self._job

    def state(self):
        return self._update_job()['executionStatus']['state']

    def cancel(self):
        self._engine.cancel_job(self.job_resource_name)

    def results(self) -> List[EngineTrialResult]:
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
    elif isinstance(sweepable, Sweep):
        return [sweepable]
    elif isinstance(sweepable, Iterable):
        iterable = cast(Iterable, sweepable)
        if isinstance(next(iter(iterable)), Sweep):
            sweeps = iterable
            return list(sweeps)
        else:
            resolvers = iterable
            return [_resolver_to_sweep(p) for p in resolvers]
    else:
        raise TypeError('Unexpected Sweepable')


def _resolver_to_sweep(resolver: ParamResolver) -> Sweep:
    return Zip(*[Points(key, [value]) for key, value in
                 resolver.param_dict.items()]) if len(
        resolver.param_dict) else Unit
