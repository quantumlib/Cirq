# Copyright 2018 Google LLC
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
from typing import Dict, Optional, Union
from typing import List  # pylint: disable=unused-import

import numpy as np
import oauth2client
from apiclient.discovery import build
from google.protobuf.json_format import MessageToDict

from cirq.circuits import Circuit, ExpandComposite
from cirq.circuits.drop_empty_moments import DropEmptyMoments
from cirq.devices import Device
from cirq.google.convert_to_xmon_gates import ConvertToXmonGates
from cirq.google.programs import schedule_to_proto, unpack_results
from cirq.schedules import Schedule, moment_by_moment_schedule
from cirq.study import Executor, TrialResult
from cirq.study.resolver import ParamResolver
from cirq.api.google.v1 import program_pb2

gcs_prefix_pattern = re.compile('gs://[a-z0-9._/-]+')


class EngineOptions:
    """Options for the Engine.
    """

    def __init__(self, project_id: str,
                 credentials: Optional[oauth2client.client.Credentials] = None,
                 program_id: Optional[str] = None,
                 job_id: Optional[str] = None,
                 gcs_prefix: Optional[str] = None,
                 gcs_program: Optional[str] = None,
                 gcs_results: Optional[str] = None) -> None:
        """Engine options for running the Engine. At a minimum requires
        project_id and either gcs_prefix or gcs_program and gcs_results.

        Args:
            credentials: Credentials to use.
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
        self.credentials = credentials
        self.program_id = program_id or 'prog-%s' % ''.join(
            random.choice(string.ascii_uppercase + string.digits) for _ in
            range(6))
        self.job_id = job_id or 'job-0'
        if not gcs_prefix and (not gcs_program or not gcs_results):
            raise TypeError('Either gcs_prefix must be provided or both'
                            ' gcs_program and gcs_results are required.')
        if gcs_prefix and not gcs_prefix_pattern.match(gcs_prefix):
            raise TypeError('gcs_prefix must be of the form "gs://'
                            '<bucket name and optional object prefix>/"')
        self.gcs_prefix = gcs_prefix if not gcs_prefix or gcs_prefix.endswith(
            '/') else gcs_prefix + '/'
        self.gcs_program = gcs_program or '%scirq/%s-program' % (
            self.gcs_prefix, self.program_id)
        self.gcs_results = gcs_results or '%scirq/%s-%s-results' % (
            self.gcs_prefix, self.program_id, self.job_id)


class Engine(Executor):
    """Executor for Google Quantum Engine
    """

    def __init__(self, api_key: str, api: str = 'quantum',
                 version: str = 'v1alpha1',
                 discovery_url: Optional[str] = None) -> None:
        self.api_key = api_key
        self.api = api
        self.version = version
        self.discovery_url = discovery_url or ('https://{api}.googleapis.com/'
                                               '$discovery/rest'
                                               '?version={apiVersion}&key=%s')

    def run(
        self,
        program: Union[Circuit, Schedule],
        param_resolver: ParamResolver = None,
        repetitions: int = 1,
        options: EngineOptions = None,
        device: Device = None,
        priority: int = 50,
    ) -> 'EngineTrialResult':
        """Runs the entire supplied Circuit or Schedule via Google Quantum
         Engine.

        Args:
            program: The circuit to execute.
            param_resolver: A ParamResolver for determining values of
                ParameterizedValues.
            repetitions: The number of circuit repetitions to run.
            options: Options configuring the engine.
            device: The device on which to run.
            priority: The priority to run at, 0-100.

        Returns:
            Results for this run.
        """
        if not 0 <= priority < 100:
            raise TypeError('priority must be between 0 and 100')

        if isinstance(program, Schedule):
            schedule = program
        else:
            # Convert to a schedule.
            circuit_copy = Circuit(program.moments)
            ConvertToXmonGates().optimize_circuit(circuit_copy)
            DropEmptyMoments().optimize_circuit(circuit_copy)

            schedule = moment_by_moment_schedule(device, circuit_copy)

        service = build(self.api, self.version,
                        discoveryServiceUrl=self.discovery_url % (
                            self.api_key,),
                        credentials=options.credentials)

        proto_program = program_pb2.Program()
        proto_program.parameter_sweeps.add().repetitions = repetitions
        proto_program.operations.extend(list(schedule_to_proto(schedule)))

        code = {
            '@type': 'type.googleapis.com/cirq.api.google.v1.Program'}
        code.update(MessageToDict(proto_program))

        request = {
            'name': 'projects/%s/programs/%s' % (options.project_id,
                                                 options.program_id,),
            'gcs_code_location': {'uri': options.gcs_program, },
            'code': code,
        }

        response = service.projects().programs().create(
            parent='projects/%s' % options.project_id, body=request).execute()

        request = {
            'name': '%s/jobs/%s' % (response['name'], options.job_id),
            'output_config': {
                'gcs_results_location': {
                    'uri': options.gcs_results
                }
            },
            'scheduling_config': {
                'priority': priority,
                # TODO get route from device
                'target_route': '/xmonsim'
            },
        }
        response = service.projects().programs().jobs().create(
            parent=response['name'], body=request).execute()

        for _ in range(1000):
            if response['executionStatus']['state'] in ['SUCCESS', 'FAILURE',
                                                        'CANCELLED']:
                break
            time.sleep(0.5)
            response = service.projects().programs().jobs().get(
                name=response['name']).execute()

        if response['executionStatus']['state'] != 'SUCCESS':
            raise RuntimeError('Job %s did not succeed. It is in state %s.' % (
                response['name'], response['executionStatus']['state']))

        response = service.projects().programs().jobs().getResult(
            parent=response['name']).execute()

        # Only a single sweep is supported for now
        sweep_results = response['result']['sweepResults'][0]
        repetitions = sweep_results['repetitions']
        key_sizes=[(m['key'], m['size'])
                   for m in sweep_results['measurementKeys']]
        data = base64.standard_b64decode(
            sweep_results['parameterizedResults'][0]['measurementResults'])

        measurements = unpack_results(data, repetitions, key_sizes)

        return EngineTrialResult(
            params=param_resolver,
            repetitions=repetitions,
            measurements=measurements)


class EngineTrialResult(TrialResult):
    """Results of a single run of an executor.

    Attributes:
        measurements: A dictionary from measurement gate key to measurement
            results. If a key is reused, the measurement values are returned
            in the order they appear in the Circuit being executed.
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
