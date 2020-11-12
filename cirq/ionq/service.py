# Copyright 2020 The Cirq Developers
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
"""Service to access IonQs API."""

from typing import Optional, TYPE_CHECKING

from cirq.ionq import ionq_client, job, serializer

if TYPE_CHECKING:
    import cirq


class Service:
    """A class to access IonQ's API."""

    def __init__(self,
                 remote_host: str,
                 api_key: str,
                 default_target: str = None,
                 api_version='v0.1',
                 max_retry_seconds: int = 3600,
                 verbose=False):
        """Creates the Service to access IonQ's API.

        Args:
            remote_host: The location of the api in the form of an url.
            api_key: A string key which allows access to the api.
            default_target: Which target to default to using. If set to None, no
                default is set and target must always be specified in calls.
                If set, then this default is used, unless a target is specified
                for a given call. Supports either 'qpu' or 'simulator'.
            api_version: Version of the api. Defaults to 'v0.1'.
            max_retry_seconds: The number of seconds to retry calls for.
                Defaults to one hour.
            verbose: Whether to print to stdio and stderr on retriable errors.
        """
        self._client = ionq_client._IonQClient(
            remote_host=remote_host,
            api_key=api_key,
            default_target=default_target,
            api_version=api_version,
            max_retry_seconds=max_retry_seconds,
            verbose=verbose)

    def create_job(self,
                   circuit: 'cirq.Circuit',
                   repetitions: int = None,
                   name: Optional[str] = None,
                   target: Optional[str] = None) -> job.Job:
        """Create a new job to run the given circuit.

        Args:
            circuit: The circuit to run.
            repetitions: The number of times to repeat the circuit. Should
                only be set if the target is `qpu`.
            name: An optional name for the created job. Different from the
                `job_id`.
            target: Where to run the job. Can be 'qpu' or 'simulator'.

        Returns:
            A `cirq.ionq.IonQJob` which can be queried for status or results.

        Raises:
            IonQException: If there was an error accessing the API.
        """
        serialized_circuit = serializer.Serializer().serialize(circuit)
        result = self._client.create_job(circuit_dict=serialized_circuit,
                                         repetitions=repetitions,
                                         target=target,
                                         name=name)
        # The returned job does not have fully populated fields, so make
        # a second call and return the results of the fully filled out job.
        return self.get_job(result['id'])

    def get_job(self, job_id: str) -> job.Job:
        """Gets a job that has been created on via the API.

        Args:
            job_id: The UUID of the job. Jobs are assigned these numbers by
                the server during the creation of the job.

        Returns:
            A `cirq.ionq.IonQJob` which can be queried for status or results.

        Raises:
            IonQNotFoundException: If there was no job with the given `job_id`.
            IonQException: If there was an error accessing the API.
        """
        job_dict = self._client.get_job(job_id=job_id)
        return job.Job(client=self._client, job_dict=job_dict)
