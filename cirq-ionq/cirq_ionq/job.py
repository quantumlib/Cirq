# Copyright 2020 The Cirq Developers
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
"""Represents a job created via the IonQ API."""

import time
from typing import Dict, Sequence, Union, TYPE_CHECKING

from cirq_ionq import ionq_exceptions, results
from cirq._doc import document

import cirq

if TYPE_CHECKING:
    import cirq_ionq


def _little_endian_to_big(value: int, bit_count: int) -> int:
    return cirq.big_endian_bits_to_int(
        cirq.big_endian_int_to_bits(value, bit_count=bit_count)[::-1]
    )


class Job:
    """A job created on the IonQ API.

    Note that this is mutable, when calls to get status or results are made
    the job updates itself to the results returned from the API.

    If a job is canceled or deleted, only the job id and the status remain
    valid.
    """

    TERMINAL_STATES = ('completed', 'canceled', 'failed', 'deleted')
    document(
        TERMINAL_STATES,
        'States of the IonQ API job from which the job cannot transition. '
        'Note that deleted can only exist in a return call from a delete '
        '(subsequent calls will return not found).',
    )

    NON_TERMINAL_STATES = ('ready', 'submitted', 'running')
    document(
        NON_TERMINAL_STATES, 'States of the IonQ API job which can transition to other states.'
    )

    ALL_STATES = TERMINAL_STATES + NON_TERMINAL_STATES
    document(ALL_STATES, 'All states that an IonQ API job can exist in.')

    UNSUCCESSFUL_STATES = ('canceled', 'failed', 'deleted')
    document(
        UNSUCCESSFUL_STATES,
        'States of the IonQ API job when it was not successful and so does not have any '
        'data associated with it beyond an id and a status.',
    )

    def __init__(self, client: 'cirq_ionq.ionq_client._IonQClient', job_dict: dict):
        """Construct an IonQJob.

        Users should not call this themselves. If you only know the `job_id`, use `get_job`
        on `cirq_ionq.Service`.

        Args:
            client: The client used for calling the API.
            job_dict: A dict representing the response from a call to get_job on the client.
        """
        self._client = client
        self._job = job_dict

    def _refresh_job(self):
        """If the last fetched job is not terminal, gets the job from the API."""
        if self._job['status'] not in self.TERMINAL_STATES:
            self._job = self._client.get_job(self.job_id())

    def _check_if_unsuccessful(self):
        if self.status() in self.UNSUCCESSFUL_STATES:
            raise ionq_exceptions.IonQUnsuccessfulJobException(self.job_id(), self.status())

    def job_id(self) -> str:
        """Returns the job id (UID) for the job.

        This is the id used for identifying the job by the API.
        """
        return self._job['id']

    def status(self) -> str:
        """Gets the current status of the job.

        This will get a new job if the status of the job previously was determined to not be in
        a terminal state. A full list of states is given in `cirq_ionq.IonQJob.ALL_STATES`.

        Raises:
            IonQException: If the API is not able to get the status of the job.

        Returns:
            The job status.
        """
        self._refresh_job()
        return self._job['status']

    def target(self) -> str:
        """Returns the target where the job is to be run, or was run.

        Returns:
            'qpu' or 'simulator' depending on where the job was run or is running.

        Raises:
            IonQUnsuccessfulJob: If the job has failed, been canceled, or deleted.
            IonQException: If unable to get the status of the job from the API.
        """
        self._check_if_unsuccessful()
        return self._job['target']

    def name(self) -> str:
        """Returns the name of the job which was supplied during job creation.

        This is different than the `job_id`.

        Raises:
            IonQUnsuccessfulJob: If the job has failed, been canceled, or deleted.
            IonQException: If unable to get the status of the job from the API.
        """
        self._check_if_unsuccessful()
        return self._job['name']

    def num_qubits(self) -> int:
        """Returns the number of qubits for the job.

        Raises:
            IonQUnsuccessfulJob: If the job has failed, been canceled, or deleted.
            IonQException: If unable to get the status of the job from the API.
        """
        self._check_if_unsuccessful()
        return int(self._job['qubits'])

    def repetitions(self) -> int:
        """Returns the number of repetitions for the job.

        Raises:
            IonQUnsuccessfulJob: If the job has failed, been canceled, or deleted.
            IonQException: If unable to get the status of the job from the API.
        """
        self._check_if_unsuccessful()
        return int(self._job['metadata']['shots'])

    def measurement_dict(self) -> Dict[str, Sequence[int]]:
        """Returns a dictionary of measurement keys to target qubit index."""
        measurement_dict: Dict[str, Sequence[int]] = {}
        if 'metadata' in self._job:
            full_str = ''.join(
                value
                for key, value in self._job['metadata'].items()
                if key.startswith('measurement')
            )
            if full_str == '':
                return measurement_dict
            for key_value in full_str.split(chr(30)):
                key, value = key_value.split(chr(31))
                measurement_dict[key] = [int(t) for t in value.split(',')]
        return measurement_dict

    # TODO(#3388) Add documentation for Raises.
    # pylint: disable=missing-raises-doc
    def results(
        self, timeout_seconds: int = 7200, polling_seconds: int = 1
    ) -> Union[results.QPUResult, results.SimulatorResult]:
        """Polls the IonQ api for results.

        Args:
            timeout_seconds: The total number of seconds to poll for.
            polling_seconds: The interval with which to poll.

        Returns:
            Either a `cirq_ionq.QPUResults` or `cirq_ionq.SimulatorResults` depending on whether
            the job was running on an actual quantum processor or a simulator.

        Raises:
            IonQUnsuccessfulJob: If the job has failed, been canceled, or deleted.
            IonQException: If unable to get the results from the API.
        """
        time_waited_seconds = 0
        while time_waited_seconds < timeout_seconds:
            # Status does a refresh.
            if self.status() in self.TERMINAL_STATES:
                break
            time.sleep(polling_seconds)
            time_waited_seconds += polling_seconds
        if self.status() != 'completed':
            if 'failure' in self._job and 'error' in self._job['failure']:
                error = self._job['failure']['error']
                raise RuntimeError(f'Job failed. Error message: {error}')
            if time_waited_seconds >= timeout_seconds:
                raise TimeoutError(f'Job timed out after waiting {time_waited_seconds} seconds.')
            raise RuntimeError(
                f'Job was not completed successful. Instead had status: {self.status()}'
            )
        # IonQ returns results in little endian, Cirq prefers to use big endian, so we convert.
        if self.target() == 'qpu':
            repetitions = self.repetitions()
            counts = {
                _little_endian_to_big(int(k), self.num_qubits()): int(repetitions * float(v))
                for k, v in self._job['data']['histogram'].items()
            }
            return results.QPUResult(
                counts=counts,
                num_qubits=self.num_qubits(),
                measurement_dict=self.measurement_dict(),
            )
        else:
            probabilities = {
                _little_endian_to_big(int(k), self.num_qubits()): float(v)
                for k, v in self._job['data']['histogram'].items()
            }
            return results.SimulatorResult(
                probabilities=probabilities,
                num_qubits=self.num_qubits(),
                measurement_dict=self.measurement_dict(),
                repetitions=self.repetitions(),
            )

    # pylint: enable=missing-raises-doc
    def cancel(self):
        """Cancel the given job.

        This mutates the job to only have a job id and status `canceled`.
        """
        self._job = self._client.cancel_job(job_id=self.job_id())

    def delete(self):
        """Delete the given job.

        This mutates the job to only have a job id and status `deleted`. Subsequence attempts to
        get the job with this job id will return not found.
        """
        self._job = self._client.delete_job(job_id=self.job_id())

    def __str__(self) -> str:
        return f'cirq_ionq.Job(job_id={self.job_id()})'
