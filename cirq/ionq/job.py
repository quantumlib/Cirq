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
from typing import Optional, Union, TYPE_CHECKING

from cirq.ionq import results
from cirq.value import digits
from cirq._doc import document

if TYPE_CHECKING:
    import cirq


def _little_endian_to_big(value: int, bit_count: int) -> int:
    return digits.big_endian_bits_to_int(
        digits.big_endian_int_to_bits(value, bit_count=bit_count)[::-1])


class Job:
    """A job created on the IonQ API.

    Note that this is mutable, when calls to get status or results are made
    the job updates itself to the results returned from the API.
    """

    TERMINAL_STATES = ('completed', 'canceled', 'failed')
    document(
        TERMINAL_STATES,
        'States of the IonQ API job from which the job cannot transition.')

    NON_TERMINAL_STATES = ('ready', 'submitted', 'running')
    document(
        NON_TERMINAL_STATES,
        'States of the IonQ API job which can transition to other states.')

    ALL_STATES = TERMINAL_STATES + NON_TERMINAL_STATES
    document(ALL_STATES, 'All states that an IonQ API job can exist in.')

    def __init__(self, client: 'cirq.ionq.ionq_client._IonQClient',
                 job_dict: dict):
        """Construct an IonQJob.

        Users should not call this themselves. If you only know the `job_id`,
        use `get_job` on `cirq.ionq.Service`.

        Args:
            client: The client used for calling the API.
            job: A dict representing the response from a call to get_job on the
                client.
        """
        self._client = client
        self._job = job_dict

    def _refresh_job(self):
        """If the last fetched job is not terminal, gets the job from the API.
        """
        if self._job['status'] not in self.TERMINAL_STATES:
            self._job = self._client.get_job(self.job_id())

    def job_id(self) -> str:
        """Returns the job id (UID) for the job.

        This is the id used for identifying the job by the API.
        """
        return self._job['id']

    def status(self) -> str:
        """Gets the current status of the job.

        This will get a new job if the status of the job previously was
        determined to not be in a terminal state. A full list of states is
        given in  `cirq.ionq.IonQJob.ALL_STATES`.
        """
        self._refresh_job()
        return self._job['status']

    def target(self) -> str:
        """Returns the target where the job is to be run, or was run.

        Returns:
            'qpu' or 'simulator' depending on where the job was run or is
            running.
        """
        return self._job['target']

    def name(self) -> str:
        """Returns the name of the job which was supplied during job creation.

        This is different than the `job_id`.
        """
        return self._job['name']

    def num_qubits(self) -> int:
        """Returns the number of qubits for the job."""
        return int(self._job['qubits'])

    def repetitions(self) -> Optional[int]:
        """Returns the number of repetitions for the job.

        If run on the simulator this will return None.
        """
        if 'metadata' in self._job and 'shots' in self._job['metadata']:
            return int(self._job['metadata']['shots'])
        return None

    def results(self, timeout_seconds: int = 7200, polling_seconds: int = 1
               ) -> Union[results.QPUResult, results.SimulatorResult]:
        """Polls the IonQ api for results.

        Args:
            timeout_seconds: The total number of seconds to poll for.
            polling_seconds: The interval with which to poll.

        Returns:
            Either a `cirq.ionq.QPUResults` or `cirq.ionq.SimulatorResults`
            depending on whether the job was running on an actual quantum
            processor or a simulator.

        Raises:
            RuntimeError: if the job was not successfully completed (cancelled
                or failed).
        """
        time_waited_seconds = 0
        while time_waited_seconds < timeout_seconds:
            # Status does a refresh.
            if self.status() in self.TERMINAL_STATES:
                break
            time.sleep(polling_seconds)
            time_waited_seconds += polling_seconds
        if self.status() != 'completed':
            raise RuntimeError('Job was not completed successful. Instead had'
                               f' status: {self.status()}')
        # IonQ returns results in little endian, Cirq prefers to use big endian,
        # so we convert.
        if self.target() == 'qpu':
            repetitions = self.repetitions()
            assert repetitions is not None
            counts = {
                _little_endian_to_big(int(k), self.num_qubits()):
                int(repetitions * float(v))
                for k, v in self._job['data']['histogram'].items()
            }
            return results.QPUResult(counts=counts,
                                     num_qubits=self.num_qubits())
        else:
            probabilities = {
                _little_endian_to_big(int(k), self.num_qubits()): float(v)
                for k, v in self._job['data']['histogram'].items()
            }
            return results.SimulatorResult(probabilities=probabilities,
                                           num_qubits=self.num_qubits())

    def __str__(self) -> str:
        return f'cirq.ionq.Job(job_id={self.job_id()})'
