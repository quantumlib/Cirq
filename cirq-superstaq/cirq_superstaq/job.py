# Copyright 2021 The Cirq Developers
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
"""Represents a job created via the SuperstaQ API."""

import collections
import time

from cirq._doc import document

import cirq_superstaq
from cirq_superstaq import superstaq_client


class Job:
    """A job created on the SuperstaQ API.

    Note that this is mutable, when calls to get status or results are made
    the job updates itself to the results returned from the API.

    If a job is canceled or deleted, only the job id and the status remain
    valid.
    """

    TERMINAL_STATES = ("Done", "Canceled", "Failed", "Deleted")
    document(
        TERMINAL_STATES,
        "States of the SuperstaQ API job from which the job cannot transition. "
        "Note that deleted can only exist in a return call from a delete "
        "(subsequent calls will return not found).",
    )

    NON_TERMINAL_STATES = ("Ready", "Submitted", "Running")
    document(
        NON_TERMINAL_STATES, "States of the SuperstaQ API job which can transition to other states."
    )

    ALL_STATES = TERMINAL_STATES + NON_TERMINAL_STATES
    document(ALL_STATES, "All states that an SuperstaQ API job can exist in.")

    UNSUCCESSFUL_STATES = ("Canceled", "Failed", "Deleted")
    document(
        UNSUCCESSFUL_STATES,
        "States of the SuperstaQ API job when it was not successful and so does not have any "
        "data associated with it beyond an id and a status.",
    )

    def __init__(self, client: superstaq_client._SuperstaQClient, job_dict: dict):
        """Construct a Job.

        Users should not call this themselves. If you only know the `job_id`, use `get_job`
        on `cirq_superstaq.Service`.

        Args:
            client: The client used for calling the API.
            job_dict: A dict representing the response from a call to get_job on the client.
        """
        self._client = client
        self._job = job_dict

    def _refresh_job(self) -> None:
        """If the last fetched job is not terminal, gets the job from the API."""
        if self._job["status"] not in self.TERMINAL_STATES:
            self._job = self._client.get_job(self.job_id())

    def _check_if_unsuccessful(self) -> None:
        if self.status() in self.UNSUCCESSFUL_STATES:
            raise cirq_superstaq.superstaq_exceptions.SuperstaQUnsuccessfulJobException(
                self.job_id(), self.status()
            )

    def job_id(self) -> str:
        """Returns the job id (UID) for the job.

        This is the id used for identifying the job by the API.
        """
        return self._job["job_id"]

    def status(self) -> str:
        """Gets the current status of the job.

        If the current job is in a non-terminal state,
        this will update the job and return the current status.
        A full list of states is given in
        `cirq_superstaq.Job.ALL_STATES`.

        Raises:
            SuperstaQException: If the API is not able to get the status of the job.

        Returns:
            The job status.
        """
        self._refresh_job()
        return self._job["status"]

    def target(self) -> str:
        """Returns the target where the job is to be run, or was run.

        Returns:
            'qpu' or 'simulator' depending on where the job was run or is running.

        Raises:
            SuperstaQUnsuccessfulJob: If the job has failed, been canceled, or deleted.
            SuperstaQException: If unable to get the status of the job from the API.
        """
        self._check_if_unsuccessful()
        return self._job["target"]

    def num_qubits(self) -> int:
        """Returns the number of qubits for the job.

        Raises:
            SuperstaQUnsuccessfulJob: If the job has failed, been canceled, or deleted.
            SuperstaQException: If unable to get the status of the job from the API.
        """
        self._check_if_unsuccessful()
        return self._job["num_qubits"]

    def repetitions(self) -> int:
        """Returns the number of repetitions for the job.

        Raises:
            SuperstaQUnsuccessfulJob: If the job has failed, been canceled, or deleted.
            SuperstaQException: If unable to get the status of the job from the API.
        """
        self._check_if_unsuccessful()
        return int(self._job["shots"][0]["shots"])

    def counts(
        self, timeout_seconds: int = 7200, polling_seconds: float = 1.0
    ) -> collections.Counter:
        """Polls the SuperstaQ API for results.

        Args:
            timeout_seconds: The total number of seconds to poll for.
            polling_seconds: The interval with which to poll.

        Returns:
            collections.Counter that represents the results of the measurements

        Raises:
            SuperstaQUnsuccessfulJob: If the job has failed, been canceled, or deleted.
            SuperstaQException: If unable to get the results from the API.
        """
        time_waited_seconds: float = 0.0
        while time_waited_seconds < timeout_seconds:
            # Status does a refresh.
            if self.status() in self.TERMINAL_STATES:
                break
            time.sleep(polling_seconds)
            time_waited_seconds += polling_seconds
        if self.status() != "Done":
            if "failure" in self._job and "error" in self._job["failure"]:
                error = self._job["failure"]["error"]
                raise RuntimeError(f"Job failed. Error message: {error}")
            raise RuntimeError(
                f"Job was not completed successful. Instead had status: {self.status()}"
            )
        return self._job["samples"]

    def __str__(self) -> str:
        return f"Job with job_id={self.job_id()}"
