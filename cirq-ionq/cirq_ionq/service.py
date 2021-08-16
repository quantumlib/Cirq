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

import datetime
import os
from typing import cast, Optional, Sequence

import cirq
from cirq_ionq import calibration, ionq_client, job, results, sampler, serializer


class Service:
    """A class to access IonQ's API.

    To access the API, this class requires a remote host url and an API key. These can be
    specified in the constructor via the parameters `remote_host` and `api_key`. Alternatively
    these can be specified by setting the environment variables `IONQ_REMOTE_HOST` and
    `IONQ_API_KEY`.
    """

    # TODO(#3388) Add documentation for Raises.
    # pylint: disable=missing-raises-doc
    def __init__(
        self,
        remote_host: Optional[str] = None,
        api_key: Optional[str] = None,
        default_target: str = None,
        api_version='v0.1',
        max_retry_seconds: int = 3600,
        verbose=False,
    ):
        """Creates the Service to access IonQ's API.

        Args:
            remote_host: The location of the api in the form of an url. If this is None,
                then this instance will use the environment variable `IONQ_REMOTE_HOST`. If that
                variable is not set, then this uses `https://api.ionq.co/{api_version}`, where
                `{api_version}` is the `api_version` specified below.
            api_key: A string key which allows access to the api. If this is None,
                then this instance will use the environment variable  `IONQ_API_KEY`. If that
                variable is not set, then this will raise an `EnvironmentError`.
            default_target: Which target to default to using. If set to None, no default is set
                and target must always be specified in calls. If set, then this default is used,
                unless a target is specified for a given call. Supports either 'qpu' or
                'simulator'.
            api_version: Version of the api. Defaults to 'v0.1'.
            max_retry_seconds: The number of seconds to retry calls for. Defaults to one hour.
            verbose: Whether to print to stdio and stderr on retriable errors.

        Raises:
            EnvironmentError: if the `api_key` is None and has no corresponding environment
                variable set.
        """
        self.remote_host = (
            remote_host or os.getenv('IONQ_REMOTE_HOST') or f'https://api.ionq.co/{api_version}'
        )
        self.api_key = api_key or os.getenv('IONQ_API_KEY')
        if not self.api_key:
            raise EnvironmentError(
                f'Parameter api_key was not specified and the environment variable '
                f'IONQ_API_KEY was also not set.'
            )

        self._client = ionq_client._IonQClient(
            remote_host=self.remote_host,
            api_key=self.api_key,
            default_target=default_target,
            api_version=api_version,
            max_retry_seconds=max_retry_seconds,
            verbose=verbose,
        )

    # pylint: enable=missing-raises-doc
    def run(
        self,
        circuit: cirq.Circuit,
        repetitions: int,
        name: Optional[str] = None,
        target: Optional[str] = None,
        param_resolver: cirq.ParamResolverOrSimilarType = cirq.ParamResolver({}),
        seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
    ) -> cirq.Result:
        """Run the given circuit on the IonQ API.

        Args:
            circuit: The circuit to run.
            repetitions: The number of times to run the circuit.
            name: An optional name for the created job. Different from the `job_id`.
            target: Where to run the job. Can be 'qpu' or 'simulator'.
            param_resolver: A `cirq.ParamResolver` to resolve parameters in  `circuit`.
            seed: If the target is `simulation` the seed for generating results. If None, this
                will be `np.random`, if an int, will be `np.random.RandomState(int)`, otherwise
                must be a modulate similar to `np.random`.

        Returns:
            A `cirq.Result` for running the circuit.
        """
        resolved_circuit = cirq.resolve_parameters(circuit, param_resolver)
        result = self.create_job(resolved_circuit, repetitions, name, target).results()
        if isinstance(result, results.QPUResult):
            return result.to_cirq_result(params=cirq.ParamResolver(param_resolver))
        else:
            sim_result = cast(results.SimulatorResult, result)
            return sim_result.to_cirq_result(params=cirq.ParamResolver(param_resolver), seed=seed)

    def sampler(self, target: Optional[str] = None, seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None):
        """Returns a `cirq.Sampler` object for accessing the sampler interface.

        Args:
            target: The target to sample against. Either this or `default_target` on this
                service must be specified. If this is None, uses the `default_target`. If
                both `default_target` and `target` are specified, uses `target`.
            seed: If the target is `simulation` the seed for generating results. If None, this
                will be `np.random`, if an int, will be `np.random.RandomState(int)`, otherwise
                must be a modulate similar to `np.random`.
        Returns:
            A `cirq.Sampler` for the IonQ API.
        """
        return sampler.Sampler(service=self, target=target, seed=seed)

    def create_job(
        self,
        circuit: cirq.AbstractCircuit,
        repetitions: int = 100,
        name: Optional[str] = None,
        target: Optional[str] = None,
    ) -> job.Job:
        """Create a new job to run the given circuit.

        Args:
            circuit: The circuit to run.
            repetitions: The number of times to repeat the circuit. Defaults to 100.
            name: An optional name for the created job. Different from the `job_id`.
            target: Where to run the job. Can be 'qpu' or 'simulator'.

        Returns:
            A `cirq_ionq.IonQJob` which can be queried for status or results.

        Raises:
            IonQException: If there was an error accessing the API.
        """
        serialized_program = serializer.Serializer().serialize(circuit)
        result = self._client.create_job(
            serialized_program=serialized_program, repetitions=repetitions, target=target, name=name
        )
        # The returned job does not have fully populated fields, so make
        # a second call and return the results of the fully filled out job.
        return self.get_job(result['id'])

    def get_job(self, job_id: str) -> job.Job:
        """Gets a job that has been created on the IonQ API.

        Args:
            job_id: The UUID of the job. Jobs are assigned these numbers by the server during the
            creation of the job.

        Returns:
            A `cirq_ionq.IonQJob` which can be queried for status or results.

        Raises:
            IonQNotFoundException: If there was no job with the given `job_id`.
            IonQException: If there was an error accessing the API.
        """
        job_dict = self._client.get_job(job_id=job_id)
        return job.Job(client=self._client, job_dict=job_dict)

    def list_jobs(
        self, status: Optional[str] = None, limit: int = 100, batch_size: int = 1000
    ) -> Sequence[job.Job]:
        """Lists jobs that have been created on the IonQ API.

        Args:
            status: If supplied will filter to only jobs with this status.
            limit: The maximum number of jobs to return.
            batch_size: The size of the batches requested per http GET call.

        Returns:
            A sequence of jobs.

        Raises:
            IonQException: If there was an error accessing the API.
        """
        job_dicts = self._client.list_jobs(status=status, limit=limit, batch_size=batch_size)
        return tuple(job.Job(client=self._client, job_dict=job_dict) for job_dict in job_dicts)

    def get_current_calibration(self) -> calibration.Calibration:
        """Gets the most recent calbration via the API.

        Note that currently there is only one target, so this returns the calibration of that
        target.

        The calibration include device specification (number of qubits, connectivity), as well
        as fidelities and timings of gates.

        Returns:
            A `cirq_ionq.Calibration` containing the device specification and calibrations.

        Raises:
            IonQException: If there was an error accessing the API.
        """
        calibration_dict = self._client.get_current_calibration()
        return calibration.Calibration(calibration_dict=calibration_dict)

    def list_calibrations(
        self,
        start: datetime.datetime = None,
        end: datetime.datetime = None,
        limit: int = 100,
        batch_size: int = 1000,
    ) -> Sequence[calibration.Calibration]:
        """List calibrations via the API.

        Args:
            start: If supplied, only calibrations after this date and time. Accurate to seconds.
            end: If supplied, only calibrations before this date and time. Accurate to seconds.
            limit: The maximum number of calibrations to return.
            batch_size: The size of the batches requested per http GET call.

        Returns:
            A sequence of calibrations.

        Raises:
            IonQException: If there was an error accessing the API.
        """
        calibration_dicts = self._client.list_calibrations(
            start=start, end=end, limit=limit, batch_size=batch_size
        )
        return [calibration.Calibration(calibration_dict=c) for c in calibration_dicts]
