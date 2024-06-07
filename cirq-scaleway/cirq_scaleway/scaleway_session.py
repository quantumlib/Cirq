# Copyright 2024 Scaleway
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
import time
import json
import cirq
import httpx
import randomname

from cirq.study import ResultDict
from typing import Union, Optional, Dict, List
from pytimeparse.timeparse import timeparse

from .scaleway_client import QaaSClient
from .scaleway_models import (
    JobPayload,
    ClientPayload,
    BackendPayload,
    RunPayload,
    SerializationType,
    CircuitPayload,
)
from .versions import USER_AGENT


class ScalewaySession(cirq.work.Sampler):
    def __init__(
        self,
        device,
        client: QaaSClient,
        name: Optional[str],
        deduplication_id: Optional[str],
        max_duration: Union[int, str],
        max_idle_duration: Union[int, str],
    ):
        self.__id = None
        self.__device = device
        self.__client = client
        self.__name = name
        self.__deduplication_id = deduplication_id

        if isinstance(max_duration, str):
            max_duration = f"{timeparse(max_duration)}s"

        if isinstance(max_idle_duration, str):
            max_idle_duration = f"{timeparse(max_idle_duration)}s"

        self.__max_duration = max_duration
        self.__max_idle_duration = max_idle_duration

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.stop()
        return False

    @property
    def status(self) -> str:
        """Returns the current status of the device session.

        Returns:
            str: the current status of the session. Can be either: starting, runnng, stopping, stopped
        """
        if not self.__id:
            raise Exception("session not started")

        dict = self.__client.get_session(session_id=self.__id)

        return dict.get("status", "unknown_status")

    @property
    def id(self) -> str:
        """The unique identifier of the device session.

        Returns:
            str: The UUID of the current session.
        """
        return self.__id

    @property
    def name(self) -> str:
        """Name of the device session.

        Returns:
            str: the session's name.
        """
        return self.__name

    def start(self) -> "ScalewaySession":
        """Starts a new device session to run job against it.

        Args:
            name (str): name of the session. Used only for convenient purpose.
            deduplication_id (str): an identifier you can use to clearly identify a session.
                The deduplication_id allows to retrieve a same session and to share it among people
            max_duration (str, int): the maximum duration before the session is automatically killed.
                Can be either a string like 1h, 20m15s or an int representing seconds
            max_idle_duration (str, int): the maximum duration without job before the session is automatically killed.
                Can be either a string like 1h, 20m15s or an int representing seconds
        Returns:
            ScalewaySession: a new freshly starting QPU session
        """
        if self.__id:
            raise Exception("session already started")

        self.__id = self.__client.create_session(
            self.__name,
            platform_id=self.__device.id,
            deduplication_id=self.__deduplication_id,
            max_duration=self.__max_duration,
            max_idle_duration=self.__max_idle_duration,
        )

        return self

    def stop(self) -> "ScalewaySession":
        """Stops to the running device session.
        All attached jobs and their results will are kept up to 7 days before total deletion.
        """
        if not self.__id:
            raise Exception("session not started")

        self.__client.terminate_session(session_id=self.__id)

        return self

    def delete(self) -> None:
        """Immediately stop and delete to the running device session.
        All attached jobs and their results will be purged from Scaleway service.
        """
        if not self.__id:
            raise Exception("session not started")

        self.__client.delete_session(session_id=self.__id)

    def run_sweep(
        self, program: cirq.AbstractCircuit, params: cirq.study.Sweepable, repetitions: int = 1
    ) -> List[cirq.study.Result]:
        """Samples from the given Circuit.

        This allows for sweeping over different parameter values,
        unlike the `run` method.  The `params` argument will provide a
        mapping from `sympy.Symbol`s used within the circuit to a set of
        values.  Unlike the `run` method, which specifies a single
        mapping from symbol to value, this method allows a "sweep" of
        values.  This allows a user to specify execution of a family of
        related circuits efficiently.

        Args:
            program: The circuit to sample from.
            params: Parameters to run with the program.
            repetitions: The number of times to sample.

        Returns:
            Result list for this run; one for each possible parameter resolver.
        """
        trial_results = []

        if not self.__id:
            raise Exception("session not started")

        for param_resolver in cirq.study.to_resolvers(params):
            circuit = cirq.protocols.resolve_parameters(program, param_resolver)
            serialized_circuit = cirq.to_json(circuit)

            run_opts = RunPayload(
                options={"shots": repetitions},
                circuits=[
                    CircuitPayload(
                        serialization_type=SerializationType.JSON,
                        circuit_serialization=serialized_circuit,
                    )
                ],
            )

            results = self._submit(run_opts, self.__id)
            trial_results.append(results)

        return trial_results

    def _extract_payload_from_response(self, result_response: Dict) -> str:
        result = result_response.get("result", None)

        if result is None or result == "":
            url = result_response.get("url", None)

            if url is not None:
                resp = httpx.get(url)
                resp.raise_for_status()

                return resp.text
            else:
                raise Exception("Got result with both empty data and url fields")
        else:
            return result

    def _wait_for_result(
        self, job_id: str, timeout: Optional[int] = None, fetch_interval: int = 2
    ) -> Dict | None:
        start_time = time.time()

        while True:
            time.sleep(fetch_interval)

            elapsed = time.time() - start_time

            if timeout is not None and elapsed >= timeout:
                raise Exception("Timed out waiting for result")

            job = self.__client.get_job(job_id)

            if job["status"] == "completed":
                return self.__client.get_job_results(job_id)

            if job["status"] in ["error", "unknown_status"]:
                raise Exception("Job error")

    def _to_cirq_result(self, job_results: List) -> cirq.Result:
        if len(job_results) == 0:
            raise Exception("Empty result list")

        payload = self._extract_payload_from_response(job_results[0])
        payload_dict = json.loads(payload)
        cirq_result = ResultDict._from_json_dict_(**payload_dict)

        return cirq_result

    def _submit(self, run_opts: RunPayload, session_id: str) -> cirq.study.Result:
        backend_opts = BackendPayload(
            name=self.__device.name, version=self.__device.version, options={}
        )

        client_opts = ClientPayload(user_agent=USER_AGENT)

        job_payload = JobPayload.schema().dumps(
            JobPayload(backend=backend_opts, run=run_opts, client=client_opts)
        )

        job_id = self.__client.create_job(
            name=randomname.get_name(), session_id=session_id, circuits=job_payload
        )

        job_results = self._wait_for_result(job_id, 60 * 10, 2)
        result = self._to_cirq_result(job_results)

        return result
