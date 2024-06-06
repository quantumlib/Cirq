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
from typing import List, Union, Sequence, Optional

from .scaleway_client import QaaSClient
from .scaleway_models import (
    JobPayload,
    ClientPayload,
    BackendPayload,
    RunPayload,
    SerializationType,
    CircuitPayload,
)
from .scaleway_session import ScalewaySession
from .scaleway_device import ScalewayDevice
from .versions import USER_AGENT


class ScalewaySampler(cirq.work.Sampler):
    def __init__(self, client: QaaSClient, device: ScalewayDevice, **kwarg) -> None:
        self.__client = client
        self.__device = device

    def _extract_payload_from_response(self, result_response: dict) -> str:
        result = result_response.get("result", None)

        if result is None or result == "":
            url = result_response.get("url", None)

            if url is not None:
                resp = httpx.get(url)
                resp.raise_for_status()

                return resp.text
            else:
                raise Exception("Got result with empty data and url fields")
        else:
            return result

    def _wait_for_result(self, timeout=None, fetch_interval: int = 2) -> dict | None:
        start_time = time.time()

        while True:
            time.sleep(fetch_interval)

            elapsed = time.time() - start_time

            if timeout is not None and elapsed >= timeout:
                raise Exception("Timed out waiting for result")

            job = self.__client.get_job(self._job_id)

            if job["status"] == "completed":
                return self.__client.get_job_results(self._job_id)

            if job["status"] in ["error", "unknown_status"]:
                raise Exception("Job error")

    def _to_cirq_result(self, job_results) -> cirq.Result:
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

        self._job_id = self.__client.create_job(
            name=randomname.get_name(), session_id=session_id, circuits=job_payload
        )

        job_results = self._wait_for_result(None, 2)
        result = self._to_cirq_result(job_results)

        return result

    def run_sweep(
        self,
        program: cirq.AbstractCircuit,
        params: cirq.study.Sweepable,
        repetitions: int = 1,
        session: Union[str, ScalewaySession] = None,
    ) -> List[cirq.study.Result]:
        trial_results = []

        if not session:
            session = self.__device.start_session()

        for param_resolver in cirq.study.to_resolvers(params):
            circuit = cirq.protocols.resolve_parameters(program, param_resolver)
            serialized_circuit = cirq.to_json(circuit)

            run_opts = RunPayload(
                options={"shots": repetitions},
                circuit=CircuitPayload(
                    serialization_type=SerializationType.JSON,
                    circuit_serialization=serialized_circuit,
                ),
            )

            results = self._submit(run_opts, session.id)
            trial_results.append(results)

        return trial_results

    def run_batch(
        self,
        programs: Sequence[cirq.AbstractCircuit],
        params_list: Optional[Sequence[cirq.Sweepable]] = None,
        repetitions: Union[int, Sequence[int]] = 1,
        session: Union[str, ScalewaySession] = None,
    ) -> Sequence[Sequence[cirq.Result]]:
        params_list, repetitions = self._normalize_batch_args(programs, params_list, repetitions)

        return [
            self.run_sweep(circuit, params=params, repetitions=repetitions, session=session)
            for circuit, params, repetitions in zip(programs, params_list, repetitions)
        ]

    def run(
        self,
        program: 'cirq.AbstractCircuit',
        param_resolver: 'cirq.ParamResolverOrSimilarType' = None,
        repetitions: int = 1,
        session: Union[str, ScalewaySession] = None,
    ) -> cirq.Result:
        return self.run_sweep(program, param_resolver, repetitions, session)[0]
