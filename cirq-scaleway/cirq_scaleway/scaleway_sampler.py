# Copyright 2024 The Cirq Developers
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

from cirq.study import ResultDict
from typing import List, Optional

from .scaleway_client import QaaSClient
from .scaleway_models import JobPayload, ClientPayload, BackendPayload, RunPayload, SerializationType, CircuitPayload
from .versions import USER_AGENT

class ScalewaySampler(cirq.work.Sampler):
    def __init__(
        self,
        client: QaaSClient,
        sampler_id: str,
        name: str,
        availability: str,
        version: str,
        num_qubits: int,
        metadata: str,
    ) -> None:
        """Inits ScalewaySampler.

        Args:
        """
        self._id = sampler_id
        self._availability = availability
        self._client = client
        self._version = version
        self._num_qubits = num_qubits

    # def _serialize_circuit(
    #     self,
    #     circuit: cirq.circuits.AbstractCircuit,
    #     param_resolver: cirq.study.ParamResolverOrSimilarType,
    # ) -> str:
    #     """Serialize a given Circuit.
    #     Args:
    #         circuit: The circuit to be run
    #         param_resolver: Param resolver for the
    #     Returns:
    #         json serialized string
    #     """
    #     circuit = cirq.protocols.resolve_parameters(circuit, param_resolver)
    #     serialized_circuit = cirq.to_json(circuit)

    #     return serialized_circuit

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
            elapsed = time.time() - start_time

            if timeout is not None and elapsed >= timeout:
                raise Exception("Timed out waiting for result")

            status = self.status()

            if status == JobStatus.DONE:
                return self._client.get_job_results(self._job_id)

            if status == JobStatus.ERROR:
                raise JobError("Job error")

            time.sleep(fetch_interval)

    def _to_cirq_result(self, job_results) -> cirq.Result:
        if len(job_results) == 0:
            raise Exception("Empty result list")

        payload = self._extract_payload_from_response(job_results[0])
        payload_dict = json.loads(payload)
        cirq_result = ResultDict._from_json_dict_(**payload_dict)

        return cirq_result

    def _send_serialized_circuit(
        self, serialization_str: str, repetitions: int = 1
    ) -> cirq.study.Result:

        run_opts = RunPayload(
            options={"shots": options.pop("shots")},
            circuit=CircuitPayload(
                serialization_type=SerializationType.QASM_V2,
                circuit_serialization=qasm2.dumps(circuit),
            ),
        )

        backend_opts = BackendPayload(
            name=self.backend().name,
            version=self.backend().version,
            options=options,
        )

        client_opts = ClientPayload(
            user_agent=USER_AGENT,
        )

        job_payload = JobPayload.schema().dumps(
            JobPayload(
                backend=backend_opts,
                run=run_opts,
                client=client_opts,
            )
        )

        self._job_id = self._client.create_job(
            name=self._name,
            session_id=session_id,
            circuits=job_payload,
        )

        job_results = self._wait_for_result(None, 2)

        result = self._to_cirq_result(job_results)

        return result

    def run_sweep(
        self, program: cirq.AbstractCircuit, params: cirq.study.Sweepable, repetitions: int = 1
    ) -> List[cirq.study.Result]:
        """Samples from the given Circuit.
        In contrast to run, this allows for sweeping over different parameter
        values.
        Args:
            program: The circuit to simulate.
            params: Parameters to run with the program.
            repetitions: The number of repetitions to simulate.
        Returns:
            Result list for this run; one for each possible parameter
            resolver.
        """
        trial_results = []

        for param_resolver in cirq.study.to_resolvers(params):
            # json_str = self._serialize_circuit(circuit=program, param_resolver=param_resolver)
            results = self._send_serialized_circuit(
                serialization_str=json_str, repetitions=repetitions
            )
            trial_results.append(results)

        return trial_results
