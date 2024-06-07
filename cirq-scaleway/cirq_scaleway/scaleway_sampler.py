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
from typing import List, Union, Sequence, Optional, Dict

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
    def __init__(self, client: QaaSClient, device: ScalewayDevice):
        self.__client = client
        self.__device = device

    def run(
        self,
        program: 'cirq.AbstractCircuit',
        param_resolver: 'cirq.ParamResolverOrSimilarType' = None,
        repetitions: int = 1,
        session: Union[str, ScalewaySession, None] = None,
    ) -> cirq.Result:
        """Samples from the given `Circuit`.

        This mode of operation for a sampler will provide results
        in the form of measurement outcomes.  It will not provide
        access to state vectors (even if the underlying
        sampling mechanism is a simulator).  This method will substitute
        parameters in the `param_resolver` attributes for `sympy.Symbols`
        used within the Circuit.  This circuit will be executed a number
        of times specified in the `repetitions` attribute, though some
        simulated implementations may instead sample from the final
        distribution rather than execute the circuit each time.

        If no provided device session, a new one will be automatically created
        with all default parameters.

        Args:
            program: The circuit to sample from.
            param_resolver: Parameters to run with the program.
            repetitions: The number of times to sample.
            session: The target device session to run the quantum circuit.

        Returns:
            `cirq.Result` that contains all the measurements for a run.
        """
        return self.run_sweep(program, param_resolver, repetitions, session)[0]

    def run_sweep(
        self,
        program: cirq.AbstractCircuit,
        params: cirq.study.Sweepable,
        repetitions: int = 1,
        session: Union[str, ScalewaySession, None] = None,
    ) -> List[cirq.study.Result]:
        """Samples from the given Circuit.

        This allows for sweeping over different parameter values,
        unlike the `run` method.  The `params` argument will provide a
        mapping from `sympy.Symbol`s used within the circuit to a set of
        values.  Unlike the `run` method, which specifies a single
        mapping from symbol to value, this method allows a "sweep" of
        values.  This allows a user to specify execution of a family of
        related circuits efficiently.

        If no provided device session, a new one will be automatically created
        with all default parameters.

        Args:
            program: The circuit to sample from.
            params: Parameters to run with the program.
            repetitions: The number of times to sample.
            session: The target device session to run the quantum circuit.

        Returns:
            Result list for this run; one for each possible parameter resolver.
        """
        trial_results = []

        if not session:
            session = self.__device.start_session()

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

            results = self._submit(run_opts, session.id)
            trial_results.append(results)

        return trial_results

    def run_batch(
        self,
        programs: Sequence[cirq.AbstractCircuit],
        params_list: Optional[Sequence[cirq.Sweepable]] = None,
        repetitions: Union[int, Sequence[int]] = 1,
        session: Union[str, ScalewaySession, None] = None,
    ) -> Sequence[Sequence[cirq.Result]]:
        """Runs the supplied circuits.

        Each circuit provided in `programs` will pair with the optional
        associated parameter sweep provided in the `params_list`, and be run
        with the associated repetitions provided in `repetitions` (if
        `repetitions` is an integer, then all runs will have that number of
        repetitions). If `params_list` is specified, then the number of
        circuits is required to match the number of sweeps. Similarly, when
        `repetitions` is a list, the number of circuits is required to match
        the length of this list.

        By default, this method simply invokes `run_sweep` sequentially for
        each (circuit, parameter sweep, repetitions) tuple. Child classes that
        are capable of sampling batches more efficiently should override it to
        use other strategies. Note that child classes may have certain
        requirements that must be met in order for a speedup to be possible,
        such as a constant number of repetitions being used for all circuits.
        Refer to the documentation of the child class for any such requirements.

        If no provided device session, a new one will be automatically created
        with all default parameters.

        Args:
            programs: The circuits to execute as a batch.
            params_list: Parameter sweeps to use with the circuits. The number
                of sweeps should match the number of circuits and will be
                paired in order with the circuits.
            repetitions: Number of circuit repetitions to run. Can be specified
                as a single value to use for all runs, or as a list of values,
                one for each circuit.
            session: The target device session to run the quantum circuit.

        Returns:
            A list of lists of TrialResults. The outer list corresponds to
            the circuits, while each inner list contains the TrialResults
            for the corresponding circuit, in the order imposed by the
            associated parameter sweep.

        Raises:
            ValueError: If length of `programs` is not equal to the length
                of `params_list` or the length of `repetitions`.
        """
        params_list, repetitions = self._normalize_batch_args(programs, params_list, repetitions)

        return [
            self.run_sweep(circuit, params=params, repetitions=repetitions, session=session)
            for circuit, params, repetitions in zip(programs, params_list, repetitions)
        ]

    def _extract_payload_from_response(self, result_response: Dict) -> str:
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

        job_results = self._wait_for_result(job_id, None, 2)
        result = self._to_cirq_result(job_results)

        return result
