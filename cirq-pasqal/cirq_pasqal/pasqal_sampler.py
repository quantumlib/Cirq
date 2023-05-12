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
from typing import List, Optional

import time
import requests

import cirq
import cirq_pasqal


class PasqalSampler(cirq.work.Sampler):
    def __init__(
        self,
        remote_host: str,
        access_token: str = '',
        device: Optional[cirq_pasqal.PasqalDevice] = None,
    ) -> None:
        """Inits PasqalSampler.

        Args:
            remote_host: Address of the remote device.
            access_token: Access token for the remote api.
            device: Optional cirq_pasqal.PasqalDevice to use with
                the sampler.
        """
        self.remote_host = remote_host
        self._authorization_header = {"Authorization": access_token}
        self._device = device

    def _serialize_circuit(
        self,
        circuit: cirq.circuits.AbstractCircuit,
        param_resolver: cirq.study.ParamResolverOrSimilarType,
    ) -> str:
        """Serialize a given Circuit.
        Args:
            circuit: The circuit to be run
            param_resolver: Param resolver for the
        Returns:
            json serialized string
        """
        circuit = cirq.protocols.resolve_parameters(circuit, param_resolver)
        serialized_circuit = cirq.to_json(circuit)

        return serialized_circuit

    def _retrieve_serialized_result(self, task_id: str) -> str:
        """Retrieves the results from the remote Pasqal device
        Args:
            task_id: id of the current task.
        Returns:
            json representation of the results
        """

        url = f'{self.remote_host}/get-result/{task_id}'
        while True:
            response = requests.get(url, headers=self._authorization_header, verify=False)
            response.raise_for_status()

            result = response.text
            if result:
                return result

            time.sleep(1.0)

    def _send_serialized_circuit(
        self, serialization_str: str, repetitions: int = 1
    ) -> cirq.study.Result:
        """Sends the json string to the remote Pasqal device
        Args:
            serialization_str: Json representation of the circuit.
            repetitions: Number of repetitions.
        Returns:
            json representation of the results
        """
        simulate_url = f'{self.remote_host}/simulate/no-noise/submit'
        submit_response = requests.post(
            simulate_url,
            verify=False,
            headers={"Repetitions": str(repetitions), **self._authorization_header},
            data=serialization_str,
        )
        submit_response.raise_for_status()

        task_id = submit_response.text

        result_serialized = self._retrieve_serialized_result(task_id)
        result = cirq.read_json(json_text=result_serialized)

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
        device = self._device
        assert isinstance(
            device, cirq_pasqal.PasqalDevice
        ), "Device must inherit from cirq.PasqalDevice."
        device.validate_circuit(program)
        trial_results = []

        for param_resolver in cirq.study.to_resolvers(params):
            json_str = self._serialize_circuit(circuit=program, param_resolver=param_resolver)
            results = self._send_serialized_circuit(
                serialization_str=json_str, repetitions=repetitions
            )
            trial_results.append(results)

        return trial_results
