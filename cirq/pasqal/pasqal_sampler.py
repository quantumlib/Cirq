from typing import List, Union
import uuid

import time
import requests

import cirq
from cirq import devices, study, protocols
from cirq.protocols.resolve_parameters import resolve_parameters
from cirq.circuits import Circuit
from cirq.work import Sampler

from cirq.pasqal import PasqalDevice


class PasqalSampler(Sampler):

    def __init__(self, remote_host: str, access_token: str = '') -> None:
        """
        Args:
            remote_host: Address of the remote device.
            access_token: Access token for the remote api.
        """
        self.remote_host = remote_host

        self._authorization_header = {
            "Authorization": access_token
        }

    def _serialize_circuit(self,
                           circuit: Circuit,
                           param_resolver: study.ParamResolverOrSimilarType
                           ) -> str:

        # Serialize the resolved circuit
        circuit = resolve_parameters(circuit, param_resolver)
        serialized_circuit = cirq.to_json(circuit)

        return serialized_circuit

    def _retrieve_serialized_result(self, task_id: str) -> str:
        url = f'{self.remote_host}/get-result/{task_id}'

        while True:
            response = requests.get(
                url,
                headers=self._authorization_header,
                verify=False,
            )
            response.raise_for_status()

            result = response.text
            if result:
                return result

            time.sleep(1.0)

    def _send_serialized_circuit(self,
                                 *,
                                 serialization_str: str,
                                 id_str: Union[str, uuid.UUID],
                                 repetitions: int = 1
                                 ) -> study.TrialResult:

        simulate_url = f'{self.remote_host}/simulate/no-noise/submit'

        submit_response = requests.post(
            simulate_url,
            verify=False,
            headers={
                "Repetitions": str(repetitions),
                **self._authorization_header,
            },
            data=serialization_str,
        )
        submit_response.raise_for_status()

        task_id = submit_response.text

        result_serialized = self._retrieve_serialized_result(task_id)
        result = cirq.read_json(json_text=result_serialized)

        return result



    def run_sweep(self,
                  program: 'Circuit',
                  params: study.Sweepable,
                  simulate_ideal : bool,
                  repetitions: int
                  ) -> List[study.TrialResult]:
        """Samples from the given Circuit.
        In contrast to run, this allows for sweeping over different parameter
        values.
        Args:
            program: The circuit to simulate.
            params: Parameters to run with the program.
            repetitions: The number of repetitions to simulate.
        Returns:
            TrialResult list for this run; one for each possible parameter
            resolver.
        """
        #meas_name = 'm'
        # Complain if this is not using the PasqalDevice
        assert isinstance(program.device, PasqalDevice)

        trial_results = []
        for param_resolver in study.to_resolvers(params):

            id_str = uuid.uuid1()
            json_str = self._serialize_circuit(circuit=program,
                                               param_resolver=param_resolver)

            results = self._send_serialized_circuit(
                serialization_str=json_str,
                id_str=id_str,
                repetitions=repetitions
                )
            trial_results.append(results)

        return trial_results


    def run(self,
            program: 'Circuit',
            param_resolver: 'cirq.ParamResolverOrSimilarType' = None,
            simulate_ideal : bool=True,
            repetitions: int = 1
            ) -> List[study.TrialResult]:

        trial_results=self.run_sweep(program,
                              study.ParamResolver(param_resolver),
                              simulate_ideal,
                              repetitions)[0]


        return trial_results
