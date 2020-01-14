from typing import List, Union
import uuid

import time
import requests

import cirq
from cirq import Circuit, devices, Sampler, study, resolve_parameters, protocols
#from cirq import DensityMatrixSimulator
from cirq.pasqal import PasqalDevice#, PasqalNoiseModel



class PasqalCircuit(Circuit):

    def __init__(self,
                 cirq_circuit: Circuit,
                 device: devices.Device
                 ) -> None:

        if (device is None) \
                or (not isinstance(device, PasqalDevice)):
            raise ValueError("PasqalDevice necessary for constructor!")



        super().__init__([], device)
        for moment in cirq_circuit:
            for op in moment:
                # This should call decompose on the gates
                self.append(op)


    def _json_dict_(self):
        return protocols.obj_to_dict_helper(self, ['moments', 'device'])





class PasqalSampler(Sampler):

    def __init__(self, remote_host: str, access_token: str = '') -> None:
        """
        Args:
            remote_host: Address of the remote device.
            access_token: Access token for the remote api.
        """
        self.remote_host = remote_host
        self.access_token = access_token

    def _serialize_circuit(self,
                           circuit: Circuit,
                           param_resolver: study.ParamResolverOrSimilarType
                           ) -> str:

        # Serialize the resolved circuit
        circuit = resolve_parameters(circuit, param_resolver)
        serialized_circuit = cirq.to_json(circuit)

        return serialized_circuit


    def _send_serialized_circuit(self,
                                 *,
                                 serialization_str: str,
                                 id_str: Union[str, uuid.UUID],
                                 repetitions: int = 1
                                 ) -> study.TrialResult:

        simulate_url = f'{self.remote_host}/simulate/no-noise'
        result_url = f'{self.remote_host}/get-result'

        submit_response = requests.put(
            simulate_url,
            verify=False,
            headers={
                "Repetitions": str(repetitions),
            },
            data=serialization_str,
        )

        # Get task ID
        task_id = submit_response.text

        # Retrieve results
        time.sleep(1)

        result_response = requests.get(
            f'{result_url}/{task_id}',
            verify=False,
        )
        result = cirq.read_json(json_text=result_response.content)

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
        #assert isinstance(program.device, PasqalDevice)

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
