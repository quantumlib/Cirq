from typing import List, Dict, cast
import uuid

import time
import requests
import numpy as np

import cirq
from cirq import Circuit, devices, Sampler, study, resolve_parameters, protocols
from cirq import DensityMatrixSimulator
from cirq.pasqal import PasqalDevice, PasqalNoiseModel



class PasqalCircuit(Circuit):

    def __init__(self,
                 cirq_circuit,
                 device
                 ):

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

    def _serialize_circuit(self, circuit):

        # Serialize the resolved circuit
        serialized_circuit = cirq.to_json(circuit)

        return serialized_circuit


    def _send_serialized_circuit(self,
                                 *,
                                 serialization_str,
                                 id_str,
                                 repetitions= 1,
                                 remote_host,
                                 access_token,
                                 ):

        simulate_url = f'{remote_host}/simulate/no-noise'
        result_url = f'{remote_host}/get-result'

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
        print("\nTask ID =", task_id)

        # Retrieve results
        print("\nWaiting for result...")
        time.sleep(1)

        result_response = requests.get(
            f'{result_url}/{task_id}',
            verify=False,
        )

        result = cirq.read_json(json_text=result_response.content)
        print("\nResult = ", result)

        return result



    def run_sweep(self,
                  program: 'Circuit',
                  remote_host: str,
                  params: study.Sweepable,
                  simulate_ideal : bool,
                  access_token: str,
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
        meas_name = 'm'  # TODO: Get measurement name from circuit.

        # Complain if this is not using the PasqalDevice
        #assert isinstance(program.device, PasqalDevice)

        trial_results = []
        for param_resolver in study.to_resolvers(params):

            id_str = uuid.uuid1()
            json_str = self._serialize_circuit(circuit=program)

            results = self._send_serialized_circuit(
                serialization_str=json_str,
                id_str=id_str,
                remote_host=remote_host,
                access_token=access_token,
                repetitions=repetitions
                )
            trial_results.append(results)
            #results = results.astype(bool)
            #res_dict = {meas_name: results}
            #trial_results.append(
            #    study.TrialResult(params=param_resolver,
#                                       measurements=res_dict))

        return trial_results


    def run(self,
                  program: 'Circuit',
                  remote_host: str,
                  param_resolver: 'cirq.ParamResolverOrSimilarType' = None,
                  simulate_ideal : bool=True,
                  access_token: str='',
                  repetitions: int = 1
                  ) -> List[study.TrialResult]:

        trial_results=self.run_sweep(program,
                              remote_host,
                              study.ParamResolver(param_resolver),
                              simulate_ideal,
                              access_token,
                              repetitions)[0]


        return trial_results
