from typing import List, Dict, cast
import uuid

import requests
import numpy as np

from cirq import Circuit, devices, Sampler, study, resolve_parameters
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


class PasqalSampler(Sampler):

    def _serialize_circuit(self, circuit, param_resolver):

        # Generate a resolved circuit
        resolved_circuit = resolve_parameters(circuit, param_resolver)

        # Serialize the resolved circuit
        serialized_circuit = repr(resolved_circuit)

        return serialized_circuit


    def _send_serialized_circuit(self,
                                 *,
                                 serialization_str,
                                 id_str,
                                 repetitions= 1,
                                 num_qubits= 1,
                                 remote_host,
                                 access_token,
                                 ):
        """Sends the serialization string to the remote Pasqal device
        The interface is given by PUT requests to a single endpoint URL.
        The first PUT will insert the circuit into the remote queue,
        given a valid access key.
        Every subsequent PUT will return a dictionary, where the key "status"
        is either 'queued', if the circuit has not been processed yet or
        'finished' if the circuit has been processed.
        The experimental data is returned via the key 'data'
        Args:
            serialization_str: representation of the circuit.
            id_str: Unique id of the datapoint.
            repetitions: Number of repetitions.
            num_qubits: Number of qubits present in the device.
        Returns:
            Measurement results as an array of boolean.
        """
        # header = {"Ocp-Apim-Subscription-Key": access_token, "SDK": "cirq"}
        #print(remote_host)
        response = requests.post(remote_host,
                                data={
                           'cirq_circuit_repr': serialization_str,
                           # 'access_token': access_token,
                           'nr_repetitions': repetitions
                       }
                        # , headers = header
                    )
        #print(response)
        response = response.json()

        data = cast(Dict, response)

        # Status of the response
        if 'status' not in data.keys():
            raise RuntimeError('Got unexpected return data from server: \n' +
                               str(data))
        if data['status'] == 'error':
            raise RuntimeError('Pasqal server reported error: \n' + str(data))

        """
            No ID for the moment
        """
        # # ID of the job on the remote cloud
        # if 'id' not in data.keys():
        #     raise RuntimeError(
        #         'Got unexpected return data
        #           from Pasqal server: \n' + str(data))
        # id_str = data['id']

        """
            No Polling for the moment
        """
        # while True:
        #     response = requests.put(remote_host,
        #                    data={
        #                        'id': id_str,
        #                        'access_token': access_token
        #                    }
        #                     # ,headers=header
        #                 )
        #     response = response.json()
        #
        #     data = cast(Dict, response)
        #     if 'status' not in data.keys():
        #         raise RuntimeError(
        #             'Got unexpected return data from Pasqal server: \n' +
        #             str(data))
        #     if data['status'] == 'finished':
        #         break
        #     elif data['status'] == 'error':
        #         raise RuntimeError(
        #             'Got unexpected return data from Pasqal server: \n' +
        #             str(data))
        #     time.sleep(1.0)
        print(data['samples'])
        measurements_int = data['samples']
        measurements = np.zeros((len(measurements_int), num_qubits))

        for i, result_int in enumerate(measurements_int):
            for j in range(num_qubits):
                measurements[i, j] = np.floor(result_int / 2**j)

        return measurements


    def simulate_samples(self, program: 'Circuit',
                         simulate_ideal :bool,
                         repetitions: int) -> study.TrialResult:
        """Samples the circuit
        Args:
        program: The circuit to simulate.
        repetitions: Number of times the circuit is simulated
        Returns:
        TrialResult from Cirq.Simulator
        """
        if simulate_ideal:
            noise_model = devices.NO_NOISE
        else:
            noise_model= PasqalNoiseModel()

        sim = DensityMatrixSimulator(noise = noise_model)
        result = sim.run(program, repetitions=repetitions)

        return result

    def run_sweep(self,
                  program: 'Circuit',
                  params: study.Sweepable,
                  remote_host: str = 'local_host',
                  simulate_ideal : bool=True,
                  access_token: str='',
                  repetitions: int = 1
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
        assert isinstance(program.device, PasqalDevice)

        trial_results = []
        for param_resolver in study.to_resolvers(params):

            id_str = uuid.uuid1()
            num_qubits = len(program.device.qubits)
            json_str = self._serialize_circuit(circuit=program,
                                               param_resolver=param_resolver)

            results = self._send_serialized_circuit(
                serialization_str=json_str,
                id_str=id_str,
                num_qubits=num_qubits,
                remote_host=remote_host,
                access_token=access_token,
                repetitions=repetitions
                )

            results = results.astype(bool)
            res_dict = {meas_name: results}
            trial_results.append(
                study.TrialResult(params=param_resolver,
                                       measurements=res_dict))

        return trial_results


    def run(self,
                  program: 'Circuit',
                  param_resolver: 'cirq.ParamResolverOrSimilarType' = None,
                  remote_host: str = 'http://0.0.0.0:5000/receive_circuit.cirq',
                  simulate_ideal : bool=True,
                  access_token: str='',
                  repetitions: int = 1
                  ) -> List[study.TrialResult]:

        trial_results=self.run_sweep(program,
                              study.ParamResolver(param_resolver),
                              remote_host,
                              simulate_ideal,
                              access_token,
                              repetitions)[0]


        return trial_results
