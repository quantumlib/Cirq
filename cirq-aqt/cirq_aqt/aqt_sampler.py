# Copyright 2019 The Cirq Developers
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
"""Samplers to access the AQT ion trap devices via the provided API. For
more information on these devices see the AQT homepage:

https://www.aqt.eu

API keys for classical simulators and quantum devices can be obtained at:

https://gateway-portal.aqt.eu/

"""

import json
import time
import uuid
from typing import List, Union, Tuple, Dict, cast

import numpy as np
from requests import put

import cirq
from cirq_aqt.aqt_device import AQTSimulator, get_op_string


class AQTSampler(cirq.Sampler):
    """cirq.Sampler for the AQT ion trap device
    This sampler connects to the AQT machine and
    runs a single circuit or an entire sweep remotely
    """

    def __init__(self, remote_host: str, access_token: str):
        """Inits AQTSampler.

        Args:
            remote_host: Address of the remote device.
            access_token: Access token for the remote api.
        """
        self.remote_host = remote_host
        self.access_token = access_token

    # TODO(#3388) Add documentation for Raises.
    # pylint: disable=missing-raises-doc
    def _generate_json(
        self,
        circuit: cirq.AbstractCircuit,
        param_resolver: cirq.ParamResolverOrSimilarType,
    ) -> str:
        """Generates the JSON string from a Circuit.

        The json format is defined as follows:

        [[op_string,gate_exponent,qubits]]

        which is a list of sequential quantum operations,
        each operation defined by:

        op_string: str that specifies the operation type: "X","Y","Z","MS"
        gate_exponent: float that specifies the gate_exponent of the operation
        qubits: list of qubits where the operation acts on.

        Args:
            circuit: Circuit to be run
            param_resolver: Param resolver for the

        Returns:
            json formatted string of the sequence
        """

        seq_list: List[
            Union[Tuple[str, float, List[int]], Tuple[str, float, float, List[int]]]
        ] = []
        circuit = cirq.resolve_parameters(circuit, param_resolver)
        for op in circuit.all_operations():
            line_qubit = cast(Tuple[cirq.LineQubit], op.qubits)
            op = cast(cirq.GateOperation, op)
            qubit_idx = [obj.x for obj in line_qubit]
            op_str = get_op_string(op)
            gate: Union[cirq.EigenGate, cirq.PhasedXPowGate]
            if op_str == 'R':
                gate = cast(cirq.PhasedXPowGate, op.gate)
                seq_list.append(
                    (op_str, float(gate.exponent), float(gate.phase_exponent), qubit_idx)
                )
            else:
                gate = cast(cirq.EigenGate, op.gate)
                seq_list.append((op_str, float(gate.exponent), qubit_idx))
        if len(seq_list) == 0:
            raise RuntimeError('Cannot send an empty circuit')
        json_str = json.dumps(seq_list)
        return json_str

    # TODO(#3388) Add documentation for Raises.
    def _send_json(
        self,
        *,
        json_str: str,
        id_str: Union[str, uuid.UUID],
        repetitions: int = 1,
        num_qubits: int = 1,
    ) -> np.ndarray:
        """Sends the json string to the remote AQT device

        The interface is given by PUT requests to a single endpoint URL.
        The first PUT will insert the circuit into the remote queue,
        given a valid access key.
        Every subsequent PUT will return a dictionary, where the key "status"
        is either 'queued', if the circuit has not been processed yet or
        'finished' if the circuit has been processed.
        The experimental data is returned via the key 'data'

        Args:
            json_str: Json representation of the circuit.
            id_str: Unique id of the datapoint.
            repetitions: Number of repetitions.
            num_qubits: Number of qubits present in the device.

        Returns:
            Measurement results as an array of boolean.
        """
        header = {"Ocp-Apim-Subscription-Key": self.access_token, "SDK": "cirq"}
        response = put(
            self.remote_host,
            data={
                'data': json_str,
                'access_token': self.access_token,
                'repetitions': repetitions,
                'no_qubits': num_qubits,
            },
            headers=header,
        )
        response = response.json()
        data = cast(Dict, response)
        if 'status' not in data.keys():
            raise RuntimeError('Got unexpected return data from server: \n' + str(data))
        if data['status'] == 'error':
            raise RuntimeError('AQT server reported error: \n' + str(data))

        if 'id' not in data.keys():
            raise RuntimeError('Got unexpected return data from AQT server: \n' + str(data))
        id_str = data['id']

        while True:
            response = put(
                self.remote_host,
                data={'id': id_str, 'access_token': self.access_token},
                headers=header,
            )
            response = response.json()
            data = cast(Dict, response)
            if 'status' not in data.keys():
                raise RuntimeError('Got unexpected return data from AQT server: \n' + str(data))
            if data['status'] == 'finished':
                break
            elif data['status'] == 'error':
                raise RuntimeError('Got unexpected return data from AQT server: \n' + str(data))
            time.sleep(1.0)
        measurements_int = data['samples']
        measurements = np.zeros((len(measurements_int), num_qubits))
        for i, result_int in enumerate(measurements_int):
            measurement_int_bin = format(result_int, f'0{num_qubits}b')
            for j in range(num_qubits):
                measurements[i, j] = int(measurement_int_bin[j])
        return measurements

    # pylint: enable=missing-raises-doc
    def run_sweep(
        self, program: cirq.AbstractCircuit, params: cirq.Sweepable, repetitions: int = 1
    ) -> List[cirq.Result]:
        """Samples from the given Circuit.

        In contrast to run, this allows for sweeping over different parameter
        values.

        Args:
            program: The circuit to simulate.
            Should be generated using AQTSampler.generate_circuit_from_list
            params: Parameters to run with the program.
            repetitions: The number of repetitions to simulate.

        Returns:
            Result list for this run; one for each possible parameter
            resolver.
        """
        # TODO: Use measurement name from circuit.
        # Github issue: https://github.com/quantumlib/Cirq/issues/2199
        meas_name = 'm'
        assert isinstance(program.device, cirq.IonDevice)
        trial_results = []  # type: List[cirq.Result]
        for param_resolver in cirq.to_resolvers(params):
            id_str = uuid.uuid1()
            num_qubits = len(program.device.qubits)
            json_str = self._generate_json(circuit=program, param_resolver=param_resolver)
            results = self._send_json(
                json_str=json_str, id_str=id_str, repetitions=repetitions, num_qubits=num_qubits
            )
            results = results.astype(bool)
            res_dict = {meas_name: results}
            trial_results.append(cirq.Result(params=param_resolver, measurements=res_dict))
        return trial_results


class AQTSamplerLocalSimulator(AQTSampler):
    """cirq.Sampler using the AQT simulator on the local machine.

    Can be used as a replacement for the AQTSampler
    When the attribute simulate_ideal is set to True,
    an ideal circuit is sampled
    If not, the error model defined in aqt_simulator_test.py is used
    Example for running the ideal sampler:

    sampler = AQTSamplerLocalSimulator()
    sampler.simulate_ideal=True
    """

    def __init__(self, remote_host: str = '', access_token: str = '', simulate_ideal: bool = False):
        """Args:
        remote_host: Remote host is not used by the local simulator.
        access_token: Access token is not used by the local simulator.
        simulate_ideal: Boolean that determines whether a noisy or
                        an ideal simulation is performed.
        """
        self.remote_host = remote_host
        self.access_token = access_token
        self.simulate_ideal = simulate_ideal

    def _send_json(
        self,
        *,
        json_str: str,
        id_str: Union[str, uuid.UUID],
        repetitions: int = 1,
        num_qubits: int = 1,
    ) -> np.ndarray:
        """Replaces the remote host with a local simulator

        Args:
            json_str: Json representation of the circuit.
            id_str: Unique id of the datapoint.
            repetitions: Number of repetitions.
            num_qubits: Number of qubits present in the device.

        Returns:
            Measurement results as an ndarray of booleans.
        """
        sim = AQTSimulator(num_qubits=num_qubits, simulate_ideal=self.simulate_ideal)
        sim.generate_circuit_from_list(json_str)
        data = sim.simulate_samples(repetitions)
        return data.measurements['m']
