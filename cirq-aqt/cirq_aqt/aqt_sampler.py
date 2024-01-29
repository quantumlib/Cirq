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

[https://www.aqt.eu](https://www.aqt.eu){:.external}

API keys for classical simulators and quantum devices can be obtained at:

[https://gateway-portal.aqt.eu/](https://gateway-portal.aqt.eu/){:.external}

"""

import json
import time
import uuid
from typing import cast, Dict, List, Sequence, Tuple, Union
from urllib.parse import urljoin

import numpy as np
from requests import post, get

import cirq
from cirq_aqt.aqt_device import AQTSimulator, get_op_string


class AQTSampler(cirq.Sampler):
    """cirq.Sampler for the AQT ion trap device
    This sampler connects to the AQT machine and
    runs a single circuit or an entire sweep remotely
    """

    def __init__(self, workspace: str, resource: str, access_token: str, remote_host: str = "https://arnica.aqt.eu/api/v1/"):
        """Inits AQTSampler.

        Args:
            workspace: the ID of the workspace you have access to.
            resource: the ID of the resource to run the circuit on.
            access_token: Access token for the AQT API.
            remote_host: Address of the AQT API.
        """
        self.workspace = workspace
        self.resource = resource
        self.remote_host = remote_host
        self.access_token = access_token

    @staticmethod
    def fetch_resources(access_token: str, remote_host: str = "https://arnica.aqt.eu/api/v1/") -> None:
        """Lists the workspaces and resources that are accessible with access_token.

        Prints a table to STDOUT containing the workspaces and resources that the
        passed access_token gives access to. The IDs in this table can be used to
        submit jobs using the run and run_sweep methods.

        The printed table contains four columns:
            - WORKSPACE ID: the ID of the workspace. Use this value to submit circuits.
            - RESOURCE NAME: the human-readable name of the resource.
            - RESOURCE ID: the ID of the resource. Use this value to submit circuits.
            - D/S: whether the resource is a (D)evice or (S)imulator.

        Args:
            access_token: Access token for the AQT API.
            remote_host: Address of the AQT API. Defaults to "https://arnica.aqt.eu/api/v1/".

        Raises:
            RuntimeError: If there was an unexpected response from the server.
        """
        headers = {"Authorization": f"Bearer {access_token}", "SDK": "cirq"}
        url = urljoin(remote_host if remote_host[-1] == "/" else remote_host + "/", "workspaces")

        response = get(url, headers=headers)
        if response.status_code != 200:
            raise RuntimeError('Got unexpected return data from server: \n' + str(response.json()))
        
        workspaces = cast(list, response.json())
        col_widths = [19, 21, 20, 3]

        for workspace in workspaces:
            col_widths[0] = max(col_widths[0], len(workspace['id']))
            for resource in workspace['resources']:
                col_widths[1] = max(col_widths[1], len(resource['name']))
                col_widths[2] = max(col_widths[2], len(resource['id']))

        print("+-" + col_widths[0]*"-"+ "-+-" + col_widths[1]*"-" + "-+-" + col_widths[2]*"-"  + "-+-" + col_widths[3]*"-" + "-+")
        print(f"| {'WORKSPACE ID'.ljust(col_widths[0])} | {'RESOURCE NAME'.ljust(col_widths[1])} | {'RESOURCE ID'.ljust(col_widths[2])} | {'D/S'.ljust(col_widths[3])} |" )
        print("+-" + col_widths[0]*"-"+ "-+-" + col_widths[1]*"-" + "-+-" + col_widths[2]*"-"  + "-+-" + col_widths[3]*"-" + "-+")

        for workspace in workspaces:            
            next_workspace = workspace['id']
            for resource in workspace["resources"]:
                print(f"| {next_workspace.ljust(col_widths[0])} | {resource['name'].ljust(col_widths[1])} | {resource['id'].ljust(col_widths[2])} | {resource['type'][0].upper().ljust(col_widths[3])} |" )
                next_workspace = ""
            print(f"+-----------------------+-----------------------+----------------------+---+" )

    def _generate_json(
        self, circuit: cirq.AbstractCircuit, param_resolver: cirq.ParamResolverOrSimilarType
    ) -> str:
        """Generates the JSON string from a Circuit.

        The json format is defined as follows:

        [[op_string,gate_exponent,qubits]]

        which is a list of sequential quantum operations,
        each operation defined by:

        op_string: str that specifies the operation type: "Z","MS","R","Meas"
        gate_exponent: float that specifies the gate_exponent of the operation
        qubits: list of qubits where the operation acts on.

        Args:
            circuit: Circuit to be run.
            param_resolver: Param resolver for resolving parameters in circuit.

        Returns:
            json formatted string of the sequence.

        Raises:
            RuntimeError: If the circuit is empty.
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
    
    def _parse_legacy_circuit_json(self, json_str: str) -> list:
        """Converts a legacy JSON circuit representation.
        
        Converts a JSON created for the legacy API into one that will work
        with the new API.

        Args:
            json_str: A JSON-formatted string that could be used as the
                data parameter in the body of a request to the old AQT API.
        """
        circuit = []
        number_of_measurements = 0

        for legacy_op in json.loads(json_str):
            if number_of_measurements > 0:
                raise ValueError(
                    "Need exactly one `MEASURE` operation at the end of the circuit."
                )

            instruction = {}

            if legacy_op[0] == "Z":
                instruction["operation"] = "RZ"
                instruction["qubit"] = legacy_op[2][0]
                instruction["phi"] = legacy_op[1]

            elif legacy_op[0] == "R":
                instruction["operation"] = "R"
                instruction["qubit"] = legacy_op[3][0]
                instruction["theta"] = legacy_op[1]
                instruction["phi"] = legacy_op[2]

            elif legacy_op[0] == "MS":
                instruction["operation"] = "RXX"
                instruction["qubits"] = legacy_op[2]
                instruction["theta"] = legacy_op[1]

            elif legacy_op[0] == "Meas":
                instruction["operation"] = "MEASURE"
                number_of_measurements += 1

            else:
                raise ValueError(f'Got unknown gate on operation: {legacy_op}.')
            
            circuit.append(instruction)
        
        if circuit[-1]["operation"] != "MEASURE":
            circuit.append({"operation": "MEASURE"})

        return circuit

    def _send_json(
        self,
        *,
        json_str: str,
        id_str: str,
        repetitions: int = 1,
        num_qubits: int = 1,
    ) -> np.ndarray:
        """Sends the json string to the remote AQT device.

        Submits a pre-prepared JSON string representing a circuit to the AQT
        API, then polls for the result, which is parsed and returned when 
        available.

        Please consider that due to the potential for long wait-times, there is
        no timeout in the result polling.

        Args:
            json_str: Json representation of the circuit.
            id_str: A label to help identify a datapoint.
            repetitions: Number of repetitions.
            num_qubits: Number of qubits present in the device.

        Returns:
            Measurement results as an array of boolean.

        Raises:
            RuntimeError: If there was an unexpected response from the server.
        """
        headers = {"Authorization": f"Bearer {self.access_token}", "SDK": "cirq"}
        quantum_circuit = self._parse_legacy_circuit_json(json_str)
        submission_data = {
            "job_type": "quantum_circuit",
            "label": id_str,
            "payload": {
                "circuits": [
                    {
                        "repetitions": repetitions,
                        "quantum_circuit": quantum_circuit,
                        "number_of_qubits": num_qubits,
                    },
                ],
            },
        }

        submission_url = urljoin(self.remote_host, f"submit/{self.workspace}/{self.resource}")

        response = post(
            submission_url,
            json=submission_data,
            headers=headers,
        )
        response = response.json()
        data = cast(Dict, response)

        if 'response' not in data.keys() or 'status' not in data['response'].keys():
            raise RuntimeError('Got unexpected return data from server: \n' + str(data))
        if data['response']['status'] == 'error':
            raise RuntimeError('AQT server reported error: \n' + str(data))

        if 'job' not in data.keys() or 'job_id' not in data['job'].keys():
            raise RuntimeError('Got unexpected return data from AQT server: \n' + str(data))
        job_id = data['job']['job_id']

        result_url = urljoin(self.remote_host, f"result/{job_id}")
        while True:
            response = get(result_url, headers=headers)
            response = response.json()
            data = cast(Dict, response)

            if 'response' not in data.keys() or 'status' not in data['response'].keys():
                raise RuntimeError('Got unexpected return data from AQT server: \n' + str(data))
            if data['response']['status'] == 'finished':
                break
            elif data['response']['status'] == 'error':
                raise RuntimeError('Got unexpected return data from AQT server: \n' + str(data))
            time.sleep(1.0)

        if 'result' not in data['response'].keys():
            raise RuntimeError('Got unexpected return data from AQT server: \n' + str(data))
        
        measurement_int = data['response']['result']['0']
        measurements = np.zeros((repetitions, num_qubits), dtype=int)
        for i, repetition in enumerate(measurement_int):
            for j in range(num_qubits):
                measurements[i, j] = repetition[j]
        
        return measurements

    def run_sweep(
        self, program: cirq.AbstractCircuit, params: cirq.Sweepable, repetitions: int = 1
    ) -> Sequence[cirq.Result]:
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
        trial_results: List[cirq.Result] = []
        for param_resolver in cirq.to_resolvers(params):
            id_str = str(uuid.uuid1())
            num_qubits = len(program.all_qubits())
            json_str = self._generate_json(circuit=program, param_resolver=param_resolver)
            results = self._send_json(
                json_str=json_str, id_str=id_str, repetitions=repetitions, num_qubits=num_qubits
            )
            results = results.astype(bool)
            res_dict = {meas_name: results}
            trial_results.append(cirq.ResultDict(params=param_resolver, measurements=res_dict))
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

    def __init__(self, workspace: str = "", resource: str = "", access_token: str = "", remote_host: str = "", simulate_ideal: bool = False):
        """Args:
        workspace: Workspace is not used by the local simulator.
        resource: Resource is not used by the local simulator.
        access_token: Access token is not used by the local simulator.
        remote_host: Remote host is not used by the local simulator.
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
        id_str: str,
        repetitions: int = 1,
        num_qubits: int = 1,
    ) -> np.ndarray:
        """Replaces the remote host with a local simulator

        Args:
            json_str: Json representation of the circuit.
            id_str: A label to help identify a datapoint.
            repetitions: Number of repetitions.
            num_qubits: Number of qubits present in the device.

        Returns:
            Measurement results as an ndarray of booleans.
        """
        sim = AQTSimulator(num_qubits=num_qubits, simulate_ideal=self.simulate_ideal)
        sim.generate_circuit_from_list(json_str)
        data = sim.simulate_samples(repetitions)
        return data.measurements['m']
