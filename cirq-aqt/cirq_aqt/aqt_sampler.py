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
from typing import Callable, cast, Dict, List, Literal, Sequence, Tuple, TypedDict, Union
from urllib.parse import urljoin

import numpy as np
from requests import get, post

import cirq
from cirq_aqt.aqt_device import AQTSimulator, get_op_string, OperationString

_DEFAULT_HOST = "https://arnica.aqt.eu/api/v1/"


class SingleQubitGate(TypedDict):
    """Abstract single qubit rotation."""

    qubit: int


class GateRZ(SingleQubitGate):
    """A single-qubit rotation rotation around the Bloch sphere's z-axis."""

    operation: Literal["RZ"]
    phi: float


class GateR(SingleQubitGate):
    """A single-qubit rotation around an arbitrary axis on the Bloch sphere's equatorial plane."""

    operation: Literal["R"]
    phi: float
    theta: float


class GateRXX(TypedDict):
    """A two-qubit entangling gate of Mølmer-Sørenson-type."""

    operation: Literal["RXX"]
    qubits: list[int]
    theta: float


class Measure(TypedDict):
    """Measurement operation.

    The MEASURE operation instructs the resource
    to perform a projective measurement of all qubits.
    """

    operation: Literal["MEASURE"]


Gate = GateRZ | GateR | GateRXX
Operation = Gate | Measure


class Resource(TypedDict):
    """A quantum computing device."""

    id: str
    name: str
    type: Literal["device", "simulator"]


class Workspace(TypedDict):
    """A user workspace."""

    id: str
    resources: list[Resource]


class AQTSampler(cirq.Sampler):
    """cirq.Sampler for the AQT ion trap device
    This sampler connects to the AQT machine and
    runs a single circuit or an entire sweep remotely
    """

    def __init__(
        self, workspace: str, resource: str, access_token: str, remote_host: str = _DEFAULT_HOST
    ):
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
    def fetch_resources(access_token: str, remote_host: str = _DEFAULT_HOST) -> list[Workspace]:
        """Lists the workspaces and resources that are accessible with access_token.

        Returns a list containing the workspaces and resources that the passed
        access_token gives access to. The workspace and resource IDs in this list can be
        used to submit jobs using the run and run_sweep methods.

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

        workspaces = [
            Workspace(
                id=w['id'],
                resources=[
                    Resource(id=r['id'], name=r['name'], type=r['type']) for r in w['resources']
                ],
            )
            for w in response.json()
        ]

        return workspaces

    @staticmethod
    def print_resources(
        access_token: str, emit: Callable = print, remote_host: str = _DEFAULT_HOST
    ) -> None:
        """Displays the workspaces and resources that are accessible with access_token.

        Prints a table using the function passed as 'emit' containing the workspaces and
        resources that the passed access_token gives access to. The IDs in this table
        can be used to submit jobs using the run and run_sweep methods.

        The printed table contains four columns:
            - WORKSPACE ID: the ID of the workspace. Use this value to submit circuits.
            - RESOURCE NAME: the human-readable name of the resource.
            - RESOURCE ID: the ID of the resource. Use this value to submit circuits.
            - D/S: whether the resource is a (D)evice or (S)imulator.

        Args:
            access_token: Access token for the AQT API.
            emit (optional): A Callable which will be called once with a single string argument,
                containing the table. Defaults to print from the standard library.
            remote_host (optional): Address of the AQT API. Defaults to
                "https://arnica.aqt.eu/api/v1/".

        Raises:
            RuntimeError: If there was an unexpected response from the server.
        """
        table_lines = []
        workspaces = AQTSampler.fetch_resources(access_token, remote_host)

        if len(workspaces) == 0:
            return emit("No workspaces are accessible with this access token.")
        if any(len(w['resources']) == 0 for w in workspaces):
            return emit("No workspaces accessible with this access token contain resources.")

        col_widths = [
            max([len(w['id']) for w in workspaces]),
            max([len(d['name']) for w in workspaces for d in w['resources']]),
            max([len(d['id']) for w in workspaces for d in w['resources']]),
            3,
        ]
        SEPARATOR = "+-" + "-+-".join(col_width * "-" for col_width in col_widths) + "-+"

        table_lines.append(SEPARATOR)
        table_lines.append(
            f"| {'WORKSPACE ID'.ljust(col_widths[0])} |"
            f" {'RESOURCE NAME'.ljust(col_widths[1])} |"
            f" {'RESOURCE ID'.ljust(col_widths[2])} |"
            f" {'D/S'.ljust(col_widths[3])} |"
        )
        table_lines.append(SEPARATOR)

        for workspace in workspaces:
            next_workspace = workspace['id']
            for resource in workspace["resources"]:
                table_lines.append(
                    f"| {next_workspace.ljust(col_widths[0])} |"
                    f" {resource['name'].ljust(col_widths[1])} |"
                    f" {resource['id'].ljust(col_widths[2])} |"
                    f" {resource['type'][0].upper().ljust(col_widths[3])} |"
                )
                next_workspace = ""
            table_lines.append(SEPARATOR)

        emit("\n".join(table_lines))

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

        seq_list: List[Union[Tuple[str, float, List[int]], Tuple[str, float, float, List[int]]]] = (
            []
        )
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

    def _parse_legacy_circuit_json(self, json_str: str) -> list[Operation]:
        """Converts a legacy JSON circuit representation.

        Converts a JSON created for the legacy API into one that will work
        with the Arnica v1 API.

        Raises:
            ValueError:
                * if there is not exactly one measurement operation at the end
                    of the circuit.

                * if an operation is found in json_str that is not in
                    OperationString.

        Args:
            json_str: A JSON-formatted string that could be used as the
                data parameter in the body of a request to the old AQT API.
        """
        circuit = []
        number_of_measurements = 0
        instruction: Operation

        for legacy_op in json.loads(json_str):
            if number_of_measurements > 0:
                raise ValueError("Need exactly one `MEASURE` operation at the end of the circuit.")

            if legacy_op[0] == OperationString.Z.value:
                instruction = GateRZ(operation="RZ", qubit=legacy_op[2][0], phi=legacy_op[1])

            elif legacy_op[0] == OperationString.R.value:
                instruction = GateR(
                    operation="R", qubit=legacy_op[3][0], theta=legacy_op[1], phi=legacy_op[2]
                )

            elif legacy_op[0] == OperationString.MS.value:
                instruction = GateRXX(operation="RXX", qubits=legacy_op[2], theta=legacy_op[1])

            elif legacy_op[0] == OperationString.MEASURE.value:
                instruction = Measure(operation="MEASURE")
                number_of_measurements += 1

            else:
                raise ValueError(f'Got unknown gate on operation: {legacy_op}.')

            circuit.append(instruction)

        if circuit[-1]["operation"] != "MEASURE":
            circuit.append({"operation": "MEASURE"})

        return circuit

    def _send_json(
        self, *, json_str: str, id_str: str, repetitions: int = 1, num_qubits: int = 1
    ) -> np.ndarray:
        """Sends the json string to the remote AQT device.

        Submits a pre-prepared JSON string representing a circuit to the AQT
        API, then polls for the result, which is parsed and returned when
        available.

        Please consider that due to the potential for long wait-times, there is
        no timeout in the result polling.

        Args:
            json_str: Json representation of the circuit.
            id_str: A label to help identify a circuit.
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
                    }
                ]
            },
        }

        submission_url = urljoin(self.remote_host, f"submit/{self.workspace}/{self.resource}")

        response = post(submission_url, json=submission_data, headers=headers)
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

    def __init__(
        self,
        workspace: str = "",
        resource: str = "",
        access_token: str = "",
        remote_host: str = "",
        simulate_ideal: bool = False,
    ):
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
        self, *, json_str: str, id_str: str, repetitions: int = 1, num_qubits: int = 1
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
