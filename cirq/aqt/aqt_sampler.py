import json
import time
import uuid
from typing import Iterable, List, Union

import numpy as np
from requests import put
from cirq import circuits, schedules, study, Sampler, ops, resolve_parameters
from cirq.study.sweeps import Sweep
from cirq.aqt.aqt_device import AQTSimulator

Sweepable = Union[study.ParamResolver, Iterable[study.ParamResolver], Sweep,
                  Iterable[Sweep]]
"""Samplers for the AQT ion trap device"""


def get_op_string(op_obj: ops.EigenGate):
    """Find the string representation for a given gate
    Params:
        op_obj: Gate object, out of: XXPowGate, XPowGate, YPowGate"""
    if isinstance(op_obj, ops.XXPowGate):
        op_str = 'MS'
    elif isinstance(op_obj, ops.XPowGate):
        op_str = 'X'
    elif isinstance(op_obj, ops.YPowGate):
        op_str = 'Y'
    else:
        raise ValueError('Got unknown gate:', op_obj)
    return op_str


class AQTSampler(Sampler):
    """Sampler for the AQT ion trap device
    This sampler connects to the AQT machine and
    runs a single circuit or an entire sweep remotely
    """

    def __init__(self, remote_host: str, access_token: str):
        """
        Args:
            remote_host: Address of the remote device.
            access_token: Access token for the remote api.
        """
        self.remote_host = remote_host
        self.access_token = access_token

    def _run_api(
            self,
            circuit: circuits.Circuit,
            param_resolver: Sweepable,
    ) -> str:
        """
        Args:
            circuit: Circuit to be run
            param_resolver: Param resolver for the

        Returns:
            json formatted string of the sequence
        """

        #seq_list: List[Tuple[str, float, List[int]]] = []
        seq_list = []
        circuit = resolve_parameters(circuit, param_resolver)  # type: ignore
        # TODO: Check if circuit is empty
        for op in circuit.all_operations():
            qubits = [obj.x for obj in op.qubits]  # type: ignore
            op_str = get_op_string(op.gate)  # type: ignore
            seq_list.append((op_str, op.gate.exponent, qubits))  # type: ignore
        json_list = json.dumps(seq_list)
        return json_list

    def _send_json(
            self,
            *,
            json_str: str,
            id_str: Union[str, uuid.UUID],
            repetitions: int = 1,
            num_qubits: int = 1):
        """Sends the json string to the remote AQT device
        Args:
            json_str: Json representation of the circuit.
            id_str: Unique id of the datapoint.
            repetitions: Number of repetitions.
            num_qubits: Number of qubits present in the device.

        Returns:
            Measurement results as an array of boolean.
        """
        while True:
            time.sleep(1.0)
            data = put(self.remote_host,
                       data={
                           'data': json_str,
                           'id': id_str,
                           'acccess_token': self.access_token,
                           'repetitions': repetitions,
                           'num_qubits': num_qubits
                       }).json()
            if data['status'] == 'finished':
                break
        measurements_int = data['samples']
        measurements = np.zeros((len(measurements_int), num_qubits))
        for i, result_int in enumerate(measurements_int):
            for j in range(num_qubits):
                measurements[i, j] = np.floor(result_int / 2**j)
        return measurements

    def run_sweep(self,
                  program: Union[circuits.Circuit, schedules.Schedule],
                  params: study.Sweepable,
                  repetitions: int = 1,
                  num_qubits: int = 1) -> List[study.TrialResult]:
        """Samples from the given Circuit or Schedule.

        In contrast to run, this allows for sweeping over different parameter
        values.

        Args:
            program: The circuit or schedule to simulate.
            Should be generated using AQTSampler.generate_circuit_from_list
            params: Parameters to run with the program.
            repetitions: The number of repetitions to simulate.
            num_qubits: The number of qubits in the system.

        The parameters remote_host and access_token are not used.

        Returns:
            TrialResult list for this run; one for each possible parameter
            resolver.
        """
        # TODO: Where should we get the num_qubits??
        # TODO: Probably from the measurement in the circuit!
        meas_name = 'm'  # TODO: Get measurement name from circuit
        circuit = (program if isinstance(program, circuits.Circuit) else
                   program.to_circuit())
        param_resolvers = study.to_resolvers(params)
        trial_results = []  # type: List[study.TrialResult]
        for param_resolver in param_resolvers:
            id_str = uuid.uuid1()
            json_list = self._run_api(circuit=circuit,
                                      param_resolver=param_resolver)
            results = self._send_json(json_str=json_list,
                                      id_str=id_str,
                                      repetitions=repetitions,
                                      num_qubits=num_qubits)
            results = results.astype(bool)
            res_dict = {meas_name: results}
            trial_results.append(
                study.TrialResult(params=param_resolver,
                                  repetitions=repetitions,
                                  measurements=res_dict))
        return trial_results

    def run(self,
            program: Union[circuits.Circuit, schedules.Schedule],
            param_resolver: 'study.ParamResolverOrSimilarType' = None,
            repetitions: int = 1,
            num_qubits: int = 4) -> study.TrialResult:
        """Samples from the given Circuit or Schedule.

        Args:
            program: The circuit or schedule to simulate.
            Should be generated using AQTSampler.generate_circuit_from_list
            param_resolver: Parameters to run with the program.
            repetitions: The number of repetitions to simulate.
            num_qubits: The number of qubits.

        Returns:
            TrialResult for a run.
        """
        return self.run_sweep(program,
                              study.ParamResolver(param_resolver),
                              repetitions=repetitions,
                              num_qubits=num_qubits)[0]


class AQTSamplerSim(AQTSampler):
    """Sampler using the AQT simulator
    When the attribute simulate_ideal is set to True,0
    an ideal circuit is sampled

    If not, the error model defined in aqt_simulator_test.py is used
    Example for running the ideal sampler:

    sampler = AQTSamplerSim()
    sampler.simulate_ideal=True
    """

    def __init__(self,
                 remote_host: str = '',
                 access_token: str = '',
                 simulate_ideal: bool = False):
        """

        Args:
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
        sim = AQTSimulator(num_qubits=num_qubits,
                           simulate_ideal=self.simulate_ideal)
        sim.generate_circuit_from_list(json_str)
        data = sim.simulate_samples(repetitions)
        return data.measurements['m']
