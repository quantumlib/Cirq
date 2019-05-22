import uuid
import time
import json
from requests import put #TODO: Specify as a dependency
from typing import cast, Iterable, List, Union
import numpy as np

from cirq.aqt.aqt_simulator import AQTSimulator

from cirq import circuits, schedules, study, Sampler, resolve_parameters
from cirq import XPowGate,YPowGate,XXPowGate
from cirq.study.resolver import ParamResolver
from cirq.study.sweeps import Sweep


Sweepable = Union[
    ParamResolver, Iterable[ParamResolver], Sweep, Iterable[Sweep]]


"""Samplers for the AQT ion trap device"""

class AQTSampler(Sampler):
    """Sampler for the AQT ion trap device
    This sampler connects to the AQT machine and runs a single circuit or an entire sweep remotely
    """

    def _run_api(self,
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
        op_dict = {}
        op_dict[XXPowGate] = 'MS'
        op_dict[XPowGate] = 'X'
        op_dict[YPowGate] = 'Y'
        seq_list = []
        circuit = resolve_parameters(circuit,param_resolver)
        #TODO: Check if circuit is empty
        for op in circuit.all_operations():
            qubits = [obj.x for obj in op.qubits]
            for key in op_dict:
                if isinstance(op.gate, key):
                    seq_list.append([op_dict[key], op.gate.exponent, qubits])
        json_list = json.dumps(seq_list)
        return json_list

    def _send_json(self,
                   json_str: str,
                   id: Union[str,uuid.UUID],
                   remote_host: str = 'http://localhost:5000',
                   access_token: str = '',
                   repetitions: int = 1,
                   no_qubit: int = 1,
                   ):
        """Sends the json string to the remote AQT device
        Args:
            json_str: json representation of the circuit
            id: Unique id of the datapoint
            remote_host: address of the remote device
            repetitions: Number of repetitions
            no_qubit: Number of qubits present in the device

        Returns:
            measurement results as an array of boolean
        """
        data = put(remote_host, data={'data': json_str, 'id':id,'repetitions':repetitions,
                                      'no_qubits':no_qubit}).json()
        while data['status'] != 'finished':
            time.sleep(1.0)
            data = put(remote_host, data={'data': json_str, 'id': id, 'acccess_token':access_token,
                                          'repetitions':repetitions,'no_qubits':no_qubit}).json()
        measurements_int = data['samples']
        measurements = np.zeros((len(measurements_int), no_qubit))
        for i, result_int in enumerate(measurements_int):
            for j in range(no_qubit):
                # print(i,j,bin(result_int)[-j],np.floor(result_int/2**j))
                measurements[i, j] = np.floor(result_int / 2 ** j)
        # TODO: Check Big endian/little endian encoding!
        return measurements

    def run_sweep(
            self,
            program: Union[circuits.Circuit, schedules.Schedule],
            params: study.Sweepable,
            repetitions: int = 1,
            no_qubit: int = 1,
            remote_host: str = 'http://localhost:5000',
            access_token: str = ''
    ) -> List[study.TrialResult]:
        """Samples from the given Circuit or Schedule.

        In contrast to run, this allows for sweeping over different parameter
        values.

        Args:
            program: The circuit or schedule to simulate. Should be generated using AQTSampler.generate_circuit_from_list
            params: Parameters to run with the program.
            repetitions: The number of repetitions to simulate.
            no_qubit: The number of qubits in the system
            remote_host: address of the remote device
            access_token: access token for the remote api

        Returns:
            TrialResult list for this run; one for each possible parameter
            resolver.
        """
        # TODO: Where should we get the no_qubits?? Probably from the measurement in the circuit!
        meas_name = 'm' # TODO: Get measurement name from circuit
        circuit = (program if isinstance(program, circuits.Circuit)
        else program.to_circuit())
        param_resolvers = study.to_resolvers(params)
        trial_results = []  # type: List[study.TrialResult]
        for param_resolver in param_resolvers:
            id = uuid.uuid1()
            json_list = self._run_api(circuit=circuit,
                                 param_resolver=param_resolver)
            results = self._send_json(json_list, id, repetitions=repetitions, no_qubit=no_qubit,
                                      remote_host=remote_host, access_token=access_token)
            trial_results.append(study.TrialResult(params=param_resolver,
                                               repetitions=repetitions,
                                               measurements={meas_name: results}))
        return trial_results

    def run(
            self,
            program: Union[circuits.Circuit, schedules.Schedule],
            param_resolver: 'study.ParamResolverOrSimilarType' = None,
            repetitions: int = 1,
            no_qubit: int = 4,
            remote_host: str = 'http://localhost:5000',
    ) -> study.TrialResult:
        """Samples from the given Circuit or Schedule.

        Args:
            program: The circuit or schedule to simulate. Should be generated using AQTSampler.generate_circuit_from_list
            param_resolver: Parameters to run with the program.
            repetitions: The number of repetitions to simulate.
            no_qubit: The number of qubits

        Returns:
            TrialResult for a run.
        """
        return self.run_sweep(program, study.ParamResolver(param_resolver),
                              repetitions=repetitions, no_qubit=no_qubit, remote_host=remote_host)[0]

class AQTSamplerSim(AQTSampler):
    """Sampler using the AQT simulator
    when the attribute simulate_ideal is set to True, an ideal circuit is sampled
    If not, the error model defined in aqt_simulator_test.py is used
    Example for running the ideal sampler:

    sampler = AQTSamplerSim()
    sampler.simulate_ideal=True
    """



    def _send_json(self,
                   json_str: str,
                   id: Union[str,uuid.UUID],
                   remote_host: str = 'http://localhost:5000',
                   access_token: str = '',
                   repetitions: int = 1,
                   no_qubit: int = 1,
                   ):
        """Replaces the remote host with a local imulator
        Args:
            json_str: json representation of the circuit
            id: Unique id of the datapoint
            remote_host: address of the remote device
            repetitions: Number of repetitions
            no_qubit: Number of qubits present in the device

        Returns:
            measurement results as an array of boolean
        """
        try:
            simulate_ideal = self.simulate_ideal
        except AttributeError:
            simulate_ideal = False
        sim = AQTSimulator(no_qubit=no_qubit, simulate_ideal=simulate_ideal)
        sim.generate_circuit_from_list(json_str)
        data = sim.simulate_samples(repetitions)
        return data.measurements['m']
