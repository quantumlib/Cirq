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
"""Current device parameters for the AQT/UIBK ion trap device

The device is based on a linear calcium ion string with
arbitrary connectivity. For more information see:

https://quantumoptics.at/en/publications/journal-articles.html

https://iopscience.iop.org/article/10.1088/1367-2630/15/12/123012/meta

The native gate set consists of the local gates: X,Y, and XX entangling gates

"""
import json
from typing import Union, Tuple, List, Dict, Sequence, Any, cast
import numpy as np

from cirq.ops import (MeasurementGate, X, Y, Z, XX, ZPowGate, XPowGate,
                      YPowGate, XXPowGate, Moment, Operation, Qid, depolarize,
                      measure)
from cirq import Circuit, Duration, DensityMatrixSimulator
from cirq.devices import LineQubit, NoiseModel, NO_NOISE
from cirq.ion import IonDevice
from cirq.study import TrialResult

# A global dictionary used for de/serialising the circuits from/to the
# remote server, as well as for determining the crosstalk per gate
AQT_GATE_DICT = {'X': X, 'Y': Y, 'Z': Z, 'MS': XX}


class AQTNoiseModel(NoiseModel):
    """A noise model for the AQT ion trap """

    def __init__(self):
        self.noise_op_dict = get_aqt_default_noise_dict()

    def noisy_moment(self, moment: Moment,
                     system_qubits: Sequence[Qid]) -> List[Operation]:
        """Returns a list of noisy moments.

        The model includes
        - Depolarizing noise with gate-dependent strength
        - Crosstalk  between neighboring qubits

        Args:
            moment: ideal moment
            system_qubits: List of qubits

        Returns:
            List of ideal and noisy moments
        """
        noise_list = []
        for op in moment.operations:
            op_str = _get_aqt_device_op_string(op)
            try:
                noise_op = self.noise_op_dict[op_str]
            except KeyError:
                break
            for qubit in op.qubits:
                noise_list.append(noise_op.on(qubit))
            noise_list += self.get_crosstalk_operation(op, system_qubits)
        return list(moment) + noise_list

    def get_crosstalk_operation(self, operation: Operation,
                                system_qubits: Sequence[Qid]
                               ) -> List[Operation]:
        """Returns a list of operations including crosstalk

        Args:
            operation: Ideal operation
            system_qubits: Tuple of line qubits

        Returns:
            List of operations including crosstalk
        """
        cast(Tuple[LineQubit], system_qubits)
        num_qubits = len(system_qubits)
        xtlk_arr = np.zeros(num_qubits)
        idx_list = []
        for qubit in operation.qubits:
            idx = system_qubits.index(qubit)
            idx_list.append(idx)
            neighbors = [idx - 1, idx + 1]
            for neigh_idx in neighbors:
                if neigh_idx >= 0 and neigh_idx < num_qubits:
                    xtlk_arr[neigh_idx] = self.noise_op_dict['crosstalk']
        for idx in idx_list:
            xtlk_arr[idx] = 0
        xtlk_op_list = []
        op_str = _get_aqt_device_op_string(operation)
        if len(operation.qubits) == 1:
            for idx in xtlk_arr.nonzero()[0]:
                exponent = operation.gate.exponent  #type:ignore
                exponent = exponent * xtlk_arr[idx]
                xtlk_op = AQT_GATE_DICT[op_str].on(system_qubits[idx])**exponent
                xtlk_op_list.append(xtlk_op)
        elif len(operation.qubits) == 2:
            for op_qubit in operation.qubits:
                for idx in xtlk_arr.nonzero()[0]:
                    exponent = operation.gate.exponent  # type:ignore
                    exponent = exponent * xtlk_arr[idx]
                    xtlk_op = AQT_GATE_DICT[op_str].on(
                        op_qubit, system_qubits[idx])**exponent
                    xtlk_op_list.append(xtlk_op)
        return xtlk_op_list


class AQTSimulator:
    """A simulator for the AQT device."""

    def __init__(self,
                 num_qubits: int,
                 circuit: Circuit = Circuit(),
                 simulate_ideal: bool = False,
                 noise_dict: Union[dict, None] = None):
        """Initializes the AQT simulator

        Args:
            num_qubits: Number of qubits
            circuit: Optional, circuit to be simulated.
            Last moment needs to be a measurement over all qubits with key 'm'
            simulate_ideal: If True, an ideal circuit will be simulated
        """
        self.circuit = circuit
        self.num_qubits = num_qubits
        self.qubit_list = LineQubit.range(num_qubits)
        if noise_dict is None:
            noise_dict = get_aqt_default_noise_dict()
        self.noise_dict = noise_dict
        self.simulate_ideal = simulate_ideal

    def generate_circuit_from_list(self, json_string: str):
        """Generates a list of cirq operations from a json string.

        The default behavior is to add a measurement to any qubit at the end
        of the circuit as there are no measurements defined in the AQT API.

        Args:
            json_string: json that specifies the sequence
        """
        self.circuit = Circuit()
        json_obj = json.loads(json_string)
        for gate_list in json_obj:
            gate = gate_list[0]
            angle = gate_list[1]
            qubits = [self.qubit_list[i] for i in gate_list[2]]
            self.circuit.append(AQT_GATE_DICT[gate].on(*qubits)**angle)
        # TODO: Better solution for measurement at the end. Issue #2199
        self.circuit.append(
            measure(*[qubit for qubit in self.qubit_list], key='m'))

    def simulate_samples(self, repetitions: int) -> TrialResult:
        """Samples the circuit

        Args:
            repetitions: Number of times the circuit is simulated

        Returns:
            TrialResult from Cirq.Simulator
        """
        if self.simulate_ideal:
            noise_model = NO_NOISE
        else:
            noise_model = AQTNoiseModel()
        if self.circuit == Circuit():
            raise RuntimeError('simulate ideal called without a valid circuit')
        sim = DensityMatrixSimulator(noise=noise_model)
        result = sim.run(self.circuit, repetitions=repetitions)
        return result


def get_aqt_device(num_qubits: int) -> Tuple[IonDevice, List[LineQubit]]:
    """Returns an AQT ion device

    Args:
        num_qubits: number of qubits

    Returns:
         IonDevice, qubit_list
    """
    qubit_list = LineQubit.range(num_qubits)
    us = 1000 * Duration(nanos=1)
    ion_device = IonDevice(measurement_duration=100 * us,
                           twoq_gates_duration=200 * us,
                           oneq_gates_duration=10 * us,
                           qubits=qubit_list)
    return ion_device, qubit_list


def get_aqt_default_noise_dict() -> Dict[str, Any]:
    """Returns the current noise parameters"""
    default_noise_dict = {
        'X': depolarize(1e-3),
        'Y': depolarize(1e-3),
        'Z': depolarize(1e-3),
        'MS': depolarize(1e-2),
        'crosstalk': 0.03
    }
    return default_noise_dict


def _get_aqt_device_op_string(op_obj: Operation) -> str:
    """Find the string representation for a given gate

    Args:
        op_obj: Gate object, one of: XXPowGate, XPowGate, YPowGate, ZPowGate
        or MeasurementGate

    Returns:
        String representing the gate operations
    """
    if isinstance(op_obj, XXPowGate) or isinstance(op_obj.gate, XXPowGate):
        op_str = 'MS'
    elif isinstance(op_obj, XPowGate) or isinstance(op_obj.gate, XPowGate):
        op_str = 'X'
    elif isinstance(op_obj, YPowGate) or isinstance(op_obj.gate, YPowGate):
        op_str = 'Y'
    elif isinstance(op_obj, ZPowGate) or isinstance(op_obj.gate, ZPowGate):
        op_str = 'Z'
    elif isinstance(op_obj, MeasurementGate) or isinstance(
            op_obj.gate, MeasurementGate):
        op_str = 'Meas'
    else:
        raise ValueError('Got unknown gate:', op_obj)
    return op_str
