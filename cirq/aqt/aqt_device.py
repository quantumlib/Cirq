import json
from typing import Union, Tuple, List
from cirq import Circuit, Simulator, LineQubit, study, IonDevice
from cirq import measure, X, Y, XX, Duration, depolarize


"""Current device parameters for the AQT/UIBK ion trap device

The device is based on a linear calcium ion string with
arbitrary connectivity. For more information see:

https://quantumoptics.at/en/publications/journal-articles.html

https://iopscience.iop.org/article/10.1088/1367-2630/15/12/123012/meta

The native gate set is local gates: X,Y, and XX entangling gates

"""



gate_dict = {'X': X, 'Y': Y, 'MS': XX}


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
            noise_dict = get_default_noise_dict()
        self.noise_dict = noise_dict
        self.simulate_ideal = simulate_ideal

    def add_noise(self, gate: str, qubits: list, angle: float):
        """Adds a noise operation after a gate including specified crosstalk
        Args:
            gate: Operation where noise should be added
            qubits: List of integers, specifying the qubits
            angle: rotation angle of the operation.
                   Required for crosstalk simulation
        """
        if self.simulate_ideal:
            return None
        for qubit_idx in qubits:
            self.circuit.append(self.noise_dict[gate].on(  # type: ignore
                self.qubit_list[qubit_idx]))
            crosstalk_list = [qubit_idx + 1, qubit_idx - 1]
            for crosstalk_qubit in crosstalk_list:
                try:
                    if crosstalk_qubit >= 0 and gate != 'MS':
                        # TODO: Add MS gate crosstalk
                        xtalk_amp = self.noise_dict['crosstalk']
                        xtalk_op = gate_dict[gate].on(
                            self.qubit_list[crosstalk_qubit]) ** \
                                   (angle * xtalk_amp)  # type: ignore
                        self.circuit.append(xtalk_op)
                except IndexError:
                    pass

    def generate_circuit_from_list(self, json_string: str):
        """Generates a list of cirq operations from a json string
        Args:
            json_string: json that specifies the sequence
        """
        self.circuit = Circuit()
        # TODO add ion device here, is this still required?
        json_obj = json.loads(json_string)
        for gate_list in json_obj:
            gate = gate_list[0]
            angle = gate_list[1]
            qubits = [self.qubit_list[i] for i in gate_list[2]]
            self.circuit.append(gate_dict[gate].on(*qubits)**angle)
            self.add_noise(gate, gate_list[2], angle)
        # TODO: Better solution for measurement at the end
        self.circuit.append(
            measure(*[qubit for qubit in self.qubit_list], key='m'))

    def simulate_samples(self, repetitions: int) -> study.TrialResult:
        """Samples the circuit
        Args:
            repetitions: Number of times the circuit is simulated
        Returns:
            TrialResult from Cirq.Simulator
        """
        if self.circuit == Circuit():
            raise RuntimeError('simulate ideal called without a valid circuit')
        sim = Simulator()
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


def get_default_noise_dict():
    default_noise_dict = {
        'X': depolarize(1e-3),
        'Y': depolarize(1e-3),
        'MS': depolarize(1e-2),
        'crosstalk': 0.03
    }
    return default_noise_dict