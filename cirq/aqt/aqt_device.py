"""Current device parameters for the AQT/UIBK ion trap device

The device is based on a linear calcium ion string with
arbitrary connectivity. For more information see:

https://quantumoptics.at/en/publications/journal-articles.html

https://iopscience.iop.org/article/10.1088/1367-2630/15/12/123012/meta

The native gate set consists of the local gates: X,Y, and XX entangling gates

"""
import json
from typing import Union, Tuple, List, Sequence
from cirq import ops, devices, study
from cirq import Circuit, LineQubit, IonDevice, Duration
from cirq import DensityMatrixSimulator, Qid, Moment

gate_dict = {'X': ops.X, 'Y': ops.Y, 'MS': ops.XX}


def get_op_string(op_obj: ops.Operation):
    """Find the string representation for a given gate
    Params:
        op_obj: Gate object, out of: XXPowGate, XPowGate, YPowGate"""
    if isinstance(op_obj, ops.XXPowGate) or ops.op_gate_of_type(
            op_obj, ops.XXPowGate):
        op_str = 'MS'
    elif isinstance(op_obj, ops.XPowGate) or ops.op_gate_of_type(
            op_obj, ops.XPowGate):
        op_str = 'X'
    elif isinstance(op_obj, ops.YPowGate) or ops.op_gate_of_type(
            op_obj, ops.YPowGate):
        op_str = 'Y'
    elif isinstance(op_obj, ops.MeasurementGate) or ops.op_gate_of_type(
            op_obj, ops.MeasurementGate):
        op_str = 'Meas'
    else:
        raise ValueError('Got unknown gate:', op_obj)
    return op_str


class AQTNoiseModel(devices.NoiseModel):
    def __init__(self, single_qubit_p=0.01, ms_p=0.03):
        self.single_qubit_p = single_qubit_p
        self.ms_p = ms_p

    def noisy_moment(self, moment: 'Moment', system_qubits: Sequence['Qid']):
        noise_list = []
        #TODO: check whether this works with multiple
        # operations in a single moment.
        for op in moment.operations:
            op_str = get_op_string(op)
            if op_str in ['X', 'Y']:
                noise_op = ops.depolarize(self.single_qubit_p)
            elif op_str == 'MS':
                noise_op = ops.depolarize(self.ms_p)
            else:
                break
            for qubit in system_qubits:
                if qubit in op.qubits:
                    noise_list.append(noise_op.on(qubit))
        return list(moment) + noise_list


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
            #self.add_noise(gate, gate_list[2], angle)
        # TODO: Better solution for measurement at the end
        self.circuit.append(
            ops.measure(*[qubit for qubit in self.qubit_list], key='m'))

    def simulate_samples(self, repetitions: int) -> study.TrialResult:
        """Samples the circuit
        Args:
            repetitions: Number of times the circuit is simulated
        Returns:
            TrialResult from Cirq.Simulator
        """
        if self.simulate_ideal:
            noise_model = devices.NO_NOISE
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


def get_default_noise_dict():
    default_noise_dict = {
        'X': ops.depolarize(1e-3),
        'Y': ops.depolarize(1e-3),
        'MS': ops.depolarize(1e-2),
        'crosstalk': 0.03
    }
    return default_noise_dict
