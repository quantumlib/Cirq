import json

from cirq import Circuit, Simulator, LineQubit, study
from cirq import measure, X, Y, XX
from cirq.aqt.aqt_device import default_noise_dict
"""Simulator for the AQT ion trap device"""

gate_dict = {}
gate_dict['X'] = X
gate_dict['Y'] = Y
gate_dict['MS'] = XX


class AQTSimulator:
    """A simulator for the AQT device."""
    def __init__(self,
                 no_qubit: int,
                 circuit: Circuit = Circuit(),
                 simulate_ideal: bool = False,
                 noise_dict: dict = {}):
        """Initializes the AQT simulator
        Args:
            no_qubit: Number of qubits
            circuit: Optional, circuit to be simulated.
            Last moment needs to be a measurement over all qubits with key 'm'
            simulate_ideal: If True, an ideal circuit will be simulated
        """
        self.circuit = circuit
        self.no_qubit = no_qubit
        self.qubit_list = LineQubit.range(no_qubit)
        if noise_dict == {}:
            noise_dict = default_noise_dict
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
        if self.simulate_ideal == True:
            return None
        for qubit_idx in qubits:
            self.circuit.append(self.noise_dict[gate].on(
                self.qubit_list[qubit_idx]))
            crosstalk_list = [qubit_idx + 1, qubit_idx - 1]
            for crosstalk_qubit in crosstalk_list:
                try:
                    if crosstalk_qubit >= 0 and gate != 'MS':
                        # TODO: Add MS gate crosstalk
                        xtalk_op = gate_dict[gate].on(
                            self.qubit_list[crosstalk_qubit]) ** \
                                   (angle * self.noise_dict['crosstalk'])
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
