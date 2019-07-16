import cirq

from typing import Sequence
class AQTNoise(cirq.NoiseModel):
    def __init__(self, single_noise_strength: float, cnot_noise_strength: float ):
        if qubit_noise_gate.num_qubits() != 1:
            raise ValueError('noise.num_qubits() != 1')
        self.single_qubit_noise_gate = cirq.depolarize(single_noise_strength)
        self.cnot_qubit_noise_gate = cirq.depolarize(cnot_noise_strength)

    def noisy_moment(self, moment: 'cirq.Moment',
                     system_qubits: Sequence['cirq.Qid']):
        noise_op_list = []
        for operation in moment.operations:
            if operation._gate == cirq.CNOT:
                noise_op = self.cnot_qubit_noise_gate
            else:
                noise_op = self.single_qubit_noise_gate
            noise_op_list += [noise_op for q in operation._qubits]
        return list(moment) + noise_op_list
        #return list(moment) + [self.qubit_noise_gate(q) for q in system_qubits]


q = [cirq.LineQubit(i) for i in range(4)]
p = 0.01
p_cnot = 0.1

qubit_noise_gate = cirq.depolarize(p)
#const_model = cirq.ConstantQubitNoiseModel(qubit_noise_gate)
const_model = AQTNoise(p,p_cnot)

circuit = cirq.Circuit()
circuit.append([cirq.X(q[0])])
circuit.append([cirq.CNOT(q[0],q[1])])
circuit.append([cirq.measure(q0) for q0 in q])

noise_moments = const_model.noisy_moments(circuit, q)
print(circuit, noise_moments)

simulator = cirq.Simulator()
result = simulator.run(circuit)
print('Result of f(0)⊕f(1):')
print(result)