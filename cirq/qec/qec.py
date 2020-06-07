#This is demonstration of Shor's code

import cirq

class Code:

    def __init__(self):
        self.qubit = list(cirq.LineQubit.range(9))

    def encode(self):

        yield cirq.CNOT(self.qubits[0], self.qubits[3])
        yield cirq.CNOT(self.qubits[0], self.qubits[6])
        yield cirq.H(self.qubit[0])
        yield cirq.H(self.qubit[3])
        yield cirq.H(self.qubit[6])
        yield cirq.CNOT(self.qubit[0], self.qubit[1])
        yield cirq.CNOT(self.qubit[3], self.qubit[4])
        yield cirq.CNOT(self.qubit[6], self.qubit[7])
        yield cirq.CNOT(self.qubit[0], self.qubit[2])
        yield cirq.CNOT(self.qubit[3], self.qubit[5])
        yield cirq.CNOT(self.qubit[6], self.qubit[8])
        self.current_circuit = cirq.Circuit(cirq.CNOT(self.q0, self.q3),
                                            cirq.CNOT(self.q0, self.q6),
                                            cirq.H(self.q0),
                                            cirq.H(self.q3),
                                            cirq.H(self.q6),
                                            cirq.CNOT(self.q0, self.q1),
                                            cirq.CNOT(self.q3, self.q4),
                                            cirq.CNOT(self.q6, self.q7),
                                            cirq.CNOT(self.q0, self.q2),
                                            cirq.CNOT(self.q3, self.q5),
                                            cirq.CNOT(self.q6, self.q8))

        return self.current_circuit

    def decode(self):
        yield cirq.CNOT(self.qubit[0], self.qubit[1])
        yield cirq.CNOT(self.qubit[3], self.qubit[4])
        yield cirq.CNOT(self.qubit[6], self.qubit[7])
        yield cirq.CNOT(self.qubit[0], self.qubit[2])
        yield cirq.CNOT(self.qubit[3], self.qubit[5])
        yield cirq.CNOT(self.qubit[6], self.qubit[8])
        yield cirq.CCNOT(self.qubit[1], self.qubit[2], self.qubit[0])
        yield cirq.CCNOT(self.qubit[4], self.qubit[5], self.qubit[3])
        yield cirq.CCNOT(self.qubit[7], self.qubit[8], self.qubit[6])
        yield cirq.H(self.qubit[0])
        yield cirq.H(self.qubit[3])
        yield cirq.H(self.qubit[6])
        yield cirq.CNOT(self.qubit[0], self.qubit[3])
        yield cirq.CNOT(self.qubit[0], self.qubit[6])
        yield cirq.CCNOT(self.qubit[3], self.qubit[6], self.qubit[0])
        self.decoded_circuit = cirq.Circuit(cirq.CNOT(self.q0, self.q1),
                                            cirq.CNOT(self.q3, self.q4),
                                            cirq.CNOT(self.q6, self.q7),
                                            cirq.CNOT(self.q0, self.q2),
                                            cirq.CNOT(self.q3, self.q5),
                                            cirq.CNOT(self.q6, self.q8),
                                            cirq.CCNOT(self.q1, self.q2, self.q0),
                                            cirq.CCNOT(self.q4, self.q5, self.q3),
                                            cirq.CCNOT(self.q7, self.q8, self.q6),
                                            cirq.H(self.q0),
                                            cirq.H(self.q3),
                                            cirq.H(self.q6),
                                            cirq.CNOT(self.q0, self.q3),
                                            cirq.CNOT(self.q0, self.q6),
                                            cirq.CCNOT(self.q3, self.q6, self.q0))
        self.current_circuit = self.current_circuit + self.decoded_circuit
        return self.current_circuit


    def apply_x_error(self):
        self.current_circuit.append(cirq.Z(self.q5))
        #print(self.encoded_circuit)
        return self.current_circuit

    def measure(self):
        yield cirq.measure(self.qubit[0], self.qubit[1])
        yield cirq.measure(self.qubit[2])
        yield cirq.measure(self.qubit[3])
        yield cirq.measure(self.qubit[4])
        yield cirq.measure(self.qubit[5])
        yield cirq.measure(self.qubit[6])
        yield cirq.measure(self.qubit[7])
        yield cirq.measure(self.qubit[8])
        measure_circuit = cirq.Circuit(cirq.measure(self.q0), cirq.measure(self.q1),
                                    cirq.measure(self.q2),
                                    cirq.measure(self.q3),
                                    cirq.measure(self.q4),
                                    cirq.measure(self.q5),
                                    cirq.measure(self.q6),
                                    cirq.measure(self.q7),
                                    cirq.measure(self.q8)
                                    )

        self.current_circuit = self.current_circuit + measure_circuit
        return self.current_circuit


mycode = Code()

my_circuit = mycode.encode()

my_circuit = mycode.apply_x_error()

my_circuit = mycode.decode()

my_circuit = mycode.measure()

sim1 = cirq.Simulator()
result = sim1.run(my_circuit, repetitions=20)

print(my_circuit)
print(result)