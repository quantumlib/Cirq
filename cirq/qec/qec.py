import cirq

# Circuit
#     Measurement
#     Output
class Code:

    def __init__(self):
        self.q0, self.q1, self.q2, self.q3, self.q4, self.q5, self.q6, self.q7, self.q8 = cirq.LineQubit.range(9)

    def decode(self):
        #cirq.CNOT(self.q0, self.q1),
        #cirq.CNOT(self.q3, self.q4),
        # cirq.CNOT(self.q6, self.q7)
        # Can be applied at the same time. Use moment
        #
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
        print("Encoded and decoded ", self.encoded_circuit + self.decoded_circuit)
        return self.encoded_circuit + self.decoded_circuit




    def encode(self):

        self.encoded_circuit = cirq.Circuit(cirq.CNOT(self.q0, self.q3),
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

        return self.encoded_circuit

    def apply_x_error(self):
        self.encoded_circuit.append(cirq.X(self.q5))
        #print(self.encoded_circuit)
        return self.encoded_circuit

    def measure(self):
        measure_circuit = cirq.Circuit(cirq.measure(self.q0), cirq.measure(self.q1),
                                    cirq.measure(self.q2),
                                    cirq.measure(self.q3),
                                    cirq.measure(self.q4),
                                    cirq.measure(self.q5),
                                    cirq.measure(self.q6),
                                    cirq.measure(self.q7),
                                    cirq.measure(self.q8)
                                    )

        self.encoded_circuit = self.decoded_circuit + measure_circuit
        return self.encoded_circuit




mycode = Code()

my_circuit = mycode.encode()

my_circuit = mycode.apply_x_error()


my_circuit = mycode.decode()

my_circuit = mycode.measure()




#print(my_circuit)

    