#This is demonstration of Shor's code

import cirq
import cirq.qec.fault_tolerate_operations as ops

class OneQubitCode:

    def __init__(self, x = 0):
        self.num_physical_qubits = 9
        self.physical_qubits = list(cirq.LineQubit.range(9))
        self.physical_qubits[0] = x

    def encode(self):
        yield cirq.ops.Moment(cirq.CNOT(self.physical_qubits[0], self.physical_qubits[3]),
                              cirq.CNOT(self.physical_qubits[0], self.physical_qubits[6]))
        yield cirq.ops.Moment(cirq.H(self.physical_qubits[0]),
                              cirq.H(self.physical_qubits[3]),
                              cirq.H(self.physical_qubits[6]))
        yield cirq.ops.Moment(cirq.CNOT(self.physical_qubits[0], self.physical_qubits[1]),
                              cirq.CNOT(self.physical_qubits[3], self.physical_qubits[4]),
                              cirq.CNOT(self.physical_qubits[6], self.physical_qubits[7]))
        yield cirq.ops.Moment(cirq.CNOT(self.physical_qubits[0], self.physical_qubits[2]),
                              cirq.CNOT(self.physical_qubits[3], self.physical_qubits[5]),
                              cirq.CNOT(self.physical_qubits[6], self.physical_qubits[8]))

    def decode(self):
        yield cirq.ops.Moment(cirq.CNOT(self.physical_qubits[0], self.physical_qubits[1]),
              cirq.CNOT(self.physical_qubits[3], self.physical_qubits[4]),
              cirq.CNOT(self.physical_qubits[6], self.physical_qubits[7]))
        yield cirq.ops.Moment(cirq.CNOT(self.physical_qubits[0], self.physical_qubits[2]),
              cirq.CNOT(self.physical_qubits[3], self.physical_qubits[5]),
              cirq.CNOT(self.physical_qubits[6], self.physical_qubits[8]))
        yield cirq.ops.Moment(cirq.CCNOT(self.physical_qubits[1], self.physical_qubits[2], self.physical_qubits[0]),
              cirq.CCNOT(self.physical_qubits[4], self.physical_qubits[5], self.physical_qubits[3]),
              cirq.CCNOT(self.physical_qubits[7], self.physical_qubits[8], self.physical_qubits[6]))
        yield cirq.ops.Moment((cirq.H(self.physical_qubits[0]),
              cirq.H(self.physical_qubits[3]),
              cirq.H(self.physical_qubits[6]))
        yield cirq.ops.Moment(cirq.CNOT(self.physical_qubits[0], self.physical_qubits[3]),
              cirq.CNOT(self.physical_qubits[0], self.physical_qubits[6]),
              cirq.CCNOT(self.physical_qubits[3], self.physical_qubits[6], self.physical_qubits[0]))

    def measure(self):
        yield cirq.ops.Moment(cirq.measure(self.physical_qubits[0]),
              cirq.measure(self.physical_qubits[1]),
              cirq.measure(self.physical_qubits[2]),
              cirq.measure(self.physical_qubits[3]),
              cirq.measure(self.physical_qubits[4]),
              cirq.measure(self.physical_qubits[5]),
              cirq.measure(self.physical_qubits[6]),
              cirq.measure(self.physical_qubits[7]),
              cirq.measure(self.physical_qubits[8]))


class MultiQubitCode:
    def __init__(self, input: list["cirq.Qid"]):
        self.logical_qubits = {}
        self.encoded_circuit = {}
        for logical_qubit in input:
            self.logical_qubits[logical_qubit] = OneQubitCode(logical_qubit)
        self.physical_to_logical_ratio = self.logical_qubits[input[0]].num_physical_qubits

    def encode(self):
        for logical_qubit in self.logical_qubits.values:
            self.encoded_circuit[logical_qubit] = cirq.Circuit(logical_qubit.encode())

    def operation(self, original_circuit: cirq.Circuit):
        for op in original_circuit.all_operations:
            ops.apply_on_physical_qubits(op, )
            op_on_physical_qubits = []
            for i in range(self.physical_to_logical_ratio):
                op_on_physical_qubits.append(op.with_qubits(
                    self.encoded_circuit[lq].physical_qubits[i] for lq in op.qubits))
            self.circuit.append(cirq.ops.Moment(op_on_physical_qubits))

    def decode(self):
        for qubit in self.logical_qubits.values:
            self.circuit = cirq.Circuit(qubit.decode())

    def measure(self):
        for qubit in self.logical_qubits.values:
            self.circuit = cirq.Circuit(qubit.measure())


mycode = Code()

my_circuit = cirq.Circuit(mycode.encode())

my_circuit = mycode.apply_x_error()

my_circuit = cirq.Circuit(mycode.decode())

my_circuit = cirq.Circuit(mycode.measure())

sim1 = cirq.Simulator()
result = sim1.run(my_circuit, repetitions=20)

print(my_circuit)
print(result)