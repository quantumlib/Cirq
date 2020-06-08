#This is demonstration of Shor's code

import cirq
import qec

class OneQubitShorsCode(qec.OneQubitCode):

    def __init__(self):
        self.num_physical_qubits = 9
        self.physical_qubits = list(cirq.LineQubit.range(self.num_physical_qubits))

    def encode(self):
        yield cirq.ops.Moment([cirq.CNOT(self.physical_qubits[0], self.physical_qubits[3])])
        yield cirq.ops.Moment([cirq.CNOT(self.physical_qubits[0], self.physical_qubits[6])])
        yield cirq.ops.Moment([cirq.H(self.physical_qubits[0]),
                              cirq.H(self.physical_qubits[3]),
                              cirq.H(self.physical_qubits[6])])
        yield cirq.ops.Moment([cirq.CNOT(self.physical_qubits[0], self.physical_qubits[1]),
                              cirq.CNOT(self.physical_qubits[3], self.physical_qubits[4]),
                              cirq.CNOT(self.physical_qubits[6], self.physical_qubits[7])])
        yield cirq.ops.Moment([cirq.CNOT(self.physical_qubits[0], self.physical_qubits[2]),
                              cirq.CNOT(self.physical_qubits[3], self.physical_qubits[5]),
                              cirq.CNOT(self.physical_qubits[6], self.physical_qubits[8])])

    def decode(self):
        yield cirq.ops.Moment(
            [cirq.CNOT(self.physical_qubits[0], self.physical_qubits[1]),
             cirq.CNOT(self.physical_qubits[3], self.physical_qubits[4]),
             cirq.CNOT(self.physical_qubits[6], self.physical_qubits[7])])
        yield cirq.ops.Moment(
            [cirq.CNOT(self.physical_qubits[0], self.physical_qubits[2]),
             cirq.CNOT(self.physical_qubits[3], self.physical_qubits[5]),
             cirq.CNOT(self.physical_qubits[6], self.physical_qubits[8])])
        yield cirq.ops.Moment(
            [cirq.CCNOT(self.physical_qubits[1], self.physical_qubits[2], self.physical_qubits[0]),
             cirq.CCNOT(self.physical_qubits[4], self.physical_qubits[5], self.physical_qubits[3]),
             cirq.CCNOT(self.physical_qubits[7], self.physical_qubits[8], self.physical_qubits[6])])
        yield cirq.ops.Moment(
            [cirq.H(self.physical_qubits[0]),
             cirq.H(self.physical_qubits[3]),
             cirq.H(self.physical_qubits[6])])
        yield cirq.ops.Moment([cirq.CNOT(self.physical_qubits[0], self.physical_qubits[3])])
        yield cirq.ops.Moment([cirq.CNOT(self.physical_qubits[0], self.physical_qubits[6])])
        yield cirq.ops.Moment([cirq.CCNOT(self.physical_qubits[3], self.physical_qubits[6], self.physical_qubits[0])])

#    def measure(self):
#        return cirq.ops.Moment(cirq.measure(self.physical_qubits[0]),
#              cirq.measure(self.physical_qubits[1]),
#              cirq.measure(self.physical_qubits[2]),
#              cirq.measure(self.physical_qubits[3]),
#              cirq.measure(self.physical_qubits[4]),
#              cirq.measure(self.physical_qubits[5]),
#              cirq.measure(self.physical_qubits[6]),
#              cirq.measure(self.physical_qubits[7]),
#              cirq.measure(self.physical_qubits[8]))




