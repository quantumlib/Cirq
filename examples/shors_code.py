""" Shor's code is a stabilizer code for quantum error correction.
It uses 9 qubits to encode 1 logic qubit and is able to correct
at most one bit flip and one sign flip or their combination.

(0, 0): ───@───@───H───@───@───@───@───X───H───@───@───X───M───
           │   │       │   │   │   │   │       │   │   │
(0, 1): ───┼───┼───────X───┼───X───┼───@───────┼───┼───┼───M───
           │   │           │       │   │       │   │   │
(0, 2): ───┼───┼───────────X───────X───@───────┼───┼───┼───M───
           │   │                               │   │   │
(0, 3): ───X───┼───H───@───@───@───@───X───H───X───┼───@───M───
               │       │   │   │   │   │           │   │
(0, 4): ───────┼───────X───┼───X───┼───@───────────┼───┼───M───
               │           │       │   │           │   │
(0, 5): ───────┼───────────X───────X───@───────────┼───┼───M───
               │                                   │   │
(0, 6): ───────X───H───@───@───@───@───X───H───────X───@───M───
                       │   │   │   │   │
(0, 7): ───────────────X───┼───X───┼───@───────────────────M───
                           │       │   │
(0, 8): ───────────────────X───────X───@───────────────────M───
"""

import random

import cirq


class OneQubitShorsCode:

    def __init__(self):
        self.num_physical_qubits = 9
        self.physical_qubits = cirq.LineQubit.range(self.num_physical_qubits)

    def encode(self):
        yield cirq.ops.Moment(
            [cirq.CNOT(self.physical_qubits[0], self.physical_qubits[3])])
        yield cirq.ops.Moment(
            [cirq.CNOT(self.physical_qubits[0], self.physical_qubits[6])])
        yield cirq.ops.Moment([
            cirq.H(self.physical_qubits[0]),
            cirq.H(self.physical_qubits[3]),
            cirq.H(self.physical_qubits[6])
        ])
        yield cirq.ops.Moment([
            cirq.CNOT(self.physical_qubits[0], self.physical_qubits[1]),
            cirq.CNOT(self.physical_qubits[3], self.physical_qubits[4]),
            cirq.CNOT(self.physical_qubits[6], self.physical_qubits[7])
        ])
        yield cirq.ops.Moment([
            cirq.CNOT(self.physical_qubits[0], self.physical_qubits[2]),
            cirq.CNOT(self.physical_qubits[3], self.physical_qubits[5]),
            cirq.CNOT(self.physical_qubits[6], self.physical_qubits[8])
        ])

    def apply_gate(self, gate: cirq.Gate, pos: int):
        if pos > self.num_physical_qubits:
            raise IndexError
        else:
            return gate(self.physical_qubits[pos])

    def correct(self):
        yield cirq.ops.Moment([
            cirq.CNOT(self.physical_qubits[0], self.physical_qubits[1]),
            cirq.CNOT(self.physical_qubits[3], self.physical_qubits[4]),
            cirq.CNOT(self.physical_qubits[6], self.physical_qubits[7])
        ])
        yield cirq.ops.Moment([
            cirq.CNOT(self.physical_qubits[0], self.physical_qubits[2]),
            cirq.CNOT(self.physical_qubits[3], self.physical_qubits[5]),
            cirq.CNOT(self.physical_qubits[6], self.physical_qubits[8])
        ])
        yield cirq.ops.Moment([
            cirq.CCNOT(self.physical_qubits[1], self.physical_qubits[2],
                       self.physical_qubits[0]),
            cirq.CCNOT(self.physical_qubits[4], self.physical_qubits[5],
                       self.physical_qubits[3]),
            cirq.CCNOT(self.physical_qubits[7], self.physical_qubits[8],
                       self.physical_qubits[6])
        ])
        yield cirq.ops.Moment([
            cirq.H(self.physical_qubits[0]),
            cirq.H(self.physical_qubits[3]),
            cirq.H(self.physical_qubits[6])
        ])
        yield cirq.ops.Moment(
            [cirq.CNOT(self.physical_qubits[0], self.physical_qubits[3])])
        yield cirq.ops.Moment(
            [cirq.CNOT(self.physical_qubits[0], self.physical_qubits[6])])
        yield cirq.ops.Moment([
            cirq.CCNOT(self.physical_qubits[3], self.physical_qubits[6],
                       self.physical_qubits[0])
        ])

    def measure(self):
        for i in range(self.num_physical_qubits):
            yield cirq.measure(self.physical_qubits[i])


if __name__ == '__main__':
    # coverage: ignore
    #create circuit
    mycode1 = OneQubitShorsCode()
    my_circuit1 = cirq.Circuit(mycode1.encode())

    #create error
    my_circuit1 += cirq.Circuit(
        mycode1.apply_gate(cirq.X,
                           random.randint(0, mycode1.num_physical_qubits - 1)))

    #correct error
    my_circuit1 += cirq.Circuit(mycode1.correct())

    #decode
    my_circuit1 += cirq.Circuit(mycode1.measure())
    sim1 = cirq.Simulator()
    result1 = sim1.run(my_circuit1, repetitions=1)
    print(result1.measurements['0'])
