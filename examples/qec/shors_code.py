"""
Shor's code is a stabilizer code for quantum error correction.
It uses 9 qubits to encode 1 logic qubit and is resistant
to at least one bit flip and one sign flip or their combination.

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

import cirq
import onequbit_qec as qec


class OneQubitShorsCode(qec.OneQubitCode):

    def __init__(self, qindex: int = 0):
        self.num_physical_qubits = 9
        self.physical_qubits = cirq.GridQubit.rect(1, self.num_physical_qubits,
                                                   qindex, 0)

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
