"""
This is the base class for code for one physical qubit
"""

import cirq


class OneQubitCode:

    def __init__(self):
        self.num_physical_qubits = NotImplemented
        self.physical_qubits = NotImplemented

    def apply_gate(self, gate: cirq.Gate, pos: int):
        if pos > self.num_physical_qubits:
            raise IndexError
        else:
            return gate(self.physical_qubits[pos])

    def encode(self) -> cirq.Circuit:
        return NotImplemented

    def decode(self) -> cirq.Circuit:
        return NotImplemented

    def measure(self):
        for i in range(self.num_physical_qubits):
            yield cirq.measure(self.physical_qubits[i])
