import cirq

from typing import List

class OneQubitCode:

    def __init__(self):
        self.num_physical_qubits = NotImplemented
        self.physical_qubits = cirq.LineQubit.range(self.num_physical_qubits)

    def apply_gate(self, gate: cirq.Gate, pos: int):
        if pos > self.num_physical_qubits:
            raise IndexError
        yield gate(self.physical_qubits[pos])

    def encode(self) -> cirq.Circuit:
        return NotImplemented

    def decode(self) -> cirq.Circuit:
        return NotImplemented

    def measure(self):
        for i in range(self.num_physical_qubits):
            yield cirq.measure(self.physical_qubits[i])


class MultiQubitCode:
    def __init__(self, input: List["cirq.Qid"], codetype: OneQubitCode):
        self.logical_qubits = {}
        self.encoded_circuit = {}
        self.currentcircuit = cirq.Circuit()
        for logical_qubit in input:
            self.logical_qubits[logical_qubit] = codetype()
        self.physical_to_logical_ratio = self.logical_qubits[input[0]].num_physical_qubits

    def encode(self):
        for logical_qubit in self.logical_qubits.values():
            self.encoded_circuit[logical_qubit] = cirq.Circuit(logical_qubit.encode())
            self.currentcircuit += self.encoded_circuit[logical_qubit]
        return self.currentcircuit

    def operation(self, original_circuit: cirq.Circuit):
        for op in original_circuit.all_operations:
            op_on_physical_qubits = ops.apply_on_physical_qubits(
                op, self.logical_qubits, self.physical_to_logical_ratio)
            self.currentcircuit.append(cirq.ops.Moment(op_on_physical_qubits))
        return self.currentcircuit

    def decode(self):
        for logical_qubit in self.logical_qubits.values():
            self.currentcircuit.append(cirq.Circuit(logical_qubit.decode()))
        return self.currentcircuit

    def measure(self):
        for logical_qubit in self.logical_qubits.values():
            self.currentcircuit.append(cirq.Circuit(logical_qubit.measure()))
        return self.currentcircuit

