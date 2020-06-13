import cirq
from typing import List
import fault_tolerate_operations as ops
from onequbit_qec import OneQubitCode

class MultiQubitCode:
    def __init__(self, qinput: List["cirq.Qid"], codetype: OneQubitCode):
        self.logical_qubits = {}
        self.encoded_circuit = {}
        self.currentcircuit = cirq.Circuit()
        count = 0
        for logical_qubit in qinput:
            self.logical_qubits[logical_qubit] = codetype(count)
            count += 1
        self.physical_to_logical_ratio = self.logical_qubits[qinput[0]].num_physical_qubits

    def encode(self):
        for logical_qubit in self.logical_qubits.values():
            self.encoded_circuit[logical_qubit] = cirq.Circuit(logical_qubit.encode())
            self.currentcircuit += self.encoded_circuit[logical_qubit]
        return self.currentcircuit

    def operation(self, original_circuit: cirq.Circuit):
        for op in original_circuit.all_operations():
            op_on_physical_qubits = ops.apply_on_physical_qubits(
                op, self.logical_qubits, self.physical_to_logical_ratio)
            self.currentcircuit.append(op_on_physical_qubits)
        return self.currentcircuit

    def decode(self):
        for logical_qubit in self.logical_qubits.values():
            self.currentcircuit.append(cirq.Circuit(logical_qubit.decode()))
        return self.currentcircuit

    def measure(self):
        for logical_qubit in self.logical_qubits.values():
            self.currentcircuit.append(cirq.Circuit(logical_qubit.measure()))
        return self.currentcircuit