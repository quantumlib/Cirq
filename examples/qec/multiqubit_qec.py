"""
This is code for multiple physical qubits
"""
import random
from typing import List

import cirq
from fault_tolerate_operations import apply_on_physical_qubits
from onequbit_qec import OneQubitCode


class MultiQubitCode:

    def __init__(self, qinput: List["cirq.Qid"], codetype: OneQubitCode):
        self.logical_qubits = {}
        self.encoded_circuit = {}
        self.currentcircuit = cirq.Circuit()
        self.codetype = codetype
        count = 0
        for logical_qubit in qinput:
            self.logical_qubits[logical_qubit] = codetype(count)
            count += 1
        self.physical_to_logical_ratio = self.logical_qubits[
            qinput[0]].num_physical_qubits

    def encode(self):
        for logical_qubit in self.logical_qubits.values():
            self.encoded_circuit[logical_qubit] = cirq.Circuit(
                logical_qubit.encode())
            self.currentcircuit += self.encoded_circuit[logical_qubit]
        return self.currentcircuit

    def apply_error(self):
        logical_qubit_chosen = random.sample(self.logical_qubits.keys(), 1)[0]
        physical_qubit_chosen = random.sample(
            self.logical_qubits[logical_qubit_chosen].physical_qubits, 1)[0]
        self.currentcircuit.append(cirq.X(physical_qubit_chosen))
        return self.currentcircuit

    def operation(self, original_circuit: cirq.Circuit):
        for op in original_circuit.all_operations():
            op_on_physical_qubits = apply_on_physical_qubits(
                op, self.logical_qubits, self.physical_to_logical_ratio,
                self.codetype)
            self.currentcircuit.append(op_on_physical_qubits)
        return self.currentcircuit

    def correct(self):
        for logical_qubit in self.logical_qubits.values():
            self.currentcircuit.append(cirq.Circuit(logical_qubit.correct()))
        return self.currentcircuit

    def measure(self):
        for logical_qubit in self.logical_qubits.values():
            self.currentcircuit.append(cirq.Circuit(logical_qubit.measure()))
        return self.currentcircuit