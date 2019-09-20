from cirq.ops.raw_types import Gate


class PartialGate(Gate):

    def __init__(self, sub_gate, *qubits):
        self.sub_gate = sub_gate
        self.qubits = qubits

    def num_qubits(self):
        return self.sub_gate.num_qubits()

    def on(self, *qubits):
        return self.sub_gate(*self.qubits, *qubits)
