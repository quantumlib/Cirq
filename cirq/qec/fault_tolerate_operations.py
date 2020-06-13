import cirq
from cirq.ops.raw_types import Operation


def apply_identical_gate(op: Operation, logical_qubits: dict, num: int):
    op_on_physical_qubits = cirq.Circuit()
    for i in range(num):
        op_list = []
        for lq in op.qubits:
            op_list.append(logical_qubits[lq].physical_qubits[i])
        op_on_physical_qubits.append(cirq.Circuit(op.with_qubits(*op_list)))
        ###bug list append doesn't work?
        ###bug cirq.Circuit(for lq in op.qubits) doesn't work
    return op_on_physical_qubits

def apply_on_physical_qubits(op: Operation, logical_qubits: dict, num: int):
    if str(op.gate) == 'X' or str(op.gate) == 'Y' or str(op.gate) == 'Z' or str(op.gate) == 'CNOT':
        return apply_identical_gate(op, logical_qubits, num)
    else:
        NotImplemented
