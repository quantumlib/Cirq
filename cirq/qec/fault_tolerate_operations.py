import cirq
from cirq.ops.raw_types import Operation


def apply_identical_gate(op: Operation, original_qubits, logical_qubits: dict, num: int):
    op_on_physical_qubits = cirq.Circuit()
    for i in range(num):
        op_list = []
        for lq in original_qubits:
            op_list.append(logical_qubits[lq].physical_qubits[i])
        op_on_physical_qubits.append(cirq.Circuit(op(*op_list)))
        ###bug list append doesn't work?
        ###bug cirq.Circuit(for lq in op.qubits) doesn't work
    return op_on_physical_qubits

def apply_on_physical_qubits(op: Operation, logical_qubits: dict, num: int):
    original_qubits = op.qubits
    if str(op.gate) == 'X':
        return apply_identical_gate(cirq.Z, original_qubits, logical_qubits, num)
    if str(op.gate) == 'Z':
        return apply_identical_gate(cirq.X, original_qubits, logical_qubits, num)
    if str(op.gate) == 'Y':
        return apply_identical_gate(cirq.Y, original_qubits, logical_qubits, num)
    else:
        NotImplemented
