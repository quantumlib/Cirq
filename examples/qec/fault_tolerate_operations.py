"""
This program is to apply gates on encoded qubits
"""
import cirq
from onequbit_qec import OneQubitCode
from cirq.ops.raw_types import Operation, Gate

def apply_identical_gate(ga: Gate, original_qubits, logical_qubits: dict,
                         num: int):
    op_on_physical_qubits = cirq.Circuit()
    for i in range(num):
        op_list = []
        for lq in original_qubits:
            op_list.append(logical_qubits[lq].physical_qubits[i])
        op_on_physical_qubits.append(cirq.Circuit(ga(*op_list)))
    return op_on_physical_qubits


def apply_on_shors_code(op: Operation, logical_qubits: dict, num: int):
    original_qubits = op.qubits
    if str(op.gate) == 'X':
        return apply_identical_gate(cirq.Z, original_qubits, logical_qubits,
                                    num)
    if str(op.gate) == 'Z':
        return apply_identical_gate(cirq.X, original_qubits, logical_qubits,
                                    num)
    if str(op.gate) == 'Y':
        return apply_identical_gate(cirq.Y, original_qubits, logical_qubits,
                                    num)


def apply_on_physical_qubits(op: Operation, logical_qubits: dict, num: int,
                             codetype: OneQubitCode):
    if codetype.__name__ == 'OneQubitShorsCode':
        return apply_on_shors_code(op, logical_qubits, num)