import cirq
from cirq.qec.qec import OneQubitCode
from cirq.ops.raw_types import Operation


def apply_identical_gate(op: Operation, logical_qubits: dict["cirq.Qid", "OneQubitCode"], num: int):
    op_on_physical_qubits = []
    for i in range(num):
        op_on_physical_qubits.append(op.with_qubits(
            logical_qubits[lq].physical_qubits[i] for lq in op.qubits))
    return op_on_physical_qubits
    self.circuit.append(cirq.ops.Moment(op_on_physical_qubits))


def apply_on_physical_qubits(op: Operation, logical_qubits: dict["cirq.Qid", "OneQubitCode"], num: int):
    if isinstance(op, cirq.X) or isinstance(op, cirq.Y) or isinstance(op, cirq.Z) or isinstance(op. cirq.CNOT):
        return apply_identical_gate(op, logical_qubits, num)
    else:
        NotImplemented

