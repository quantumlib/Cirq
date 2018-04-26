from typing import Callable

from cirq import ops, circuits
from cirq.circuits import OptimizationPass


class QubitMapper(OptimizationPass):
    def __init__(self, qubit_map: Callable[[ops.QubitId], ops.QubitId]
                 ) -> None:
        self.qubit_map = qubit_map

    def map_operation(self, operation: ops.Operation) -> ops.Operation:
        return ops.Operation(operation.gate,
                             [self.qubit_map(q) for q in operation.qubits])

    def map_moment(self, moment: circuits.Moment) -> circuits.Moment:
        return circuits.Moment(self.map_operation(op)
                               for op in moment.operations)

    def optimize_circuit(self, circuit: circuits.Circuit):
        circuit.moments = [self.map_moment(m) for m in circuit.moments]


def linearize_circuit_qubits(
        circuit: circuits.Circuit,
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT
        ) -> None:
    qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(
        circuit.qubits())
    qubit_map = {q: ops.LineQubit(i)
                 for i, q in enumerate(qubits)}
    QubitMapper(qubit_map.__getitem__).optimize_circuit(circuit)
