from typing import Callable

from cirq import ops, circuits
from cirq.circuits import OptimizationPass
from cirq.contrib.quirk.line_qubit import LineQubit
from cirq.value import sorting_str


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
        qubit_order_key: Callable[[ops.QubitId], str] = None
        ) -> None:
    if qubit_order_key is None:
        qubit_order_key = sorting_str
    qubit_map = {q: LineQubit(i)
                 for i, q in enumerate(sorted(circuit.qubits(),
                                              key=qubit_order_key))}
    QubitMapper(qubit_map.__getitem__).optimize_circuit(circuit)
