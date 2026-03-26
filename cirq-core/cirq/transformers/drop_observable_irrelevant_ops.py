import cirq
from collections import defaultdict
from typing import Dict, Set

def is_diagonal_in_z(op: cirq.Operation) -> bool:
    """Returns True if the given operation is guaranteed to be diagonal in the computational (Z) basis."""
    gate = op.gate
    if gate is None:
        return False
    if isinstance(gate, cirq.ZPowGate):
        return True
    if isinstance(gate, cirq.CZPowGate):
        return True
    return False

def collect_measurements(circuit: cirq.Circuit) -> Dict[str, Set[cirq.Qid]]:
    """Collects measured qubits grouped by measurement key from the circuit."""
    measured_qubits_by_key: Dict[str, Set[cirq.Qid]] = defaultdict(set)
    for moment in circuit:
        for op in moment.operations:
            if isinstance(op.gate, cirq.MeasurementGate):
                key = op.gate.key
                measured_qubits_by_key[key].update(op.qubits)
    return dict(measured_qubits_by_key)

def collect_measurement_moments(circuit: cirq.Circuit) -> Dict[str, int]:
    """Collects the first moment index at which each measurement key appears."""
    measurement_moments: Dict[str, int] = {}
    for moment_index, moment in enumerate(circuit):
        for op in moment.operations:
            if isinstance(op.gate, cirq.MeasurementGate):
                key = op.gate.key
                measurement_moments.setdefault(key, moment_index)
    return measurement_moments

class DropObservableIrrelevantOps(cirq.transformers.Transformer):
    def __init__(self, observable=cirq.Z):
        self.observable = observable

    def transform_circuit(
        self,
        circuit: cirq.Circuit,
        context: cirq.TransformerContext | None = None,
    ) -> cirq.Circuit:
        """Returns a new circuit with observable-irrelevant operations removed."""
        measured_qubits_by_key = collect_measurements(circuit)
        measurement_moments = collect_measurement_moments(circuit)

        new_moments: list[cirq.Moment] = []

        for moment_index, moment in enumerate(circuit):
            new_ops = []
            for op in moment.operations:
                # Never remove measurements themselves
                if isinstance(op.gate, cirq.MeasurementGate):
                    new_ops.append(op)
                    continue

                if self._is_removable_at(
                    op,
                    moment_index,
                    measured_qubits_by_key,
                    measurement_moments,
                ):
                    continue  # Safe to drop

                new_ops.append(op)

            if new_ops:
                new_moments.append(cirq.Moment(new_ops))

        return cirq.Circuit(new_moments)

    def _is_removable_at(
        self,
        op: cirq.Operation,
        moment_index: int,
        measured_qubits_by_key: dict[str, set[cirq.Qid]],
        measurement_moments: dict[str, int],
    ) -> bool:
        """Time-aware and context-aware removal check."""
        if not is_diagonal_in_z(op):
            return False

        for key, measured_qubits in measured_qubits_by_key.items():
            measurement_time = measurement_moments.get(key)
            if measurement_time is None:
                continue

            if (
                moment_index < measurement_time
                and op.qubits.issubset(measured_qubits)
            ):
                return True

        return False
