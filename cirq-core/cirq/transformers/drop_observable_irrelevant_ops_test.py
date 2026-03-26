import cirq
import pytest
from drop_observable_irrelevant_ops import DropObservableIrrelevantOps

def test_removes_z_before_z_measurement():
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.Z(q), cirq.measure(q, key="m"))
    transformer = DropObservableIrrelevantOps(observable=cirq.Z)
    optimized = transformer.transform_circuit(circuit)
    expected = cirq.Circuit(cirq.measure(q, key="m"))
    assert optimized == expected

def test_does_not_remove_z_after_measurement():
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.measure(q, key="m"), cirq.Z(q))
    transformer = DropObservableIrrelevantOps(observable=cirq.Z)
    optimized = transformer.transform_circuit(circuit)
    expected = circuit
    assert optimized == expected

def test_does_not_remove_op_when_only_subset_of_qubits_measured():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.CZ(q0, q1),
        cirq.measure(q0, key="m"), 
    )
    transformer = DropObservableIrrelevantOps(observable=cirq.Z)
    optimized = transformer.transform_circuit(circuit)
    expected = circuit
    assert optimized == expected
