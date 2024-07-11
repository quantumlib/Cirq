import cirq
import cirq.transformers.noise_adding as na


def test_noise_adding():
    qubits = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(cirq.CZ(*qubits[:2]), cirq.CZ(*qubits[2:])) * 10
    transformed_circuit_p0 = na.add_depolarizing_noise_to_two_qubit_gates(circuit, 0.0)
    assert transformed_circuit_p0 == circuit
    transformed_circuit_p1 = na.add_depolarizing_noise_to_two_qubit_gates(circuit, 1.0)
    assert len(transformed_circuit_p1) == 20
    transformed_circuit_p0_03 = na.add_depolarizing_noise_to_two_qubit_gates(circuit, 0.03)
