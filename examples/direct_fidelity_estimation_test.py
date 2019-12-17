import cirq
import examples.direct_fidelity_estimation as direct_fidelity_estimation


def test_direct_fidelity_estimation():
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.Z(qubits[0])**0.25,  # T-Gate, non Clifford.
        cirq.X(qubits[1])**0.123,
        cirq.X(qubits[2])**0.456)

    noise = cirq.ConstantQubitNoiseModel(cirq.depolarize(0.1))

    estimated_fidelity = direct_fidelity_estimation.direct_fidelity_estimation(
        circuit, qubits, noise, n_trials=10, samples_per_term=10)
    assert estimated_fidelity >= -1.0 and estimated_fidelity <= 1.0


def test_parsing_args():
    direct_fidelity_estimation.parse_arguments(['--samples_per_term=10'])


def test_calling_main():
    direct_fidelity_estimation.main(n_trials=10, samples_per_term=0)
    direct_fidelity_estimation.main(n_trials=10, samples_per_term=10)
