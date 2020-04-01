import numpy as np
import pytest
import cirq
import cirq.google as cg
import examples.direct_fidelity_estimation as direct_fidelity_estimation


def test_direct_fidelity_estimation_no_noise_clifford():
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.Z(qubits[0]), cirq.X(qubits[1]),
                           cirq.X(qubits[2]))

    no_noise = cirq.ConstantQubitNoiseModel(cirq.depolarize(0.0))
    no_noise_simulator = cirq.DensityMatrixSimulator(noise=no_noise)

    estimated_fidelity = direct_fidelity_estimation.direct_fidelity_estimation(
        circuit,
        qubits,
        no_noise_simulator,
        n_trials=100,
        n_clifford_trials=3,
        samples_per_term=0)
    assert np.isclose(estimated_fidelity, 1.0, atol=0.01)


def test_direct_fidelity_estimation_no_noise_non_clifford():
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.Z(qubits[0])**0.123, cirq.X(qubits[1]), cirq.X(qubits[2]))

    no_noise = cirq.ConstantQubitNoiseModel(cirq.depolarize(0.0))
    no_noise_simulator = cirq.DensityMatrixSimulator(noise=no_noise)

    estimated_fidelity = direct_fidelity_estimation.direct_fidelity_estimation(
        circuit,
        qubits,
        no_noise_simulator,
        n_trials=100,
        n_clifford_trials=3,
        samples_per_term=0)
    assert np.isclose(estimated_fidelity, 1.0, atol=0.01)


def test_direct_fidelity_estimation_with_noise():
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.Z(qubits[0])**0.25,  # T-Gate, non Clifford.
        cirq.X(qubits[1])**0.123,
        cirq.X(qubits[2])**0.456)

    noise = cirq.ConstantQubitNoiseModel(cirq.depolarize(0.1))
    noisy_simulator = cirq.DensityMatrixSimulator(noise=noise)

    estimated_fidelity = direct_fidelity_estimation.direct_fidelity_estimation(
        circuit,
        qubits,
        noisy_simulator,
        n_trials=10,
        n_clifford_trials=3,
        samples_per_term=10)
    assert estimated_fidelity >= -1.0 and estimated_fidelity <= 1.0


def test_incorrect_sampler_raises_exception():
    qubits = cirq.LineQubit.range(1)
    circuit = cirq.Circuit(cirq.X(qubits[0]))

    sampler_incorrect_type = cg.QuantumEngineSampler(engine=None,
                                                     processor_id='dummy_id',
                                                     gate_set=[])

    with pytest.raises(TypeError):
        direct_fidelity_estimation.direct_fidelity_estimation(
            circuit,
            qubits,
            sampler_incorrect_type,
            n_trials=100,
            n_clifford_trials=3,
            samples_per_term=0)


def test_same_pauli_traces_clifford():
    # When the circuit is Clifford, there is a speedup to compute the Pauli
    # traces. Here, we test that the Pauli traces returned by the general algo
    # and the speedup algo are the same.

    # Build a Clifford circuit and its states.
    qubits = cirq.LineQubit.range(3)
    n_qubits = len(qubits)
    circuit = cirq.Circuit(cirq.CNOT(qubits[0], qubits[2]), cirq.Z(qubits[0]),
                           cirq.H(qubits[2]), cirq.CNOT(qubits[2], qubits[1]),
                           cirq.X(qubits[0]), cirq.X(qubits[1]),
                           cirq.CNOT(qubits[0], qubits[2]))

    clifford_state = cirq.CliffordState(
        qubit_map={qubits[i]: i for i in range(len(qubits))})
    for gate in circuit.all_operations():
        clifford_state.apply_unitary(gate)

    # Run both algos
    pauli_traces_clifford = (
        direct_fidelity_estimation._estimate_pauli_traces_clifford(
            n_qubits, clifford_state, n_clifford_trials=3))
    pauli_traces_general = (
        direct_fidelity_estimation._estimate_pauli_traces_general(
            qubits, circuit))

    for pauli_trace_clifford in pauli_traces_clifford:
        pauli_trace_general = [
            x for x in pauli_traces_general
            if x['P_i'] == pauli_trace_clifford['P_i']
        ]
        assert len(pauli_trace_general) == 1
        pauli_trace_general = pauli_trace_general[0]

        # The code itself checks that the rho_i is either +1 or -1, so here we
        # simply test that the sign is the same.
        assert np.isclose(pauli_trace_general['rho_i'],
                          pauli_trace_clifford['rho_i'],
                          atol=0.01)


def test_parsing_args():
    direct_fidelity_estimation.parse_arguments(['--samples_per_term=10'])


def test_calling_main():
    direct_fidelity_estimation.main(n_trials=10,
                                    n_clifford_trials=3,
                                    samples_per_term=0)
    direct_fidelity_estimation.main(n_trials=10,
                                    n_clifford_trials=3,
                                    samples_per_term=10)
