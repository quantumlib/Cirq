import numpy as np
import pytest
import cirq
import cirq.google as cg
import examples.direct_fidelity_estimation as dfe


def test_direct_fidelity_estimation_no_noise_clifford():
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.Z(qubits[0]), cirq.X(qubits[1]), cirq.X(qubits[2]))

    no_noise_simulator = cirq.DensityMatrixSimulator()

    estimated_fidelity, _ = dfe.direct_fidelity_estimation(
        circuit, qubits, no_noise_simulator, n_measured_operators=3, samples_per_term=0
    )
    assert np.isclose(estimated_fidelity, 1.0, atol=0.01)


def test_direct_fidelity_estimation_no_noise_non_clifford():
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.Z(qubits[0]) ** 0.123, cirq.X(qubits[1]), cirq.X(qubits[2]))

    no_noise_simulator = cirq.DensityMatrixSimulator()

    estimated_fidelity, _ = dfe.direct_fidelity_estimation(
        circuit, qubits, no_noise_simulator, n_measured_operators=64, samples_per_term=0
    )
    assert np.isclose(estimated_fidelity, 1.0, atol=0.01)


def test_direct_fidelity_estimation_with_noise_clifford():
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.Z(qubits[0]), cirq.X(qubits[1]), cirq.X(qubits[2]))

    noise = cirq.ConstantQubitNoiseModel(cirq.depolarize(0.1))
    noisy_simulator = cirq.DensityMatrixSimulator(noise=noise)

    estimated_fidelity, _ = dfe.direct_fidelity_estimation(
        circuit, qubits, noisy_simulator, n_measured_operators=None, samples_per_term=100
    )
    assert estimated_fidelity >= -1.0 and estimated_fidelity <= 1.0


def test_direct_fidelity_estimation_with_noise_non_clifford():
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.Z(qubits[0]) ** 0.25,  # T-Gate, non Clifford.
        cirq.X(qubits[1]) ** 0.123,
        cirq.X(qubits[2]) ** 0.456,
    )

    noise = cirq.ConstantQubitNoiseModel(cirq.depolarize(0.1))
    noisy_simulator = cirq.DensityMatrixSimulator(noise=noise)

    estimated_fidelity, _ = dfe.direct_fidelity_estimation(
        circuit, qubits, noisy_simulator, n_measured_operators=None, samples_per_term=100
    )
    assert estimated_fidelity >= -1.0 and estimated_fidelity <= 1.0


def test_incorrect_sampler_raises_exception():
    qubits = cirq.LineQubit.range(1)
    circuit = cirq.Circuit(cirq.X(qubits[0]))

    sampler_incorrect_type = cg.QuantumEngineSampler(
        engine=None, processor_id='dummy_id', gate_set=[]
    )

    with pytest.raises(TypeError):
        dfe.direct_fidelity_estimation(
            circuit, qubits, sampler_incorrect_type, n_measured_operators=3, samples_per_term=0
        )


def test_direct_fidelity_estimation_clifford_all_trials():
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.Z(qubits[0]), cirq.X(qubits[1]))

    no_noise_simulator = cirq.DensityMatrixSimulator()

    for n_measured_operators in [1, 2, 3, 4, None]:
        estimated_fidelity, _ = dfe.direct_fidelity_estimation(
            circuit,
            qubits,
            no_noise_simulator,
            n_measured_operators=n_measured_operators,
            samples_per_term=0,
        )
        assert np.isclose(estimated_fidelity, 1.0, atol=0.01)


def test_same_pauli_traces_clifford():
    # When the circuit is Clifford, there is a speedup to compute the Pauli
    # traces. Here, we test that the Pauli traces returned by the general algo
    # and the speedup algo are the same.

    # Build a Clifford circuit and its states.
    qubits = cirq.LineQubit.range(3)
    n_qubits = len(qubits)
    circuit = cirq.Circuit(
        cirq.CNOT(qubits[0], qubits[2]),
        cirq.Z(qubits[0]),
        cirq.H(qubits[2]),
        cirq.CNOT(qubits[2], qubits[1]),
        cirq.X(qubits[0]),
        cirq.X(qubits[1]),
        cirq.CNOT(qubits[0], qubits[2]),
    )

    clifford_state = cirq.CliffordState(qubit_map={qubits[i]: i for i in range(len(qubits))})
    for gate in circuit.all_operations():
        clifford_state.apply_unitary(gate)

    # Run both algos
    pauli_traces_clifford = dfe._estimate_pauli_traces_clifford(
        n_qubits, clifford_state, n_measured_operators=None
    )
    pauli_traces_general = dfe._estimate_pauli_traces_general(
        qubits, circuit, n_measured_operators=None
    )

    assert len(pauli_traces_clifford) == 2 ** n_qubits
    for pauli_trace_clifford in pauli_traces_clifford:
        pauli_trace_general = [x for x in pauli_traces_general if x.P_i == pauli_trace_clifford.P_i]
        assert len(pauli_trace_general) == 1
        pauli_trace_general = pauli_trace_general[0]

        # The code itself checks that the rho_i is either +1 or -1, so here we
        # simply test that the sign is the same.
        assert np.isclose(pauli_trace_general.rho_i, pauli_trace_clifford.rho_i, atol=0.01)


def test_direct_fidelity_estimation_intermediate_results():
    qubits = cirq.LineQubit.range(1)
    circuit = cirq.Circuit(cirq.I(qubits[0]))
    no_noise_simulator = cirq.DensityMatrixSimulator()

    _, intermediate_result = dfe.direct_fidelity_estimation(
        circuit, qubits, no_noise_simulator, n_measured_operators=1, samples_per_term=0
    )
    # We only test a few fields to be sure that they are set properly. In
    # particular, some of them are random, and so we don't test them.
    np.testing.assert_allclose(intermediate_result.clifford_state.ch_form.gamma, [0])

    np.testing.assert_equal(len(intermediate_result.pauli_traces), 1)
    assert np.isclose(intermediate_result.pauli_traces[0].rho_i, 1.0)
    assert np.isclose(intermediate_result.pauli_traces[0].Pr_i, 0.5)

    np.testing.assert_equal(len(intermediate_result.trial_results), 1)
    assert np.isclose(intermediate_result.trial_results[0].sigma_i, 1.0)

    assert np.isclose(intermediate_result.std_dev_estimate, 0.0)
    assert np.isclose(intermediate_result.std_dev_bound, 0.5)


def test_parsing_args():
    dfe.parse_arguments(['--samples_per_term=10'])


def test_calling_main():
    dfe.main(n_measured_operators=3, samples_per_term=0)
    dfe.main(n_measured_operators=3, samples_per_term=10)
