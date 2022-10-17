# pylint: disable=wrong-or-nonexistent-copyright-notice
import numpy as np
import pytest
import cirq
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
    sampler_incorrect_type = cirq.ZerosSampler

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
    n_qubits = 4

    qubits = cirq.LineQubit.range(n_qubits)
    circuit_clifford = cirq.Circuit(cirq.X(qubits[3]))

    circuit_general = cirq.Circuit(cirq.CCX(qubits[0], qubits[1], qubits[2]), circuit_clifford)

    def _run_dfe(circuit):
        class NoiseOnLastQubitOnly(cirq.NoiseModel):
            def __init__(self):
                self.qubit_noise_gate = cirq.amplitude_damp(1.0)

            def noisy_moment(self, moment, system_qubits):
                return [
                    moment,
                    cirq.Moment(
                        [
                            self.qubit_noise_gate(q).with_tags(cirq.ops.VirtualTag())
                            for q in system_qubits[-1:]
                        ]
                    ),
                ]

        noise = NoiseOnLastQubitOnly()
        noisy_simulator = cirq.DensityMatrixSimulator(noise=noise)

        _, intermediate_results = dfe.direct_fidelity_estimation(
            circuit, qubits, noisy_simulator, n_measured_operators=None, samples_per_term=1
        )
        return intermediate_results.pauli_traces, intermediate_results.clifford_tableau is not None

    # Run both algos
    pauli_traces_clifford, clifford_is_clifford = _run_dfe(circuit_clifford)
    pauli_traces_general, general_is_clifford = _run_dfe(circuit_general)

    assert clifford_is_clifford
    assert not general_is_clifford

    assert len(pauli_traces_clifford) == 2**n_qubits
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
    assert str(intermediate_result.clifford_tableau) == "+ Z "

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
