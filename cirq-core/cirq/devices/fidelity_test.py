import pytest
import cirq
from cirq.testing import assert_equivalent_op_tree
from cirq.devices.fidelity import Fidelity, NoiseModelFromFidelity
import numpy as np


def test_invalid_arguments():
    with pytest.raises(ValueError, match='At least one metric must be specified'):
        Fidelity()

    with pytest.raises(ValueError, match='xeb, pauli error, p00, and p11 must be between 0 and 1'):
        Fidelity(p00=1.2)

    with pytest.raises(ValueError, match='xeb, pauli error, p00, and p11 must be between 0 and 1'):
        Fidelity(pauli_error=-0.2)

    with pytest.raises(
        ValueError,
        match='Only one of xeb fidelity, pauli error, or decay constant should be defined',
    ):
        Fidelity(pauli_error=0.2, xeb_fidelity=0.5)

    with pytest.raises(ValueError, match='A fidelity object must be specified'):
        NoiseModelFromFidelity(None)


def test_constructor():
    xeb_fidelity = 0.95
    p00 = 0.1
    t1 = 200.0

    # Create fidelity object with a defined XEB fidelity
    f_from_xeb = Fidelity(xeb_fidelity=xeb_fidelity, p00=p00, t1=t1)

    assert f_from_xeb.p00 == p00
    assert f_from_xeb.p11 is None
    assert f_from_xeb.t1 == t1
    assert f_from_xeb.xeb == xeb_fidelity

    # Create another fidelity object with the decay constant from the first one
    decay_constant_from_xeb = f_from_xeb.decay_constant

    f_from_decay = Fidelity(decay_constant=decay_constant_from_xeb)

    # Check that their depolarization metrics match
    assert np.isclose(xeb_fidelity, f_from_decay.xeb)
    assert np.isclose(f_from_xeb.pauli_error, f_from_decay.pauli_error)
    assert np.isclose(f_from_xeb.rb_average_error(), f_from_decay.rb_average_error())
    assert np.isclose(f_from_xeb.rb_pauli_error(), f_from_decay.rb_pauli_error())


def test_readout_error():
    p00 = 0.05
    p11 = 0.1

    p = p11 / (p00 + p11)
    gamma = p11 / p

    # Create qubits and circuit
    qubits = [cirq.LineQubit(0), cirq.LineQubit(1)]
    circuit = cirq.Circuit(
        cirq.Moment([cirq.X(qubits[0])]),
        cirq.Moment([cirq.CNOT(qubits[0], qubits[1])]),
        cirq.Moment([cirq.H(qubits[1])]),
        cirq.Moment([cirq.measure(qubits[0], key='q0'), cirq.measure(qubits[1], key='q1')]),
    )

    # Create noise model from Fidelity object with specified noise
    f = Fidelity(p00=p00, p11=p11)
    noise_model = NoiseModelFromFidelity(f)

    noisy_circuit = cirq.Circuit(noise_model.noisy_moments(circuit, qubits))

    # Insert expected channels to circuit
    expected_circuit = cirq.Circuit(
        cirq.Moment([cirq.X(qubits[0])]),
        cirq.Moment([cirq.CNOT(qubits[0], qubits[1])]),
        cirq.Moment([cirq.H(qubits[1])]),
        cirq.Moment([cirq.GeneralizedAmplitudeDampingChannel(p=p, gamma=gamma).on_each(qubits)]),
        cirq.Moment([cirq.measure(qubits[0], key='q0'), cirq.measure(qubits[1], key='q1')]),
    )

    assert_equivalent_op_tree(expected_circuit, noisy_circuit)


def test_depolarization_error():
    pauli_error = 0.1

    # Create qubits and circuit
    qubits = [cirq.LineQubit(0), cirq.LineQubit(1)]
    circuit = cirq.Circuit(
        cirq.Moment([cirq.X(qubits[0])]),
        cirq.Moment([cirq.CNOT(qubits[0], qubits[1])]),
        cirq.Moment([cirq.H(qubits[1])]),
        cirq.Moment([cirq.measure(qubits[0], key='q0'), cirq.measure(qubits[1], key='q1')]),
    )

    # Create noise model from Fidelity object with specified noise
    f = Fidelity(pauli_error=pauli_error)
    noise_model = NoiseModelFromFidelity(f)

    noisy_circuit = cirq.Circuit(noise_model.noisy_moments(circuit, qubits))

    # Insert expected channels to circuit
    expected_circuit = cirq.Circuit(
        cirq.Moment([cirq.X(qubits[0])]),
        cirq.Moment([cirq.depolarize(pauli_error / 3).on_each(qubits)]),
        cirq.Moment([cirq.CNOT(qubits[0], qubits[1])]),
        cirq.Moment([cirq.depolarize(pauli_error / 3).on_each(qubits)]),
        cirq.Moment([cirq.H(qubits[1])]),
        cirq.Moment([cirq.depolarize(pauli_error / 3).on_each(qubits)]),
        cirq.Moment([cirq.measure(qubits[0], key='q0'), cirq.measure(qubits[1], key='q1')]),
    )

    assert_equivalent_op_tree(expected_circuit, noisy_circuit)


def test_ampl_damping_error():
    t1 = 200.0

    # Create qubits and circuit
    qubits = [cirq.LineQubit(0), cirq.LineQubit(1)]
    circuit = cirq.Circuit(
        cirq.Moment([cirq.X(qubits[0])]),
        cirq.Moment([cirq.CNOT(qubits[0], qubits[1])]),
        cirq.Moment([cirq.FSimGate(5 * np.pi / 2, np.pi).on_each(qubits)]),
        cirq.Moment([cirq.measure(qubits[0], key='q0'), cirq.measure(qubits[1], key='q1')]),
    )

    # Create noise model from Fidelity object with specified noise
    f = Fidelity(t1=t1)
    noise_model = NoiseModelFromFidelity(f)

    noisy_circuit = cirq.Circuit(noise_model.noisy_moments(circuit, qubits))

    # Insert expected channels to circuit
    expected_circuit = cirq.Circuit(
        cirq.Moment([cirq.X(qubits[0])]),
        cirq.Moment([cirq.amplitude_damp(1 - np.exp(-25.0 / t1)).on_each(qubits)]),
        cirq.Moment([cirq.CNOT(qubits[0], qubits[1])]),
        cirq.Moment([cirq.amplitude_damp(1 - np.exp(-25.0 / t1)).on_each(qubits)]),
        cirq.Moment([cirq.FSimGate(np.pi / 2, np.pi).on_each(qubits)]),
        cirq.Moment([cirq.amplitude_damp(1 - np.exp(-12.0 / t1)).on_each(qubits)]),
        cirq.Moment([cirq.measure(qubits[0], key='q0'), cirq.measure(qubits[1], key='q1')]),
    )

    assert_equivalent_op_tree(expected_circuit, noisy_circuit)


def test_combined_error():
    # Helper function to calculate pauli error from depolarization
    def pauli_error_from_depolarization(pauli_error, t1, duration):
        t2 = 2 * t1
        pauli_error_from_t1 = (1 - np.exp(-duration / t2)) / 2 + (1 - np.exp(-duration / t1)) / 4
        return pauli_error - pauli_error_from_t1

    t1 = 2000.0
    p11 = 0.01
    pauli_error = 0.02

    # Create qubits and circuit
    qubits = [cirq.LineQubit(0), cirq.LineQubit(1)]
    circuit = cirq.Circuit(
        cirq.Moment([cirq.X(qubits[0])]),
        cirq.Moment([cirq.CNOT(qubits[0], qubits[1])]),
        cirq.Moment([cirq.ISwapPowGate().on_each(qubits)]),
        cirq.Moment([cirq.measure(qubits[0], key='q0'), cirq.measure(qubits[1], key='q1')]),
    )

    # Create noise model from Fidelity object with specified noise
    f = Fidelity(t1=t1, p11=p11, pauli_error=pauli_error)
    noise_model = NoiseModelFromFidelity(f)

    noisy_circuit = cirq.Circuit(noise_model.noisy_moments(circuit, qubits))

    # Insert expected channels to circuit
    expected_circuit = cirq.Circuit(
        cirq.Moment([cirq.X(qubits[0])]),
        cirq.Moment(
            [
                cirq.depolarize(pauli_error_from_depolarization(pauli_error, t1, 25.0) / 3).on_each(
                    qubits
                )
            ]
        ),
        cirq.Moment([cirq.amplitude_damp(1 - np.exp(-25.0 / t1)).on_each(qubits)]),
        cirq.Moment([cirq.CNOT(qubits[0], qubits[1])]),
        cirq.Moment(
            [
                cirq.depolarize(pauli_error_from_depolarization(pauli_error, t1, 25.0) / 3).on_each(
                    qubits
                )
            ]
        ),
        cirq.Moment([cirq.amplitude_damp(1 - np.exp(-25.0 / t1)).on_each(qubits)]),
        cirq.Moment([cirq.ISwapPowGate().on_each(qubits)]),
        cirq.Moment(
            [
                cirq.depolarize(pauli_error_from_depolarization(pauli_error, t1, 32.0) / 3).on_each(
                    qubits
                )
            ]
        ),
        cirq.Moment([cirq.amplitude_damp(1 - np.exp(-32.0 / t1)).on_each(qubits)]),
        cirq.Moment([cirq.GeneralizedAmplitudeDampingChannel(p=1.0, gamma=p11).on_each(qubits)]),
        cirq.Moment([cirq.measure(qubits[0], key='q0'), cirq.measure(qubits[1], key='q1')]),
    )

    assert_equivalent_op_tree(expected_circuit, noisy_circuit)
