import pytest
import cirq
from cirq.testing import assert_equivalent_op_tree
from cirq.devices.noise_properties import (
    NoiseProperties,
    NoiseModelFromNoiseProperties,
    get_duration_ns,
)
import numpy as np


def test_invalid_arguments():
    with pytest.raises(ValueError, match='At least one metric must be specified'):
        NoiseProperties()

    with pytest.raises(ValueError, match='xeb, pauli error, p00, and p11 must be between 0 and 1'):
        NoiseProperties(p00=1.2)

    with pytest.raises(ValueError, match='xeb, pauli error, p00, and p11 must be between 0 and 1'):
        NoiseProperties(pauli_error=-0.2)

    with pytest.raises(
        ValueError,
        match='Only one of xeb fidelity, pauli error, or decay constant should be defined',
    ):
        NoiseProperties(pauli_error=0.2, xeb_fidelity=0.5)

    with pytest.raises(ValueError, match='A NoiseProperties object must be specified'):
        NoiseModelFromNoiseProperties(None)


def test_constructor_and_metrics():
    prop = NoiseProperties(p00=0.2)
    assert prop.xeb is None
    assert prop.pauli_error is None
    assert prop.decay_constant is None
    assert prop.average_error() is None

    # These and other metrics in the file are purely for testing and
    # do not necessarily represent actual hardware behavior
    xeb_fidelity = 0.95
    p00 = 0.1
    t1_ns = 200.0

    # Create fidelity object with a defined XEB fidelity
    from_xeb = NoiseProperties(xeb_fidelity=xeb_fidelity, p00=p00, t1_ns=t1_ns)

    assert from_xeb.p00 == p00
    assert from_xeb.p11 is None
    assert from_xeb.t1_ns == t1_ns
    assert from_xeb.xeb == xeb_fidelity

    # Create another fidelity object with the decay constant from the first one
    decay_constant_from_xeb = from_xeb.decay_constant

    from_decay = NoiseProperties(decay_constant=decay_constant_from_xeb)

    # Check that their depolarization metrics match
    assert np.isclose(xeb_fidelity, from_decay.xeb)
    assert np.isclose(from_xeb.pauli_error, from_decay.pauli_error)
    assert np.isclose(from_xeb.average_error(), from_decay.average_error())


def test_gate_durations():
    assert get_duration_ns(cirq.X) == 25.0
    assert get_duration_ns(cirq.FSimGate(3 * np.pi / 2, np.pi / 6)) == 12.0
    assert get_duration_ns(cirq.FSimGate(3 * np.pi / 4, np.pi / 6)) == 32.0
    assert get_duration_ns(cirq.ISWAP) == 32.0
    assert get_duration_ns(cirq.ZPowGate(exponent=5)) == 0.0
    assert get_duration_ns(cirq.MeasurementGate(1, 'a')) == 4000.0

    wait_gate = cirq.WaitGate(cirq.Duration(nanos=4))
    assert get_duration_ns(wait_gate) == 4.0

    assert get_duration_ns(cirq.CZ) == 25.0


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

    # Create noise model from NoiseProperties object with specified noise
    prop = NoiseProperties(p00=p00, p11=p11)
    noise_model = NoiseModelFromNoiseProperties(prop)

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

    # Create Noise Model with just p00
    prop_p00 = NoiseProperties(p00=p00)
    noise_model_p00 = NoiseModelFromNoiseProperties(prop_p00)

    noisy_circuit_p00 = cirq.Circuit(noise_model_p00.noisy_moments(circuit, qubits))

    # Insert expected channels to circuit
    expected_circuit_p00 = cirq.Circuit(
        cirq.Moment([cirq.X(qubits[0])]),
        cirq.Moment([cirq.CNOT(qubits[0], qubits[1])]),
        cirq.Moment([cirq.H(qubits[1])]),
        cirq.Moment([cirq.GeneralizedAmplitudeDampingChannel(p=0.0, gamma=p00).on_each(qubits)]),
        cirq.Moment([cirq.measure(qubits[0], key='q0'), cirq.measure(qubits[1], key='q1')]),
    )

    assert_equivalent_op_tree(expected_circuit_p00, noisy_circuit_p00)

    # Create Noise Model with just p11
    prop_p11 = NoiseProperties(p11=p11)
    noise_model_p11 = NoiseModelFromNoiseProperties(prop_p11)

    noisy_circuit_p11 = cirq.Circuit(noise_model_p11.noisy_moments(circuit, qubits))

    # Insert expected channels to circuit
    expected_circuit_p11 = cirq.Circuit(
        cirq.Moment([cirq.X(qubits[0])]),
        cirq.Moment([cirq.CNOT(qubits[0], qubits[1])]),
        cirq.Moment([cirq.H(qubits[1])]),
        cirq.Moment([cirq.GeneralizedAmplitudeDampingChannel(p=1.0, gamma=p11).on_each(qubits)]),
        cirq.Moment([cirq.measure(qubits[0], key='q0'), cirq.measure(qubits[1], key='q1')]),
    )

    assert_equivalent_op_tree(expected_circuit_p11, noisy_circuit_p11)


def test_depolarization_error():
    # Account for floating point errors
    # Needs Cirq issue 3965 to be resolved
    pauli_error = 0.09999999999999998

    # Create qubits and circuit
    qubits = [cirq.LineQubit(0), cirq.LineQubit(1)]
    circuit = cirq.Circuit(
        cirq.Moment([cirq.X(qubits[0])]),
        cirq.Moment([cirq.CNOT(qubits[0], qubits[1])]),
        cirq.Moment([cirq.H(qubits[1])]),
        cirq.Moment([cirq.measure(qubits[0], key='q0'), cirq.measure(qubits[1], key='q1')]),
    )

    # Create noise model from NoiseProperties object with specified noise
    prop = NoiseProperties(pauli_error=pauli_error)
    noise_model = NoiseModelFromNoiseProperties(prop)

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
        cirq.Moment([cirq.depolarize(pauli_error / 3).on_each(qubits)]),
    )
    assert_equivalent_op_tree(expected_circuit, noisy_circuit)


def test_ampl_damping_error():
    t1_ns = 200.0

    # Create qubits and circuit
    qubits = [cirq.LineQubit(0), cirq.LineQubit(1)]
    circuit = cirq.Circuit(
        cirq.Moment([cirq.X(qubits[0])]),
        cirq.Moment([cirq.CNOT(qubits[0], qubits[1])]),
        cirq.Moment([cirq.FSimGate(5 * np.pi / 2, np.pi).on_each(qubits)]),
        cirq.Moment([cirq.measure(qubits[0], key='q0'), cirq.measure(qubits[1], key='q1')]),
    )

    # Create noise model from NoiseProperties object with specified noise
    prop = NoiseProperties(t1_ns=t1_ns)
    noise_model = NoiseModelFromNoiseProperties(prop)

    noisy_circuit = cirq.Circuit(noise_model.noisy_moments(circuit, qubits))

    # Insert expected channels to circuit
    expected_circuit = cirq.Circuit(
        cirq.Moment([cirq.X(qubits[0])]),
        cirq.Moment([cirq.amplitude_damp(1 - np.exp(-25.0 / t1_ns)).on_each(qubits)]),
        cirq.Moment([cirq.CNOT(qubits[0], qubits[1])]),
        cirq.Moment([cirq.amplitude_damp(1 - np.exp(-25.0 / t1_ns)).on_each(qubits)]),
        cirq.Moment([cirq.FSimGate(np.pi / 2, np.pi).on_each(qubits)]),
        cirq.Moment([cirq.amplitude_damp(1 - np.exp(-12.0 / t1_ns)).on_each(qubits)]),
        cirq.Moment([cirq.measure(qubits[0], key='q0'), cirq.measure(qubits[1], key='q1')]),
        cirq.Moment([cirq.amplitude_damp(1 - np.exp(-4000.0 / t1_ns)).on_each(qubits)]),
    )
    assert_equivalent_op_tree(expected_circuit, noisy_circuit)


def test_combined_error():
    # Helper function to calculate pauli error from depolarization
    def pauli_error_from_depolarization(pauli_error, t1_ns, duration):
        t2 = 2 * t1_ns
        pauli_error_from_t1 = (1 - np.exp(-duration / t2)) / 2 + (1 - np.exp(-duration / t1_ns)) / 4
        if pauli_error >= pauli_error_from_t1:
            return pauli_error - pauli_error_from_t1
        return pauli_error

    t1_ns = 2000.0
    p11 = 0.01

    # Account for floating point errors
    # Needs Cirq issue 3965 to be resolved
    pauli_error = 0.019999999999999962

    # Create qubits and circuit
    qubits = [cirq.LineQubit(0), cirq.LineQubit(1)]
    circuit = cirq.Circuit(
        cirq.Moment([cirq.X(qubits[0])]),
        cirq.Moment([cirq.CNOT(qubits[0], qubits[1])]),
        cirq.Moment([cirq.measure(qubits[0], key='q0')]),
        cirq.Moment([cirq.ISwapPowGate().on_each(qubits)]),
        cirq.Moment([cirq.measure(qubits[0], key='q0'), cirq.measure(qubits[1], key='q1')]),
    )

    # Create noise model from NoiseProperties object with specified noise
    prop = NoiseProperties(t1_ns=t1_ns, p11=p11, pauli_error=pauli_error)
    noise_model = NoiseModelFromNoiseProperties(prop)

    with pytest.warns(
        RuntimeWarning, match='Pauli error from T1 decay is greater than total Pauli error'
    ):
        noisy_circuit = cirq.Circuit(noise_model.noisy_moments(circuit, qubits))

    # Insert expected channels to circuit
    expected_circuit = cirq.Circuit(
        cirq.Moment([cirq.X(qubits[0])]),
        cirq.Moment(
            [
                cirq.depolarize(
                    pauli_error_from_depolarization(pauli_error, t1_ns, 25.0) / 3
                ).on_each(qubits)
            ]
        ),
        cirq.Moment([cirq.amplitude_damp(1 - np.exp(-25.0 / t1_ns)).on_each(qubits)]),
        cirq.Moment([cirq.CNOT(qubits[0], qubits[1])]),
        cirq.Moment(
            [
                cirq.depolarize(
                    pauli_error_from_depolarization(pauli_error, t1_ns, 25.0) / 3
                ).on_each(qubits)
            ]
        ),
        cirq.Moment([cirq.amplitude_damp(1 - np.exp(-25.0 / t1_ns)).on_each(qubits)]),
        cirq.Moment([cirq.GeneralizedAmplitudeDampingChannel(p=1.0, gamma=p11).on(qubits[0])]),
        cirq.Moment([cirq.measure(qubits[0], key='q0')]),
        cirq.Moment(
            [
                cirq.depolarize(
                    pauli_error_from_depolarization(pauli_error, t1_ns, 4000.0) / 3
                ).on_each(qubits)
            ]
        ),
        cirq.Moment([cirq.amplitude_damp(1 - np.exp(-4000.0 / t1_ns)).on_each(qubits)]),
        cirq.Moment([cirq.ISwapPowGate().on_each(qubits)]),
        cirq.Moment(
            [
                cirq.depolarize(
                    pauli_error_from_depolarization(pauli_error, t1_ns, 32.0) / 3
                ).on_each(qubits)
            ]
        ),
        cirq.Moment([cirq.amplitude_damp(1 - np.exp(-32.0 / t1_ns)).on_each(qubits)]),
        cirq.Moment([cirq.GeneralizedAmplitudeDampingChannel(p=1.0, gamma=p11).on_each(qubits)]),
        cirq.Moment([cirq.measure(qubits[0], key='q0'), cirq.measure(qubits[1], key='q1')]),
        cirq.Moment(
            [
                cirq.depolarize(
                    pauli_error_from_depolarization(pauli_error, t1_ns, 4000.0) / 3
                ).on_each(qubits)
            ]
        ),
        cirq.Moment([cirq.amplitude_damp(1 - np.exp(-4000.0 / t1_ns)).on_each(qubits)]),
    )
    assert_equivalent_op_tree(expected_circuit, noisy_circuit)
