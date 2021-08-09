import pytest
import cirq
from cirq.testing import assert_equivalent_op_tree
from cirq_google.experimental.noise_models.fidelity import *

def test_invalid_arguments():
  with pytest.raises(ValueError, match = 'At least one metric must be specified'):
    Fidelity()

  with pytest.raises(ValueError, match = 'xeb, pauli error, p00, and p11 must be between 0 and 1'):
    Fidelity(p00 = 1.2)

  with pytest.raises(ValueError, match = 'xeb, pauli error, p00, and p11 must be between 0 and 1'):
    Fidelity(pauli_error = -0.2)

def test_metrics_conversions():
  pauli_error = 0.01
  N = 2 # one qubit

  decay_constant = 1 - pauli_error * N * N / (N * N - 1)
  xeb = (1 - decay_constant) * (1 - 1 / N)

  f = Fidelity(pauli_error = pauli_error)
  assert np.isclose(decay_constant, f.decay_constant)
  assert np.isclose(xeb, f.decay_constant_to_xeb_error(N = N))

def test_readout_error():
  p00 = 0.05
  p11 = 0.1

  p = p11 / (p00 + p11)
  gamma = p11 / p

  qubits = [cirq.LineQubit(0), cirq.LineQubit(1)]
  circuit = cirq.Circuit(
      cirq.Moment([cirq.X(qubits[0])]),
      cirq.Moment([cirq.CNOT(qubits[0], qubits[1])]),
      cirq.Moment([cirq.H(qubits[1])]),
      cirq.Moment([cirq.measure(qubits[0], key = 'q0'),
                   cirq.measure(qubits[1], key = 'q1')]),
  )

  f = Fidelity(p00 = p00, p11 = p11)
  noise_model = NoiseModelFromFidelity(f)

  noisy_circuit = cirq.Circuit(noise_model.noisy_moments(circuit, qubits))

  expected_circuit = cirq.Circuit(
      cirq.Moment([cirq.X(qubits[0])]),
      cirq.Moment([cirq.CNOT(qubits[0], qubits[1])]),
      cirq.Moment([cirq.H(qubits[1])]),
      cirq.Moment([cirq.GeneralizedAmplitudeDampingChannel(p = p, gamma = gamma).on_each(qubits)]),
      cirq.Moment([cirq.measure(qubits[0], key = 'q0'),
                   cirq.measure(qubits[1], key = 'q1')]),
  )

  assert_equivalent_op_tree(expected_circuit, noisy_circuit)

def test_depolarization_error():
  pauli_error = 0.1

  qubits = [cirq.LineQubit(0), cirq.LineQubit(1)]
  circuit = cirq.Circuit(
      cirq.Moment([cirq.X(qubits[0])]),
      cirq.Moment([cirq.CNOT(qubits[0], qubits[1])]),
      cirq.Moment([cirq.H(qubits[1])]),
      cirq.Moment([cirq.measure(qubits[0], key = 'q0'),
                   cirq.measure(qubits[1], key = 'q1')]),
  )

  f = Fidelity(pauli_error = pauli_error)
  noise_model = NoiseModelFromFidelity(f)

  noisy_circuit = cirq.Circuit(noise_model.noisy_moments(circuit, qubits))

  expected_circuit = cirq.Circuit(
      cirq.Moment([cirq.X(qubits[0])]),
      cirq.Moment([cirq.depolarize(pauli_error / 3).on_each(qubits)]),
      cirq.Moment([cirq.CNOT(qubits[0], qubits[1])]),
      cirq.Moment([cirq.depolarize(pauli_error / 3).on_each(qubits)]),
      cirq.Moment([cirq.H(qubits[1])]),
      cirq.Moment([cirq.depolarize(pauli_error / 3).on_each(qubits)]),
      cirq.Moment([cirq.measure(qubits[0], key = 'q0'),
                   cirq.measure(qubits[1], key = 'q1')]),
  )

  assert_equivalent_op_tree(expected_circuit, noisy_circuit)

def test_ampl_damping_error():
  t1 = 200.0

  qubits = [cirq.LineQubit(0), cirq.LineQubit(1)]
  circuit = cirq.Circuit(
      cirq.Moment([cirq.X(qubits[0])]),
      cirq.Moment([cirq.CNOT(qubits[0], qubits[1])]),
      cirq.Moment([cirq.FSimGate(np.pi / 2, np.pi).on_each(qubits)]),
      cirq.Moment([cirq.measure(qubits[0], key = 'q0'),
                   cirq.measure(qubits[1], key = 'q1')]),
  )

  f = Fidelity(t1 = t1)
  noise_model = NoiseModelFromFidelity(f)

  noisy_circuit = cirq.Circuit(noise_model.noisy_moments(circuit, qubits))

  expected_circuit = cirq.Circuit(
      cirq.Moment([cirq.X(qubits[0])]),
      cirq.Moment([cirq.amplitude_damp(1 - np.exp(-25.0 / t1)).on_each(qubits)]),
      cirq.Moment([cirq.CNOT(qubits[0], qubits[1])]),
      cirq.Moment([cirq.amplitude_damp(1 - np.exp(-25.0 / t1)).on_each(qubits)]),
      cirq.Moment([cirq.FSimGate(np.pi / 2, np.pi).on_each(qubits)]),
      cirq.Moment([cirq.amplitude_damp(1 - np.exp(-12.0 / t1)).on_each(qubits)]),
      cirq.Moment([cirq.measure(qubits[0], key = 'q0'),
                   cirq.measure(qubits[1], key = 'q1')]),
  )

  assert_equivalent_op_tree(expected_circuit, noisy_circuit)


