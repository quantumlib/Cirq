import pytest
import cirq
from cirq.testing import assert_equivalent_op_tree
import cirq_google
from cirq_google.experimental.noise_models.fidelity import *
from cirq_google.api import v2
from google.protobuf.text_format import Merge


def test_invalid_arguments():
  with pytest.raises(ValueError, match = 'At least one metric must be specified'):
    Fidelity()

  with pytest.raises(ValueError, match = 'xeb, pauli error, p00, and p11 must be between 0 and 1'):
    Fidelity(p00 = 1.2)

  with pytest.raises(ValueError, match = 'xeb, pauli error, p00, and p11 must be between 0 and 1'):
    Fidelity(pauli_error = -0.2)

  with pytest.raises(ValueError, match = 'Only one of xeb fidelity, pauli error, or decay constant should be defined'):
    Fidelity(pauli_error = 0.2, xeb_fidelity = 0.5)

  with pytest.raises(ValueError, match = 'A fidelity object must be specified'):
    NoiseModelFromFidelity(None)

def test_fidelity_from_calibration():
  xeb_1 = 0.999
  xeb_2 = 0.996

  p00_1 = 0.001
  p00_2 = 0.002
  p00_3 = 0.003

  t1_1 = 0.005
  t1_2 = 0.007
  t1_3 = 0.003

  _CALIBRATION_DATA = Merge(
      """
    timestamp_ms: 1579214873,
    metrics: [{
        name: 'xeb',
        targets: ['0_0', '0_1'],
        values: [{
            double_val: """ + str(xeb_1) + """
        }]
    }, {
        name: 'xeb',
        targets: ['0_0', '1_0'],
        values: [{
            double_val: """ + str(xeb_2) + """
        }]
    }, {
        name: 'single_qubit_p00_error',
        targets: ['0_0'],
        values: [{
            double_val: """ + str(p00_1) + """
        }]
    }, {
        name: 'single_qubit_p00_error',
        targets: ['0_1'],
        values: [{
            double_val: """ + str(p00_2) + """
        }]
    }, {
        name: 'single_qubit_p00_error',
        targets: ['1_0'],
        values: [{
            double_val: """ + str(p00_3) + """
        }]
    }, {
        name: 'single_qubit_readout_separation_error',
        targets: ['0_0'],
        values: [{
            double_val: .004
        }]
    }, {
        name: 'single_qubit_readout_separation_error',
        targets: ['0_1'],
        values: [{
            double_val: .005
        }]
    },{
        name: 'single_qubit_readout_separation_error',
        targets: ['1_0'],
        values: [{
            double_val: .006
        }]
    }, {
        name: 'single_qubit_idle_t1_micros',
        targets: ['0_0'],
        values: [{
            double_val: """ + str(t1_1) + """
        }]
    }, {
        name: 'single_qubit_idle_t1_micros',
        targets: ['0_1'],
        values: [{
            double_val: """ + str(t1_2) + """
        }]
    }, {
        name: 'single_qubit_idle_t1_micros',
        targets: ['1_0'],
        values: [{
            double_val: """ + str(t1_3) + """
        }]
    }]
""",
    v2.metrics_pb2.MetricsSnapshot(),
)

  # Create Fidelity objcet from Calibration
  calibration = cirq_google.Calibration(_CALIBRATION_DATA)
  f = fidelity_from_calibration(calibration)

  expected_t1_nanos = np.mean([t1_1, t1_2, t1_3]) * 1000
  expected_xeb_fidelity = np.mean([xeb_1, xeb_2])
  expected_p00 = np.mean([p00_1, p00_2, p00_3])

  assert np.isclose(f.t1,expected_t1_nanos)
  assert np.isclose(f.xeb,expected_xeb_fidelity)
  assert np.isclose(f.p00, expected_p00)

def test_from_calibration_rb():
  rb_pauli_1 = 0.001
  rb_pauli_2 = 0.002
  rb_pauli_3 = 0.003

  _CALIBRATION_DATA_RB = Merge(
    """
    timestamp_ms: 1579214873,
    metrics: [{

        name: 'single_qubit_rb_pauli_error_per_gate',
        targets: ['0_0'],
        values: [{
            double_val: """ + str(rb_pauli_1) + """
        }]
    }, {
        name: 'single_qubit_rb_pauli_error_per_gate',
        targets: ['0_1'],
        values: [{
            double_val: """ + str(rb_pauli_2) + """
        }]
    }, {
        name: 'single_qubit_rb_pauli_error_per_gate',
        targets: ['1_0'],
        values: [{
            double_val: """ + str(rb_pauli_3) + """
        }]
     }]
    """,
    v2.metrics_pb2.MetricsSnapshot(),
  )

 # Create Fidelity object from Calibration
  rb_calibration = cirq_google.Calibration(_CALIBRATION_DATA_RB)
  rb_fidelity = fidelity_from_calibration(rb_calibration)

  average_pauli_rb = np.mean([rb_pauli_1, rb_pauli_2, rb_pauli_3])
  assert np.isclose(average_pauli_rb, rb_fidelity.rb_pauli_error())

def test_metrics_conversions():
  pauli_error = 0.01
  N = 2 # one qubit

  decay_constant = 1 - pauli_error * N * N / (N * N - 1)
  xeb_fidelity = 1 - ((1 - decay_constant) * (1 - 1 / N))

  f = Fidelity(pauli_error = pauli_error)
  assert np.isclose(decay_constant, f.decay_constant)
  assert np.isclose(xeb_fidelity, f.decay_constant_to_xeb_fidelity(N = N))

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
      cirq.Moment([cirq.measure(qubits[0], key = 'q0'),
                   cirq.measure(qubits[1], key = 'q1')]),
  )

  # Create noise model from Fidelity object with specified noise
  f = Fidelity(p00 = p00, p11 = p11)
  noise_model = NoiseModelFromFidelity(f)

  noisy_circuit = cirq.Circuit(noise_model.noisy_moments(circuit, qubits))


  # Insert expected channels to circuit
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

  # Create qubits and circuit
  qubits = [cirq.LineQubit(0), cirq.LineQubit(1)]
  circuit = cirq.Circuit(
      cirq.Moment([cirq.X(qubits[0])]),
      cirq.Moment([cirq.CNOT(qubits[0], qubits[1])]),
      cirq.Moment([cirq.H(qubits[1])]),
      cirq.Moment([cirq.measure(qubits[0], key = 'q0'),
                   cirq.measure(qubits[1], key = 'q1')]),
  )


  # Create noise model from Fidelity object with specified noise
  f = Fidelity(pauli_error = pauli_error)
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
      cirq.Moment([cirq.measure(qubits[0], key = 'q0'),
                   cirq.measure(qubits[1], key = 'q1')]),
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
      cirq.Moment([cirq.measure(qubits[0], key = 'q0'),
                   cirq.measure(qubits[1], key = 'q1')]),
  )

  # Create noise model from Fidelity object with specified noise
  f = Fidelity(t1 = t1)
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
      cirq.Moment([cirq.measure(qubits[0], key = 'q0'),
                   cirq.measure(qubits[1], key = 'q1')]),
  )

  assert_equivalent_op_tree(expected_circuit, noisy_circuit)

def test_combined_error():
  t1 = 2000.0
  p00 = 0.01
  p11 = 0.01
  pauli_error = 0.02

 # Create qubits and circuit
  qubits = [cirq.LineQubit(0), cirq.LineQubit(1)]
  circuit = cirq.Circuit(
      cirq.Moment([cirq.X(qubits[0])]),
      cirq.Moment([cirq.CNOT(qubits[0], qubits[1])]),
      cirq.Moment([cirq.ISwapPowGate().on_each(qubits)]),
      cirq.Moment([cirq.measure(qubits[0], key = 'q0'),
                   cirq.measure(qubits[1], key = 'q1')]),
  )


  # Create noise model from Fidelity object with specified noise
  f = Fidelity(t1 = t1, p00 = p00, p11 = p11, pauli_error = pauli_error)
  noise_model = NoiseModelFromFidelity(f)

  noisy_circuit = cirq.Circuit(noise_model.noisy_moments(circuit, qubits))

  print(noisy_circuit)

  p = p11 / (p00 + p11)
  gamma = p11 / p

  # Insert expected channels to circuit
  expected_circuit = cirq.Circuit(
      cirq.Moment([cirq.X(qubits[0])]),
      cirq.Moment([cirq.depolarize(f.pauli_error_from_depolarization(25.0)/3).on_each(qubits)]),
      cirq.Moment([cirq.amplitude_damp(1 - np.exp(-25.0 / t1)).on_each(qubits)]),

      cirq.Moment([cirq.CNOT(qubits[0], qubits[1])]),
      cirq.Moment([cirq.depolarize(f.pauli_error_from_depolarization(25.0)/3).on_each(qubits)]),
      cirq.Moment([cirq.amplitude_damp(1 - np.exp(-25.0 / t1)).on_each(qubits)]),

      cirq.Moment([cirq.ISwapPowGate().on_each(qubits)]),
      cirq.Moment([cirq.depolarize(f.pauli_error_from_depolarization(32.0)/3).on_each(qubits)]),
      cirq.Moment([cirq.amplitude_damp(1 - np.exp(-32.0 / t1)).on_each(qubits)]),
      cirq.Moment([cirq.GeneralizedAmplitudeDampingChannel(p = p, gamma = gamma).on_each(qubits)]),
      cirq.Moment([cirq.measure(qubits[0], key = 'q0'),
                   cirq.measure(qubits[1], key = 'q1')]),
  )
  print(expected_circuit)

  assert_equivalent_op_tree(expected_circuit, noisy_circuit)


