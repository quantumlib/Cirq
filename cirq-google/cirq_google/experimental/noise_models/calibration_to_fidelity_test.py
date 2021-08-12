import pytest
import cirq
import cirq_google
from cirq.devices.fidelity import Fidelity, NoiseModelFromFidelity
from cirq_google.api import v2
from cirq_google.experimental.noise_models.calibration_to_fidelity import fidelity_from_calibration
from google.protobuf.text_format import Merge
import numpy as np

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

