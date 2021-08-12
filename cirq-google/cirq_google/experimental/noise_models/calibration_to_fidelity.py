import cirq
import cirq_google
import numpy as np
from cirq.devices.fidelity import Fidelity

def fidelity_from_calibration(calibration: cirq_google.Calibration):

  def unpack_from_calibration(metric_name):
    if metric_name in calibration.keys():
      return np.mean([value for qubit, value in calibration[metric_name].items()])
    else:
      return None

  def rb_error_to_decay_constant(rb_pauli_error, N: int = 2):
    if rb_pauli_error is not None:
      return 1 - rb_pauli_error / (1 - 1 / N**2)
    else:
      return None

  t1_micros = unpack_from_calibration('single_qubit_idle_t1_micros')
  t1_nanos = t1_micros * 1000 if t1_micros is not None else None
  xeb_fidelity = unpack_from_calibration('xeb')
  rb_pauli_error = unpack_from_calibration('single_qubit_rb_pauli_error_per_gate')
  p00 = unpack_from_calibration('single_qubit_p00_error')
  p11 = unpack_from_calibration('single_qubit_p11_error')
  decay_constant = rb_error_to_decay_constant(rb_pauli_error)

  if decay_constant is not None: # can't define both decay constant and xeb
    return Fidelity(t1 = t1_nanos, decay_constant = decay_constant, p00 = p00, p11 = p11)
  return Fidelity(t1 = t1_nanos, xeb_fidelity = xeb_fidelity, p00 = p00, p11 = p11)



