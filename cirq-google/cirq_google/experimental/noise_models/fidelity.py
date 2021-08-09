import cirq
import numpy as np
from typing import Optional, Sequence
from cirq.devices.noise_model import validate_all_measurements
from itertools import product

class Fidelity():
  def __init__(
      self,
      *,
      t1: Optional = None,
      decay_constant: Optional = None,
      xeb_fidelity: Optional = None,
      pauli_error: Optional = None,
      p00: Optional = None,
      p11: Optional = None
  ) -> None:
    """
    Detailed comments about arguments here
    """
    if not any([t1, decay_constant, xeb_fidelity, pauli_error, p00, p11]):
      raise ValueError('At least one metric must be specified')

    for metric in [xeb_fidelity, pauli_error, p00, p11]:
      if metric is not None and not 0.0 <= metric <= 1.0:
        raise ValueError('Invalid metric value')

    self._t1 = t1
    self._p = decay_constant
    self._p00 = p00
    self._p11 = p11
    self._pauli_error = pauli_error
    self._xeb = xeb_fidelity

  @property
  def decay_constant(self):
    return self._p

  @property
  def p00(self):
    return self._p00

  @property
  def p11(self):
    return self._p11

  @property
  def pauli_error(self):
    return self._pauli_error

  @property
  def t1(self):
    return self._t1

  def decay_constant_to_xeb_error(self, decay_constant: float, N: int = 4):
    return (1 - decay_constant) * (1 - 1 / N)

  def decay_constant_to_pauli_error(self, decay_constant: float, N: int = 2):
    return (1 - decay_constant) * (1 - 1 / N / N)

  def pauli_error_to_xeb_error(self, pauli_error: float, N: int = 4):
    decay_constant = 1 - (pauli_error / (1 - 1 / N))
    return decay_constant_to_xeb_error(decay_constant)

  def pauli_error_to_decay_constant(self, pauli_error: float, N: int = 2):
    return 1 - (pauli_error / (1 - 1 / N / N))

  def pauli_error_from_t1(self, t: float, t1: float):
    t2 = 2 * t1
    return (1 - np.exp(-t / t2)) / 2 + (1 - np.exp(-t / t1)) / 4

  def pauli_error_from_depolarization(self, t : float):
    if self._t1 is not None:
      return self._pauli_error - self.pauli_error_from_t1(t, self._t1)
    else:
      return self._pauli_error



class NoiseModelFromFidelity(cirq.NoiseModel):
  def __init__(
      self,
      fidelity: Fidelity
  ) -> None:
    if fidelity is not None:
      self._fidelity = fidelity
    else:
      raise ValueError('A fidelity object must be specified')

  def get_duration(self, gate):
    if isinstance(gate, cirq.FSimGate):
      theta,_ = gate._value_equality_values_()
      if np.abs(theta) == np.pi / 2: ## should be any mutiple of pi/2
        return 12.0 ## syc
      return 32.0 ## fsim_pi_4
    elif isinstance(gate, cirq.ISwapPowGate):
      return 32.0 #inv_fsim_pi_4
    elif isinstance(gate, cirq.ZPowGate):
      return 0.0
    elif isinstance(gate, cirq.MeasurementGate):
      return 4000.0
    return 25.0

  def noisy_moment(self, moment: cirq.Moment, system_qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
    moments = []

    if validate_all_measurements(moment): ## add after measurement gate
      p00 = self._fidelity._p00
      p11 = self._fidelity._p11
      if p00 is not None and p11 is not None:
        p = p11 / (p00 + p11)
        gamma = p11 / p
#        print(p, gamma)
        moments.append(cirq.Moment(cirq.GeneralizedAmplitudeDampingChannel(p = p, gamma = gamma)(q) for q in system_qubits))
      elif p00 is not None:
        moments.append(cirq.Moment(cirq.GeneralizedAmplitudeDampingChannel(p = 0.0, gamma = p00)(q) for q in system_qubits))
      elif p11 is not None:
        moments.append(cirq.Moment(cirq.GeneralizedAmplitudeDampingChannel(p = 1.0, gamma = p11)(q) for q in system_qubits))

    moments.append(moment)
    if self._fidelity._pauli_error is not None: ## pauli error
      duration = max([self.get_duration(op.gate) for op in moment.operations])
      pauli_error = self._fidelity.pauli_error_from_depolarization(duration)

      _sq_inds = np.arange(4)
      pauli_inds = np.array(list(product(_sq_inds, repeat = 1)))
      num_inds = len(pauli_inds)
      pI = 1 - pauli_error # probability of identity matrix
      p_other = pauli_error / (num_inds - 1) # probability of other pauli gates
      pauli_probs = (np.array([p_other] * (num_inds - 1)))
      moments.append(cirq.Moment(cirq.depolarize(p_other)(q) for q in system_qubits))

    if self._fidelity._t1 is not None: # t1 decay noise
      duration = max([self.get_duration(op.gate) for op in moment.operations])
      moments.append(cirq.Moment(cirq.amplitude_damp(1 - np.exp(-duration / self._fidelity._t1)).on_each(system_qubits)))
    return moments

