import warnings
from typing import Sequence, TYPE_CHECKING, List
from itertools import product
from cirq import ops, protocols, devices
import numpy as np

if TYPE_CHECKING:
    from typing import Iterable
    import cirq


class NoiseProperties:
    def __init__(
        self,
        *,
        t1_ns: float = None,
        decay_constant: float = None,
        xeb_fidelity: float = None,
        pauli_error: float = None,
        p00: float = None,
        p11: float = None,
    ) -> None:
        """Creates a NoiseProperties object using the provided metrics.

          Only one of decay_constant, xeb_fidelity, and pauli_error should be specified.

        Args:
          t1_ns: t1 decay constant in ns
          decay_constant: depolarization decay constant
          xeb_fidelity: 2-qubit XEB Fidelity
          pauli_error: total Pauli error
          p00: probability of qubit initialized as zero being measured as zero
          p11: probability of qubit initialized as one being measured as one

        Raises:
          ValueError: if no metrics are specified
          ValueError: if xeb fidelity, pauli error, p00, or p00 are less than 0 or greater than 1
          ValueError: if more than one of pauli error, xeb fidelity, or decay constant is specified
        """
        if not any([t1_ns, decay_constant, xeb_fidelity, pauli_error, p00, p11]):
            raise ValueError('At least one metric must be specified')

        for metric in [xeb_fidelity, pauli_error, p00, p11]:
            if metric is not None and not 0.0 <= metric <= 1.0:
                raise ValueError('xeb, pauli error, p00, and p11 must be between 0 and 1')

        if (
            np.count_nonzero(
                [metric is not None for metric in [xeb_fidelity, pauli_error, decay_constant]]
            )
            > 1
        ):
            raise ValueError(
                'Only one of xeb fidelity, pauli error, or decay constant should be defined'
            )

        self._t1_ns = t1_ns
        self._p = decay_constant
        self._p00 = p00
        self._p11 = p11

        if pauli_error is not None:
            self._p = self.pauli_error_to_decay_constant(pauli_error)
        elif xeb_fidelity is not None:
            self._p = self.xeb_fidelity_to_decay_constant(xeb_fidelity)

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
        return self.decay_constant_to_pauli_error()

    @property
    def t1_ns(self):
        return self._t1_ns

    @property
    def xeb(self):
        return self.decay_constant_to_xeb_fidelity()

    def decay_constant_to_xeb_fidelity(self, num_qubits: int = 2):
        """Calculates the XEB fidelity from the depolarization decay constant.

        Args:
            num_qubits: number of qubits
        """
        if self._p is not None:
            N = 2 ** num_qubits
            return 1 - ((1 - self._p) * (1 - 1 / N))
        return None

    def decay_constant_to_pauli_error(self, num_qubits: int = 1):
        """Calculates pauli error from the depolarization decay constant.
        Args:
            num_qubits: number of qubits
        """
        if self._p is not None:
            N = 2 ** num_qubits
            return (1 - self._p) * (1 - 1 / N / N)
        return None

    def pauli_error_to_decay_constant(self, pauli_error: float, num_qubits: int = 1):
        """Calculates depolarization decay constant from pauli error.

        Args:
            pauli_error: The pauli error
            num_qubits: Number of qubits
        """
        N = 2 ** num_qubits
        return 1 - (pauli_error / (1 - 1 / N / N))

    def xeb_fidelity_to_decay_constant(self, xeb_fidelity: float, num_qubits: int = 2):
        """Calculates the depolarization decay constant from the XEB noise_properties.

        Args:
            xeb_fidelity: The XEB noise_properties
            num_qubits: Number of qubits
        """
        N = 2 ** num_qubits
        return 1 - (1 - xeb_fidelity) / (1 - 1 / N)

    def pauli_error_from_t1(self, t: float, t1_ns: float):
        """Calculates the pauli error from amplitude damping.
        Unlike the other methods, this computes a specific case (over time t).

        Args:
            t: the duration of the gate
            t1_ns: the t1 decay constant in ns
        """
        t2 = 2 * t1_ns
        return (1 - np.exp(-t / t2)) / 2 + (1 - np.exp(-t / t1_ns)) / 4

    def pauli_error_from_depolarization(self, t: float):
        """Calculates the amount of pauli error from depolarization.
        Unlike the other methods, this computes a specific case (over time t).

        If pauli error from t1 decay is more than total pauli error, just return the pauli error.

        Args:
            t: the duration of the gate
        """
        if self.t1_ns is not None:
            pauli_error_from_t1 = self.pauli_error_from_t1(t, self.t1_ns)
            if self.pauli_error >= pauli_error_from_t1:
                return self.pauli_error - pauli_error_from_t1
            else:
                warnings.warn(
                    "Pauli error from T1 decay is greater than total Pauli error", RuntimeWarning
                )
        return self.pauli_error

    def average_error(self, num_qubits: int = 1):
        """Calculates the average error from the depolarization decay constant.

        Args:
            num_qubits: the number of qubits
        """
        if self._p is not None:
            N = 2 ** num_qubits
            return (1 - self._p) * (1 - 1 / N)
        return None


def get_duration_ns(gate):
    # Gate durations based on sycamore durations.
    # TODO: pull the gate durations from cirq_google
    # or allow users to pass them in
    if isinstance(gate, ops.FSimGate):
        theta, _ = gate._value_equality_values_()
        if np.abs(theta) % (np.pi / 2) == 0:
            return 12.0
        return 32.0
    elif isinstance(gate, ops.ISwapPowGate):
        return 32.0
    elif isinstance(gate, ops.ZPowGate):
        return 0.0
    elif isinstance(gate, ops.MeasurementGate):
        return 4000.0
    elif isinstance(gate, ops.WaitGate):
        return gate.duration.total_nanos()
    return 25.0


def _apply_readout_noise(p00, p11, moments, measurement_qubits):
    if p00 is None:
        p = 1.0
        gamma = p11
    elif p11 is None:
        p = 0.0
        gamma = p00
    else:
        p = p11 / (p00 + p11)
        gamma = p11 / p
    moments.append(
        ops.Moment(
            ops.GeneralizedAmplitudeDampingChannel(p=p, gamma=gamma)(q) for q in measurement_qubits
        )
    )


def _apply_depol_noise(pauli_error, moments, system_qubits):

    _sq_inds = np.arange(4)
    pauli_inds = np.array(list(product(_sq_inds, repeat=1)))
    num_inds = len(pauli_inds)
    p_other = pauli_error / (num_inds - 1)  # probability of X, Y, Z gates
    moments.append(ops.Moment(ops.depolarize(p_other)(q) for q in system_qubits))


def _apply_amplitude_damp_noise(duration, t1, moments, system_qubits):
    moments.append(
        ops.Moment(ops.amplitude_damp(1 - np.exp(-duration / t1)).on_each(system_qubits))
    )


class NoiseModelFromNoiseProperties(devices.NoiseModel):
    def __init__(self, noise_properties: NoiseProperties) -> None:
        """Creates a Noise Model from a NoiseProperties object that can be used with a Simulator.

        Args:
            noise_properties: the NoiseProperties object to be converted to a Noise Model.

        Raises:
            ValueError: if no NoiseProperties object is specified.
        """
        if noise_properties is not None:
            self._noise_properties = noise_properties
        else:
            raise ValueError('A NoiseProperties object must be specified')

    def noisy_moment(
        self, moment: ops.Moment, system_qubits: Sequence['cirq.Qid']
    ) -> 'cirq.OP_TREE':
        moments: List[ops.Moment] = []

        if any(
            [protocols.is_measurement(op.gate) for op in moment.operations]
        ):  # Add readout error before measurement gate
            p00 = self._noise_properties.p00
            p11 = self._noise_properties.p11
            measurement_qubits = [
                list(op.qubits)[0] for op in moment.operations if protocols.is_measurement(op.gate)
            ]
            if p00 is not None or p11 is not None:
                _apply_readout_noise(p00, p11, moments, measurement_qubits)
            moments.append(moment)
        else:
            moments.append(moment)
        if self._noise_properties.pauli_error is not None:  # Add depolarization error#
            duration = max([get_duration_ns(op.gate) for op in moment.operations])
            pauli_error = self._noise_properties.pauli_error_from_depolarization(duration)
            _apply_depol_noise(pauli_error, moments, system_qubits)

        if self._noise_properties.t1_ns is not None:  # Add amplitude damping noise
            duration = max([get_duration_ns(op.gate) for op in moment.operations])
            _apply_amplitude_damp_noise(
                duration, self._noise_properties.t1_ns, moments, system_qubits
            )
        return moments
