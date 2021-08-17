import cirq_google
import numpy as np
from cirq.devices.noise_properties import NoiseProperties


def noise_properties_from_calibration(
    calibration: cirq_google.Calibration, validate: bool = True, tolerance: float = 0.01
):
    """Translates between a Calibration object and a NoiseProperties object1
    The NoiseProperties object can then be used to create a NoiseModel for a simulator.

    Args:
        calibration: a Calibration object with hardware metrics
    """

    def unpack_from_calibration(metric_name):
        # Gets the average (over all qubits) of each metric
        # TODO: Add support for per-qubit noise
        if metric_name in calibration.keys():
            return np.mean([value for qubit, value in calibration[metric_name].items()])
        else:
            return None

    def _xeb_fidelity_to_decay_constant(xeb_fidelity, num_qubits=2):
        if xeb_fidelity is not None:
            N = 2 ** num_qubits
            return 1 - (1 - xeb_fidelity) / (1 - 1 / N)
        return None

    def _within_tolerance(val_1, val_2):
        if val_1 is None or val_2 is None:
            return True
        return abs(val_1 - val_2) <= tolerance

    def validate_calibration(
        xeb_fidelity, decay_constant_from_rb_pauli, decay_constant_from_rb_average
    ):
        if not _within_tolerance(decay_constant_from_rb_pauli, decay_constant_from_rb_average):
            raise ValueError(
                'RB Pauli error and RB Average error are not compatible. If validation is disabled, RB Pauli error will be used.'
            )
        decay_constant_from_xeb = _xeb_fidelity_to_decay_constant(xeb_fidelity)
        if not _within_tolerance(decay_constant_from_xeb, decay_constant_from_rb_pauli):
            raise ValueError(
                'RB Pauli error and XEB Fidelity are not compatible. If validation is disabled, RB Pauli error will be used.'
            )
        if not _within_tolerance(decay_constant_from_xeb, decay_constant_from_rb_average):
            raise ValueError(
                'RB average error and XEB Fidelity are not compatible. If validation is disabled, XEB Fidelity will be used.'
            )

    def rb_average_error_to_decay_constant(rb_average_error, num_qubits: int = 1):
        # Converts from randomized benchmarking average error to depolarization decay constant
        if rb_average_error is not None:
            N = 2 ** num_qubits  # Dimension of Hilbert space
            return 1 - rb_average_error / (1 - 1 / N)
        else:
            return None

    def rb_pauli_error_to_decay_constant(rb_pauli_error, num_qubits: int = 1):
        # Converts from randomized benchmarking pauli error to depolarization decay constant
        if rb_pauli_error is not None:
            N = 2 ** num_qubits  # Dimension of Hilbert space
            return 1 - rb_pauli_error / (1 - 1 / N ** 2)
        else:
            return None

    t1_micros = unpack_from_calibration('single_qubit_idle_t1_micros')
    t1_nanos = t1_micros * 1000 if t1_micros is not None else None
    xeb_fidelity = unpack_from_calibration('xeb')
    rb_pauli_error = unpack_from_calibration('single_qubit_rb_pauli_error_per_gate')
    rb_average_error = unpack_from_calibration('single_qubit_rb_average_error_per_gate')
    p00 = unpack_from_calibration('single_qubit_p00_error')
    p11 = unpack_from_calibration('single_qubit_p11_error')
    decay_constant_pauli = rb_pauli_error_to_decay_constant(rb_pauli_error)

    decay_constant_average = rb_average_error_to_decay_constant(rb_average_error)

    if validate:
        validate_calibration(xeb_fidelity, decay_constant_pauli, decay_constant_average)

    if decay_constant_pauli is not None:  # can't define both decay constant and xeb
        return NoiseProperties(
            t1_ns=t1_nanos, decay_constant=decay_constant_pauli, p00=p00, p11=p11
        )
    if xeb_fidelity is not None:
        return NoiseProperties(t1_ns=t1_nanos, xeb_fidelity=xeb_fidelity, p00=p00, p11=p11)
    return NoiseProperties(t1_ns=t1_nanos, decay_constant=decay_constant_average, p00=p00, p11=p11)
