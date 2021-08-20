import cirq_google
import numpy as np
from cirq.devices.noise_properties import NoiseProperties


def _xeb_fidelity_to_decay_constant(xeb_fidelity, num_qubits=2):
    # Converts from XEB Fidelity to depolarization decay constant
    if xeb_fidelity is not None:
        N = 2 ** num_qubits  # Dimension of Hilbert space
        return 1 - (1 - xeb_fidelity) / (1 - 1 / N)
    return None


def _rb_average_error_to_decay_constant(rb_average_error, num_qubits: int = 1):
    # Converts from randomized benchmarking average error to depolarization decay constant
    if rb_average_error is not None:
        N = 2 ** num_qubits  # Dimension of Hilbert space
        return 1 - rb_average_error / (1 - 1 / N)
    else:
        return None


def _rb_pauli_error_to_decay_constant(rb_pauli_error, num_qubits: int = 1):
    # Converts from randomized benchmarking pauli error to depolarization decay constant
    if rb_pauli_error is not None:
        N = 2 ** num_qubits  # Dimension of Hilbert space
        return 1 - rb_pauli_error / (1 - 1 / N ** 2)
    else:
        return None


def _within_tolerance(val_1, val_2, tolerance):
    # Helper function to check if two values are within tolerance
    if val_1 is None or val_2 is None:
        return True
    return abs(val_1 - val_2) <= tolerance


def _unpack_from_calibration(metric_name, calibration):
    # Gets the average (over all qubits) of each metric
    # TODO: Add support for per-qubit noise
    if metric_name in calibration.keys():
        return np.mean([value for qubit, value in calibration[metric_name].items()])
    else:
        return None


def noise_properties_from_calibration(
    calibration: cirq_google.Calibration, validate: bool = True, tolerance: float = 0.01
):
    """Translates between a Calibration object and a NoiseProperties object.
    The NoiseProperties object can then be used as input to the NoiseModelFromNoiseProperties
    class (cirq.devices.noise_properties) to create a NoiseModel that can be used with a simulator.

    If the validate argument is set to false, the depolarization decay constant will be calculated
    from the RB Pauli error if defined, the XEB Fidelity if RB Pauli error is not defined, or the
    RB Average error if the others are not defined.

    Args:
        calibration: a Calibration object with hardware metrics
        validate: whether or not to check that the depolarization decay constants calculated from
                 RB Pauli error, RB average error, & XEB Fidelity agree to within a given tolerance
        tolerance: threshold for validating decay constants frmo RB Pauli error, RB Average error,
                  and XEB fidelity.

    Raises:
        ValueError: decay constants from RB Average Error and RB Pauli Error aren't within tolerance

        ValueError: decay constants from RB Pauli Error and XEB Fidelity aren't within tolerance

        ValueError: decay constant from RB Pauli Error and XEB Fidelity aren't within tolerance
    """

    # Unpack all values from Calibration object
    t1_micros = _unpack_from_calibration('single_qubit_idle_t1_micros', calibration)
    t1_nanos = t1_micros * 1000 if t1_micros is not None else None
    xeb_error = _unpack_from_calibration('xeb', calibration)
    xeb_fidelity = 1 - xeb_error if xeb_error is not None else None
    rb_pauli_error = _unpack_from_calibration('single_qubit_rb_pauli_error_per_gate', calibration)
    rb_average_error = _unpack_from_calibration(
        'single_qubit_rb_average_error_per_gate', calibration
    )
    p00 = _unpack_from_calibration('single_qubit_p00_error', calibration)
    p11 = _unpack_from_calibration('single_qubit_p11_error', calibration)
    decay_constant_pauli = _rb_pauli_error_to_decay_constant(rb_pauli_error)

    decay_constant_average = _rb_average_error_to_decay_constant(rb_average_error)

    if validate:  # Will throw error if metrics aren't compatible
        if not _within_tolerance(decay_constant_pauli, decay_constant_average, tolerance):
            raise ValueError(
                f'Decay constant from RB Pauli error: {decay_constant_pauli}, '
                f'decay constant from RB Average error: {decay_constant_average}. '
                'If validation is disabled, RB Pauli error will be used.'
            )
        decay_constant_from_xeb = _xeb_fidelity_to_decay_constant(xeb_fidelity)
        if not _within_tolerance(decay_constant_from_xeb, decay_constant_pauli, tolerance):
            raise ValueError(
                f'Decay constant from RB Pauli error: {decay_constant_pauli}, '
                f'decay constant from XEB Fidelity: {decay_constant_from_xeb}. '
                'If validation is disabled, RB Pauli error will be used.'
            )
        if not _within_tolerance(decay_constant_from_xeb, decay_constant_average, tolerance):
            raise ValueError(
                f'Decay constant from RB Average error: {decay_constant_average}, '
                f'decay constant from XEB Fidelity: {decay_constant_from_xeb}. '
                'If validation is disabled, XEB Fidelity will be used.'
            )

    if decay_constant_pauli is not None:  # can't define both decay constant and xeb
        return NoiseProperties(
            t1_ns=t1_nanos, decay_constant=decay_constant_pauli, p00=p00, p11=p11
        )
    if xeb_fidelity is not None:
        return NoiseProperties(t1_ns=t1_nanos, xeb_fidelity=xeb_fidelity, p00=p00, p11=p11)
    return NoiseProperties(t1_ns=t1_nanos, decay_constant=decay_constant_average, p00=p00, p11=p11)
