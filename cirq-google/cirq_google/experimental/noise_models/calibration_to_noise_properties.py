# pylint: disable=wrong-or-nonexistent-copyright-notice
from typing import Dict, Optional
import cirq, cirq_google
import numpy as np
from cirq.devices.noise_properties import (
    NoiseProperties,
    SINGLE_QUBIT_GATES,
)
from cirq.devices.noise_utils import (
    OpIdentifier,
)


def _within_tolerance(val_1: Optional[float], val_2: Optional[float], tolerance: float) -> bool:
    """Helper function to check if two values are within a given tolerance."""
    if val_1 is None or val_2 is None:
        return True
    return abs(val_1 - val_2) <= tolerance


def _unpack_from_calibration(
    metric_name: str, calibration: cirq_google.Calibration
) -> Dict[cirq.Qid, float]:
    """Converts a single-qubit metric from Calibration to dict format."""
    if metric_name not in calibration:
        return {}
    return {
        cirq_google.Calibration.key_to_qubit(key): cirq_google.Calibration.value_to_float(val)
        for key, val in calibration[metric_name].items()
    }


def noise_properties_from_calibration(calibration: cirq_google.Calibration) -> NoiseProperties:
    """Translates between a Calibration object and a NoiseProperties object.
    The NoiseProperties object can then be used as input to the NoiseModelFromNoiseProperties
    class (cirq.devices.noise_properties) to create a NoiseModel that can be used with a simulator.

    Args:
        calibration: a Calibration object with hardware metrics
    """

    # TODO: acquire this based on the target device.
    # Default map of gates to their durations.
    DEFAULT_GATE_NS: Dict[type, float] = {
        cirq.ZPowGate: 25.0,
        cirq.MeasurementGate: 4000.0,
        cirq.ResetChannel: 250.0,
        cirq.PhasedXZGate: 25.0,
        cirq.FSimGate: 32.0,
        cirq.ISwapPowGate: 32.0,
        cirq.CZPowGate: 32.0,
        # cirq.WaitGate is a special case.
    }

    # Unpack all values from Calibration object
    # 1. Extract T1 for all qubits
    T1_micros = _unpack_from_calibration('single_qubit_idle_t1_micros', calibration)
    T1_ns = {q: T1_micro * 1000 for q, T1_micro in T1_micros.items()}

    # 2. Extract Tphi for all qubits
    rb_incoherent_errors = _unpack_from_calibration(
        'single_qubit_rb_incoherent_error_per_gate', calibration
    )
    Tphi_ns = {}
    if rb_incoherent_errors:
        microwave_time_ns = DEFAULT_GATE_NS[cirq.PhasedXZGate]
        for qubit, t1_ns in T1_ns.items():
            tphi_err = rb_incoherent_errors[qubit] - microwave_time_ns / (3 * t1_ns)
            if tphi_err > 0:
                tphi_ns = microwave_time_ns / (3 * tphi_err)
            else:
                tphi_ns = 1e10
            Tphi_ns[qubit] = tphi_ns

    # 3a. Extract Pauli error for single-qubit gates.
    rb_pauli_errors = _unpack_from_calibration('single_qubit_rb_pauli_error_per_gate', calibration)
    gate_pauli_errors = {
        OpIdentifier(gate, q): pauli_err
        for q, pauli_err in rb_pauli_errors.items()
        for gate in SINGLE_QUBIT_GATES
    }

    # TODO: 3a. Extract Pauli error for two-qubit gates.

    # 4. Extract readout fidelity for all qubits.
    p00 = _unpack_from_calibration('single_qubit_p00_error', calibration)
    p11 = _unpack_from_calibration('single_qubit_p11_error', calibration)
    ro_fidelities = {
        q: np.array([p00.get(q, 0), p11.get(q, 0)]) for q in set(p00.keys()) | set(p11.keys())
    }

    # TODO: include entangling errors.

    return NoiseProperties(
        gate_times_ns=DEFAULT_GATE_NS,
        T1_ns=T1_ns,
        Tphi_ns=Tphi_ns,
        ro_fidelities=ro_fidelities,
        gate_pauli_errors=gate_pauli_errors,
    )
