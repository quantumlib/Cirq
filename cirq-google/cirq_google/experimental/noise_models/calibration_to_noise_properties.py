# Copyright 2021 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Optional, Tuple
import cirq, cirq_google
import numpy as np
from cirq.devices.superconducting_qubits_noise_properties import (
    SuperconductingQubitsNoiseProperties,
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


def _unpack_1q_from_calibration(
    metric_name: str, calibration: cirq_google.Calibration
) -> Dict[cirq.Qid, float]:
    """Converts a single-qubit metric from Calibration to dict format."""
    if metric_name not in calibration:
        return {}
    return {
        cirq_google.Calibration.key_to_qubit(key): cirq_google.Calibration.value_to_float(val)
        for key, val in calibration[metric_name].items()
    }


def _unpack_2q_from_calibration(
    metric_name: str, calibration: cirq_google.Calibration
) -> Dict[Tuple[cirq.Qid, ...], float]:
    """Converts a two-qubit metric from Calibration to dict format."""
    if metric_name not in calibration:
        return {}
    return {
        cirq_google.Calibration.key_to_qubits(key): cirq_google.Calibration.value_to_float(val)
        for key, val in calibration[metric_name].items()
    }


def noise_properties_from_calibration(
    calibration: cirq_google.Calibration,
) -> SuperconductingQubitsNoiseProperties:
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
    T1_micros = _unpack_1q_from_calibration('single_qubit_idle_t1_micros', calibration)
    t1_ns = {q: T1_micro * 1000 for q, T1_micro in T1_micros.items()}

    # 2. Extract Tphi for all qubits
    rb_incoherent_errors = _unpack_1q_from_calibration(
        'single_qubit_rb_incoherent_error_per_gate', calibration
    )
    tphi_ns = {}
    if rb_incoherent_errors:
        microwave_time_ns = DEFAULT_GATE_NS[cirq.PhasedXZGate]
        for qubit, q_t1_ns in t1_ns.items():
            tphi_err = rb_incoherent_errors[qubit] - microwave_time_ns / (3 * q_t1_ns)
            if tphi_err > 0:
                q_tphi_ns = microwave_time_ns / (3 * tphi_err)
            else:
                q_tphi_ns = 1e10
            tphi_ns[qubit] = q_tphi_ns

    # 3a. Extract Pauli error for single-qubit gates.
    rb_pauli_errors = _unpack_1q_from_calibration(
        'single_qubit_rb_pauli_error_per_gate', calibration
    )
    gate_pauli_errors = {
        OpIdentifier(gate, q): pauli_err
        for q, pauli_err in rb_pauli_errors.items()
        for gate in SINGLE_QUBIT_GATES
    }

    # 3b. Extract Pauli error for two-qubit gates.
    tq_iswap_pauli_error = _unpack_2q_from_calibration(
        'two_qubit_parallel_sqrt_iswap_gate_xeb_pauli_error_per_cycle', calibration
    )
    gate_pauli_errors.update(
        {
            k: v
            for qs, pauli_err in tq_iswap_pauli_error.items()
            for k, v in {
                OpIdentifier(cirq.ISwapPowGate, *qs): pauli_err,
                OpIdentifier(cirq.ISwapPowGate, *qs[::-1]): pauli_err,
            }.items()
        }
    )

    # 4. Extract readout fidelity for all qubits.
    p00 = _unpack_1q_from_calibration('single_qubit_p00_error', calibration)
    p11 = _unpack_1q_from_calibration('single_qubit_p11_error', calibration)
    ro_fidelities = {
        q: np.array([p00.get(q, 0), p11.get(q, 0)]) for q in set(p00.keys()) | set(p11.keys())
    }

    # TODO: include entangling errors once provided by QCS.
    # These must also be accounted for in Pauli error.

    return SuperconductingQubitsNoiseProperties(
        gate_times_ns=DEFAULT_GATE_NS,
        t1_ns=t1_ns,
        tphi_ns=tphi_ns,
        ro_fidelities=ro_fidelities,
        gate_pauli_errors=gate_pauli_errors,
    )
