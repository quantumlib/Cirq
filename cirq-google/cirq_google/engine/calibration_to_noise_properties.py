# Copyright 2022 The Cirq Developers
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


"""Tools for converting Calibrations to NoiseProperties.

Given a Calibration "cal", a user can simulate noise approximating that
calibration using the following pipeline:

    >>> cal = cirq_google.engine.load_median_device_calibration("rainbow")
    >>> noise_props = cirq_google.engine.noise_properties_from_calibration(cal)
    >>> noise_model = cirq_google.NoiseModelFromGoogleNoiseProperties(noise_props)
    >>> simulator = cirq.Simulator(noise=noise_model)
    >>> circuit = cirq.Circuit(cirq.X(cirq.GridQubit(5, 2)))
    >>> result = simulator.simulate(circuit)
"""

from typing import Dict, Optional, Tuple, Type, TYPE_CHECKING

from cirq import ops
from cirq.devices import noise_utils
from cirq_google import engine
from cirq_google import ops as cg_ops
from cirq_google.devices import google_noise_properties
from cirq_google.engine import util

if TYPE_CHECKING:
    import cirq


# TODO: acquire this based on the target device.
# Default map of gates to their durations.
DEFAULT_GATE_NS: Dict[Type['cirq.Gate'], float] = {
    ops.ZPowGate: 25.0,
    ops.MeasurementGate: 4000.0,
    ops.ResetChannel: 250.0,
    ops.PhasedXZGate: 25.0,
    ops.FSimGate: 32.0,
    ops.ISwapPowGate: 32.0,
    ops.CZPowGate: 32.0,
    cg_ops.SycamoreGate: 12.0,
    # ops.WaitGate is a special case.
}
GATE_PREFIX_PAIRS: Dict[Type['cirq.Gate'], str] = {
    cg_ops.SycamoreGate: 'two_qubit_parallel_sycamore_gate',
    ops.ISwapPowGate: 'two_qubit_parallel_sqrt_iswap_gate',
}
GATE_ZPHASE_CODE_PAIRS: Dict[Type['cirq.Gate'], str] = {
    cg_ops.SycamoreGate: 'syc',
    ops.ISwapPowGate: 'sqrt_iswap',
}


def _unpack_1q_from_calibration(
    metric_name: str, calibration: engine.Calibration
) -> Dict['cirq.Qid', float]:
    """Converts a single-qubit metric from Calibration to dict format."""
    if metric_name not in calibration:
        return {}
    return {
        engine.Calibration.key_to_qubit(key): engine.Calibration.value_to_float(val)
        for key, val in calibration[metric_name].items()
    }


def _unpack_2q_from_calibration(
    metric_name: str, calibration: engine.Calibration
) -> Dict[Tuple['cirq.Qid', ...], float]:
    """Converts a two-qubit metric from Calibration to dict format."""
    if metric_name not in calibration:
        return {}
    return {
        engine.Calibration.key_to_qubits(key): engine.Calibration.value_to_float(val)
        for key, val in calibration[metric_name].items()
    }


def noise_properties_from_calibration(
    calibration: engine.Calibration,
    zphase_data: Optional[util.ZPhaseDataType] = None,
    gate_times_ns: Optional[Dict[Type['cirq.Gate'], float]] = None,
) -> google_noise_properties.GoogleNoiseProperties:
    """Translates between `cirq_google.Calibration` and NoiseProperties.

    The NoiseProperties object can then be used as input to the
    `cirq_google.NoiseModelFromGoogleNoiseProperties` class to create a
    `cirq.NoiseModel` that can be used with a simulator.

    To manually override noise properties, call `with_params` on the output:

        >>> cal = cirq_google.engine.load_median_device_calibration("rainbow")
        >>> # noise_props with all gate durations set to 37ns.
        >>> noise_props = cirq_google.engine.noise_properties_from_calibration(cal).with_params(
        ...     gate_times_ns=37)

    See `cirq_google.GoogleNoiseProperties` for details.

    Args:
        calibration: a Calibration object with hardware metrics.
        zphase_data: Optional data for Z phases not captured by Calibration -
            specifically, zeta and gamma. These values require Floquet
            calibration and can be provided here if available.
        gate_times_ns: Map of gate durations in nanoseconds. If not provided,
            defaults to the Sycamore gate times listed in `known_devices.py`.

    Returns:
        A `cirq_google.GoogleNoiseProperties` which represents the error
        present in the given Calibration object.
    """
    if gate_times_ns is None:
        gate_times_ns = DEFAULT_GATE_NS

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
        microwave_time_ns = gate_times_ns[ops.PhasedXZGate]
        for qubit, q_t1_ns in t1_ns.items():
            tphi_err = rb_incoherent_errors[qubit] - microwave_time_ns / (3 * q_t1_ns)
            q_tphi_ns = 1e10 if tphi_err <= 0 else microwave_time_ns / (3 * tphi_err)
            tphi_ns[qubit] = q_tphi_ns

    # 3a. Extract Pauli error for single-qubit gates.
    rb_pauli_errors = _unpack_1q_from_calibration(
        'single_qubit_rb_pauli_error_per_gate', calibration
    )
    gate_pauli_errors = {
        noise_utils.OpIdentifier(gate, q): pauli_err
        for q, pauli_err in rb_pauli_errors.items()
        for gate in google_noise_properties.GoogleNoiseProperties.single_qubit_gates()
    }

    # 3b. Extract Pauli error for two-qubit gates.
    for gate, prefix in GATE_PREFIX_PAIRS.items():
        pauli_error = _unpack_2q_from_calibration(
            prefix + '_xeb_pauli_error_per_cycle', calibration
        )
        gate_pauli_errors.update(
            {
                k: v
                for qs, pauli_err in pauli_error.items()
                for k, v in {
                    noise_utils.OpIdentifier(gate, *qs): pauli_err,
                    noise_utils.OpIdentifier(gate, *qs[::-1]): pauli_err,
                }.items()
            }
        )

    # 4. Extract readout fidelity for all qubits.
    p00 = _unpack_1q_from_calibration('single_qubit_p00_error', calibration)
    p11 = _unpack_1q_from_calibration('single_qubit_p11_error', calibration)
    readout_errors = {q: [p00.get(q, 0), p11.get(q, 0)] for q in set(p00.keys()) | set(p11.keys())}

    # 5. Extract entangling angle errors.
    fsim_errors = {}
    for gate, prefix in GATE_PREFIX_PAIRS.items():
        theta_errors = _unpack_2q_from_calibration(
            prefix + '_xeb_entangler_theta_error_per_cycle', calibration
        )
        phi_errors = _unpack_2q_from_calibration(
            prefix + '_xeb_entangler_phi_error_per_cycle', calibration
        )
        gate_str = GATE_ZPHASE_CODE_PAIRS[gate]
        if zphase_data and gate_str in zphase_data:
            zeta_errors = zphase_data[gate_str]["zeta"]
            gamma_errors = zphase_data[gate_str]["gamma"]
        else:
            zeta_errors = {}
            gamma_errors = {}
        angle_keys = {
            *theta_errors.keys(),
            *phi_errors.keys(),
            *zeta_errors.keys(),
            *gamma_errors.keys(),
        }
        for qubits in angle_keys:
            theta = theta_errors.get(qubits, 0)
            phi = phi_errors.get(qubits, 0)
            zeta = zeta_errors.get(qubits, 0)
            gamma = gamma_errors.get(qubits, 0)
            op_id = noise_utils.OpIdentifier(gate, *qubits)
            error_gate = ops.PhasedFSimGate(theta=theta, phi=phi, zeta=zeta, gamma=gamma)
            fsim_errors[op_id] = error_gate
            op_id_reverse = noise_utils.OpIdentifier(gate, *qubits[::-1])
            fsim_errors[op_id_reverse] = error_gate

    # Known false positive: https://github.com/PyCQA/pylint/issues/5857
    return google_noise_properties.GoogleNoiseProperties(  # pylint: disable=unexpected-keyword-arg
        gate_times_ns=gate_times_ns,
        t1_ns=t1_ns,
        tphi_ns=tphi_ns,
        readout_errors=readout_errors,
        gate_pauli_errors=gate_pauli_errors,
        fsim_errors=fsim_errors,
    )
