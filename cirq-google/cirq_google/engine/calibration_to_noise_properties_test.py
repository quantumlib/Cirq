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

import cirq, cirq_google
from cirq.devices.noise_utils import OpIdentifier

from google.protobuf.text_format import Merge
import numpy as np
import pytest


def get_mock_calibration(
    pauli_error,
    incoherent_error,
    p00_error,
    p11_error,
    t1_micros,
    syc_pauli,
    iswap_pauli,
    syc_angles,
    iswap_angles,
) -> cirq_google.Calibration:
    _CALIBRATION_DATA = Merge(
        f"""
    timestamp_ms: 1579214873,
    metrics: [{{
        name: 'single_qubit_rb_pauli_error_per_gate',
        targets: ['0_0'],
        values: [{{
            double_val: {pauli_error[0]}
        }}]
    }}, {{
        name: 'single_qubit_rb_pauli_error_per_gate',
        targets: ['0_1'],
        values: [{{
            double_val:{pauli_error[1]}
        }}]
    }}, {{
        name: 'single_qubit_rb_pauli_error_per_gate',
        targets: ['1_0'],
        values: [{{
            double_val:{pauli_error[2]}
        }}]
    }}, {{
        name: 'single_qubit_rb_incoherent_error_per_gate',
        targets: ['0_0'],
        values: [{{
            double_val: {incoherent_error[0]}
        }}]
    }}, {{
        name: 'single_qubit_rb_incoherent_error_per_gate',
        targets: ['0_1'],
        values: [{{
            double_val:{incoherent_error[1]}
        }}]
    }}, {{
        name: 'single_qubit_rb_incoherent_error_per_gate',
        targets: ['1_0'],
        values: [{{
            double_val:{incoherent_error[2]}
        }}]
    }}, {{
        name: 'single_qubit_p00_error',
        targets: ['0_0'],
        values: [{{
            double_val: {p00_error[0]}
        }}]
    }}, {{
        name: 'single_qubit_p00_error',
        targets: ['0_1'],
        values: [{{
            double_val: {p00_error[1]}
        }}]
    }}, {{
        name: 'single_qubit_p00_error',
        targets: ['1_0'],
        values: [{{
            double_val: {p00_error[2]}
        }}]
    }}, {{
        name: 'single_qubit_p11_error',
        targets: ['0_0'],
        values: [{{
            double_val: {p11_error[0]}
        }}]
    }}, {{
        name: 'single_qubit_p11_error',
        targets: ['0_1'],
        values: [{{
            double_val: {p11_error[1]}
        }}]
    }}, {{
        name: 'single_qubit_p11_error',
        targets: ['1_0'],
        values: [{{
            double_val: {p11_error[2]}
        }}]
    }}, {{
        name: 'single_qubit_idle_t1_micros',
        targets: ['0_0'],
        values: [{{
            double_val: {t1_micros[0]}
        }}]
    }}, {{
        name: 'single_qubit_idle_t1_micros',
        targets: ['0_1'],
        values: [{{
            double_val: {t1_micros[1]}
        }}]
    }}, {{
        name: 'single_qubit_idle_t1_micros',
        targets: ['1_0'],
        values: [{{
            double_val: {t1_micros[2]}
        }}]
    }}, {{
        name: 'two_qubit_parallel_sycamore_gate_xeb_pauli_error_per_cycle',
        targets: ['0_0', '0_1'],
        values: [{{
            double_val: {syc_pauli[0]}
        }}]
    }}, {{
        name: 'two_qubit_parallel_sycamore_gate_xeb_pauli_error_per_cycle',
        targets: ['0_0', '1_0'],
        values: [{{
            double_val: {syc_pauli[1]}
        }}]
    }}, {{
        name: 'two_qubit_parallel_sqrt_iswap_gate_xeb_pauli_error_per_cycle',
        targets: ['0_0', '0_1'],
        values: [{{
            double_val: {iswap_pauli[0]}
        }}]
    }}, {{
        name: 'two_qubit_parallel_sqrt_iswap_gate_xeb_pauli_error_per_cycle',
        targets: ['0_0', '1_0'],
        values: [{{
            double_val: {iswap_pauli[1]}
        }}]
    }}, {{
        name: 'two_qubit_parallel_sycamore_gate_xeb_entangler_theta_error_per_cycle',
        targets: ['0_0', '0_1'],
        values: [{{
            double_val: {syc_angles[0].theta}
        }}]
    }}, {{
        name: 'two_qubit_parallel_sycamore_gate_xeb_entangler_theta_error_per_cycle',
        targets: ['0_0', '1_0'],
        values: [{{
            double_val: {syc_angles[1].theta}
        }}]
    }}, {{
        name: 'two_qubit_parallel_sqrt_iswap_gate_xeb_entangler_theta_error_per_cycle',
        targets: ['0_0', '0_1'],
        values: [{{
            double_val: {iswap_angles[0].theta}
        }}]
    }}, {{
        name: 'two_qubit_parallel_sqrt_iswap_gate_xeb_entangler_theta_error_per_cycle',
        targets: ['0_0', '1_0'],
        values: [{{
            double_val: {iswap_angles[1].theta}
        }}]
    }}, {{
        name: 'two_qubit_parallel_sycamore_gate_xeb_entangler_phi_error_per_cycle',
        targets: ['0_0', '0_1'],
        values: [{{
            double_val: {syc_angles[0].phi}
        }}]
    }}, {{
        name: 'two_qubit_parallel_sycamore_gate_xeb_entangler_phi_error_per_cycle',
        targets: ['0_0', '1_0'],
        values: [{{
            double_val: {syc_angles[1].phi}
        }}]
    }}, {{
        name: 'two_qubit_parallel_sqrt_iswap_gate_xeb_entangler_phi_error_per_cycle',
        targets: ['0_0', '0_1'],
        values: [{{
            double_val: {iswap_angles[0].phi}
        }}]
    }}, {{
        name: 'two_qubit_parallel_sqrt_iswap_gate_xeb_entangler_phi_error_per_cycle',
        targets: ['0_0', '1_0'],
        values: [{{
            double_val: {iswap_angles[1].phi}
        }}]
    }}]
""",
        cirq_google.api.v2.metrics_pb2.MetricsSnapshot(),
    )

    return cirq_google.Calibration(_CALIBRATION_DATA)


def test_noise_properties_from_calibration():
    qubits = [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)]
    pauli_error = [0.001, 0.002, 0.003]
    incoherent_error = [0.0001, 0.0002, 0.0003]
    p00_error = [0.004, 0.005, 0.006]
    p11_error = [0.007, 0.008, 0.009]
    t1_micros = [10, 20, 30]
    syc_pauli = [0.01, 0.02]
    iswap_pauli = [0.03, 0.04]
    syc_angles = [
        cirq.PhasedFSimGate(theta=0.011, phi=-0.021),
        cirq.PhasedFSimGate(theta=-0.012, phi=0.022),
    ]
    iswap_angles = [
        cirq.PhasedFSimGate(theta=-0.013, phi=0.023),
        cirq.PhasedFSimGate(theta=0.014, phi=-0.024),
    ]

    # Create NoiseProperties object from Calibration
    calibration = get_mock_calibration(
        pauli_error,
        incoherent_error,
        p00_error,
        p11_error,
        t1_micros,
        syc_pauli,
        iswap_pauli,
        syc_angles,
        iswap_angles,
    )
    prop = cirq_google.noise_properties_from_calibration(calibration)

    for i, q in enumerate(qubits):
        assert np.isclose(
            prop.gate_pauli_errors[OpIdentifier(cirq.PhasedXZGate, q)], pauli_error[i]
        )
        assert np.allclose(prop.readout_errors[q], np.array([p00_error[i], p11_error[i]]))
        assert np.isclose(prop.t1_ns[q], t1_micros[i] * 1000)
        microwave_time_ns = 25.0
        tphi_err = incoherent_error[i] - microwave_time_ns / (3 * prop.t1_ns[q])
        if tphi_err > 0:
            tphi_ns = microwave_time_ns / (3 * tphi_err)
        else:
            tphi_ns = 1e10
        assert prop.tphi_ns[q] == tphi_ns

    qubit_pairs = [(qubits[0], qubits[1]), (qubits[0], qubits[2])]
    for i, qs in enumerate(qubit_pairs):
        for gate, values in [
            (cirq_google.SycamoreGate, syc_pauli),
            (cirq.ISwapPowGate, iswap_pauli),
        ]:
            assert np.isclose(prop.gate_pauli_errors[OpIdentifier(gate, *qs)], values[i])
            assert np.isclose(prop.gate_pauli_errors[OpIdentifier(gate, *qs[::-1])], values[i])
            assert np.isclose(prop.gate_pauli_errors[OpIdentifier(gate, *qs)], values[i])
            assert np.isclose(prop.gate_pauli_errors[OpIdentifier(gate, *qs[::-1])], values[i])

        for gate, values in [
            (cirq_google.SycamoreGate, syc_angles),
            (cirq.ISwapPowGate, iswap_angles),
        ]:
            assert prop.fsim_errors[OpIdentifier(gate, *qs)] == values[i]
            assert prop.fsim_errors[OpIdentifier(gate, *qs[::-1])] == values[i]
            assert prop.fsim_errors[OpIdentifier(gate, *qs)] == values[i]
            assert prop.fsim_errors[OpIdentifier(gate, *qs[::-1])] == values[i]


def test_zphase_data():
    qubits = [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)]
    pauli_error = [0.001, 0.002, 0.003]
    incoherent_error = [0.0001, 0.0002, 0.0003]
    p00_error = [0.004, 0.005, 0.006]
    p11_error = [0.007, 0.008, 0.009]
    t1_micros = [10, 20, 30]
    syc_pauli = [0.01, 0.02]
    iswap_pauli = [0.03, 0.04]
    syc_angles = [
        cirq.PhasedFSimGate(theta=0.011, phi=-0.021, zeta=-0.031, gamma=0.043),
        cirq.PhasedFSimGate(theta=-0.012, phi=0.022, zeta=0.032, gamma=-0.044),
    ]
    iswap_angles = [
        cirq.PhasedFSimGate(theta=-0.013, phi=0.023, zeta=0.031, gamma=-0.043),
        cirq.PhasedFSimGate(theta=0.014, phi=-0.024, zeta=-0.032, gamma=0.044),
    ]

    # Create NoiseProperties object from Calibration
    calibration = get_mock_calibration(
        pauli_error,
        incoherent_error,
        p00_error,
        p11_error,
        t1_micros,
        syc_pauli,
        iswap_pauli,
        syc_angles,
        iswap_angles,
    )

    qubit_pairs = [(qubits[0], qubits[1]), (qubits[0], qubits[2])]
    zphase_data = {
        "syc": {
            "zeta": {qubit_pairs[0]: syc_angles[0].zeta, qubit_pairs[1]: syc_angles[1].zeta},
            "gamma": {qubit_pairs[0]: syc_angles[0].gamma, qubit_pairs[1]: syc_angles[1].gamma},
        },
        "sqrt_iswap": {
            "zeta": {qubit_pairs[0]: iswap_angles[0].zeta, qubit_pairs[1]: iswap_angles[1].zeta},
            "gamma": {qubit_pairs[0]: iswap_angles[0].gamma, qubit_pairs[1]: iswap_angles[1].gamma},
        },
    }

    prop = cirq_google.noise_properties_from_calibration(calibration, zphase_data)
    for i, qs in enumerate(qubit_pairs):
        for gate, values in [
            (cirq_google.SycamoreGate, syc_angles),
            (cirq.ISwapPowGate, iswap_angles),
        ]:
            assert prop.fsim_errors[OpIdentifier(gate, *qs)] == values[i]
            assert prop.fsim_errors[OpIdentifier(gate, *qs[::-1])] == values[i]
            assert prop.fsim_errors[OpIdentifier(gate, *qs)] == values[i]
            assert prop.fsim_errors[OpIdentifier(gate, *qs[::-1])] == values[i]


def test_incomplete_calibration():
    pauli_error = [0.001, 0.002, 0.003]
    p00_error = [0.004, 0.005, 0.006]
    p11_error = [0.007, 0.008, 0.009]
    t1_micros = [10, 20, 30]

    _CALIBRATION_DATA = Merge(
        f"""
    timestamp_ms: 1579214873,
    metrics: [{{
        name: 'single_qubit_rb_pauli_error_per_gate',
        targets: ['0_0'],
        values: [{{
            double_val: {pauli_error[0]}
        }}]
    }}, {{
        name: 'single_qubit_rb_pauli_error_per_gate',
        targets: ['0_1'],
        values: [{{
            double_val:{pauli_error[1]}
        }}]
    }}, {{
        name: 'single_qubit_rb_pauli_error_per_gate',
        targets: ['1_0'],
        values: [{{
            double_val:{pauli_error[2]}
        }}]
    }}, {{
        name: 'single_qubit_p00_error',
        targets: ['0_0'],
        values: [{{
            double_val: {p00_error[0]}
        }}]
    }}, {{
        name: 'single_qubit_p00_error',
        targets: ['0_1'],
        values: [{{
            double_val: {p00_error[1]}
        }}]
    }}, {{
        name: 'single_qubit_p00_error',
        targets: ['1_0'],
        values: [{{
            double_val: {p00_error[2]}
        }}]
    }}, {{
        name: 'single_qubit_p11_error',
        targets: ['0_0'],
        values: [{{
            double_val: {p11_error[0]}
        }}]
    }}, {{
        name: 'single_qubit_p11_error',
        targets: ['0_1'],
        values: [{{
            double_val: {p11_error[1]}
        }}]
    }}, {{
        name: 'single_qubit_p11_error',
        targets: ['1_0'],
        values: [{{
            double_val: {p11_error[2]}
        }}]
    }}, {{
        name: 'single_qubit_idle_t1_micros',
        targets: ['0_0'],
        values: [{{
            double_val: {t1_micros[0]}
        }}]
    }}, {{
        name: 'single_qubit_idle_t1_micros',
        targets: ['0_1'],
        values: [{{
            double_val: {t1_micros[1]}
        }}]
    }}, {{
        name: 'single_qubit_idle_t1_micros',
        targets: ['1_0'],
        values: [{{
            double_val: {t1_micros[2]}
        }}]
    }}]
""",
        cirq_google.api.v2.metrics_pb2.MetricsSnapshot(),
    )

    # Create NoiseProperties object from Calibration
    calibration = cirq_google.Calibration(_CALIBRATION_DATA)
    with pytest.raises(ValueError, match='Keys specified for T1 and Tphi are not identical.'):
        _ = cirq_google.noise_properties_from_calibration(calibration)
