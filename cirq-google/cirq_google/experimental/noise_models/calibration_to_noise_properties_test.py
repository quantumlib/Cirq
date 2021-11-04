# pylint: disable=wrong-or-nonexistent-copyright-notice
import cirq
import cirq_google
from cirq_google.api import v2
from cirq_google.experimental.noise_models.calibration_to_noise_properties import (
    noise_properties_from_calibration,
)
from cirq.devices.noise_utils import (
    OpIdentifier,
)
from google.protobuf.text_format import Merge
import numpy as np


def test_noise_properties_from_calibration():
    qubits = [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)]
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
        v2.metrics_pb2.MetricsSnapshot(),
    )

    # Create NoiseProperties object from Calibration
    calibration = cirq_google.Calibration(_CALIBRATION_DATA)
    prop = noise_properties_from_calibration(calibration)

    for i, q in enumerate(qubits):
        assert np.isclose(
            prop.gate_pauli_errors[OpIdentifier(cirq.PhasedXZGate, q)], pauli_error[i]
        )
        assert np.allclose(prop.ro_fidelities[q], np.array([p00_error[i], p11_error[i]]))
        assert np.isclose(prop.T1_ns[q], t1_micros[i] * 1000)
        # TODO: test Tphi
