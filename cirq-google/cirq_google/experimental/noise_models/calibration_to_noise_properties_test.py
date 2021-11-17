# pylint: disable=wrong-or-nonexistent-copyright-notice
import pytest
import cirq_google
from cirq_google.api import v2
from cirq_google.experimental.noise_models.calibration_to_noise_properties import (
    noise_properties_from_calibration,
)
from google.protobuf.text_format import Merge
import numpy as np


def test_noise_properties_from_calibration():
    xeb_error_1 = 0.999
    xeb_error_2 = 0.996

    p00_1 = 0.001
    p00_2 = 0.002
    p00_3 = 0.003

    t1_1 = 0.005
    t1_2 = 0.007
    t1_3 = 0.003

    _CALIBRATION_DATA = Merge(
        f"""
    timestamp_ms: 1579214873,
    metrics: [{{
        name: 'xeb',
        targets: ['0_0', '0_1'],
        values: [{{
            double_val: {xeb_error_1}
        }}]
    }}, {{
        name: 'xeb',
        targets: ['0_0', '1_0'],
        values: [{{
            double_val:{xeb_error_2}
        }}]
    }}, {{
        name: 'single_qubit_p00_error',
        targets: ['0_0'],
        values: [{{
            double_val: {p00_1}
        }}]
    }}, {{
        name: 'single_qubit_p00_error',
        targets: ['0_1'],
        values: [{{
            double_val: {p00_2}
        }}]
    }}, {{
        name: 'single_qubit_p00_error',
        targets: ['1_0'],
        values: [{{
            double_val: {p00_3}
        }}]
    }}, {{
        name: 'single_qubit_readout_separation_error',
        targets: ['0_0'],
        values: [{{
            double_val: .004
        }}]
    }}, {{
        name: 'single_qubit_readout_separation_error',
        targets: ['0_1'],
        values: [{{
            double_val: .005
        }}]
    }},{{
        name: 'single_qubit_readout_separation_error',
        targets: ['1_0'],
        values: [{{
            double_val: .006
        }}]
    }}, {{
        name: 'single_qubit_idle_t1_micros',
        targets: ['0_0'],
        values: [{{
            double_val: {t1_1}
        }}]
    }}, {{
        name: 'single_qubit_idle_t1_micros',
        targets: ['0_1'],
        values: [{{
            double_val: {t1_2}
        }}]
    }}, {{
        name: 'single_qubit_idle_t1_micros',
        targets: ['1_0'],
        values: [{{
            double_val: {t1_3}
        }}]
    }}]
""",
        v2.metrics_pb2.MetricsSnapshot(),
    )

    # Create NoiseProperties object from Calibration
    calibration = cirq_google.Calibration(_CALIBRATION_DATA)
    prop = noise_properties_from_calibration(calibration)

    expected_t1_nanos = np.mean([t1_1, t1_2, t1_3]) * 1000
    expected_xeb_fidelity = 1 - np.mean([xeb_error_1, xeb_error_2])
    expected_p00 = np.mean([p00_1, p00_2, p00_3])

    assert np.isclose(prop.t1_ns, expected_t1_nanos)
    assert np.isclose(prop.xeb, expected_xeb_fidelity)
    assert np.isclose(prop.p00, expected_p00)


def test_from_calibration_rb():
    rb_pauli_1 = 0.001
    rb_pauli_2 = 0.002
    rb_pauli_3 = 0.003

    _CALIBRATION_DATA_RB = Merge(
        f"""
    timestamp_ms: 1579214873,
    metrics: [{{

        name: 'single_qubit_rb_pauli_error_per_gate',
        targets: ['0_0'],
        values: [{{
            double_val: {rb_pauli_1}
        }}]
    }}, {{
        name: 'single_qubit_rb_pauli_error_per_gate',
        targets: ['0_1'],
        values: [{{
            double_val: {rb_pauli_2}
        }}]
    }}, {{
        name: 'single_qubit_rb_pauli_error_per_gate',
        targets: ['1_0'],
        values: [{{
            double_val: {rb_pauli_3}
        }}]
     }}]
    """,
        v2.metrics_pb2.MetricsSnapshot(),
    )

    # Create NoiseProperties object from Calibration
    rb_calibration = cirq_google.Calibration(_CALIBRATION_DATA_RB)
    rb_noise_prop = noise_properties_from_calibration(rb_calibration)

    average_pauli_rb = np.mean([rb_pauli_1, rb_pauli_2, rb_pauli_3])
    assert np.isclose(average_pauli_rb, rb_noise_prop.pauli_error)


def test_validate_calibration():
    # RB Pauli error and RB Average Error disagree
    rb_pauli_error = 0.05
    rb_average_error = 0.1

    decay_constant_pauli = 1 - rb_pauli_error / (1 - 1 / 4)
    decay_constant_average = 1 - rb_average_error / (1 - 1 / 2)
    _CALIBRATION_DATA_PAULI_AVERAGE = Merge(
        f"""
    timestamp_ms: 1579214873,
    metrics: [{{

        name: 'single_qubit_rb_pauli_error_per_gate',
        targets: ['0_0'],
        values: [{{
            double_val: {rb_pauli_error}
        }}]
    }}, {{
        name: 'single_qubit_rb_average_error_per_gate',
        targets: ['0_1'],
        values: [{{
            double_val: {rb_average_error}
        }}]
     }}]
    """,
        v2.metrics_pb2.MetricsSnapshot(),
    )
    bad_calibration_pauli_average = cirq_google.Calibration(_CALIBRATION_DATA_PAULI_AVERAGE)
    with pytest.raises(
        ValueError,
        match=f'Decay constant from RB Pauli error: {decay_constant_pauli}, '
        f'decay constant from RB Average error: {decay_constant_average}. '
        'If validation is disabled, RB Pauli error will be used.',
    ):
        noise_properties_from_calibration(bad_calibration_pauli_average)

    assert np.isclose(
        noise_properties_from_calibration(
            bad_calibration_pauli_average, validate=False
        ).pauli_error,
        rb_pauli_error,
    )

    # RB Pauli Error and XEB Fidelity disagree
    xeb_fidelity = 0.99

    decay_constant_from_xeb = 1 - (1 - xeb_fidelity) / (1 - 1 / 4)

    _CALIBRATION_DATA_PAULI_XEB = Merge(
        f"""
    timestamp_ms: 1579214873,
    metrics: [{{

        name: 'single_qubit_rb_pauli_error_per_gate',
        targets: ['0_0'],
        values: [{{
            double_val: {rb_pauli_error}
        }}]
    }}, {{
        name: 'xeb',
        targets: ['0_0', '1_0'],
        values: [{{
            double_val:{1 - xeb_fidelity}
        }}]
     }}]
    """,
        v2.metrics_pb2.MetricsSnapshot(),
    )

    bad_calibration_pauli_xeb = cirq_google.Calibration(_CALIBRATION_DATA_PAULI_XEB)
    with pytest.raises(
        ValueError,
        match=f'Decay constant from RB Pauli error: {decay_constant_pauli}, '
        f'decay constant from XEB Fidelity: {decay_constant_from_xeb}. '
        'If validation is disabled, RB Pauli error will be used.',
    ):
        noise_properties_from_calibration(bad_calibration_pauli_xeb)

    # RB Average Error and XEB Fidelity disagree
    _CALIBRATION_DATA_AVERAGE_XEB = Merge(
        f"""
    timestamp_ms: 1579214873,
    metrics: [{{

        name: 'single_qubit_rb_average_error_per_gate',
        targets: ['0_0'],
        values: [{{
            double_val: {rb_average_error}
        }}]
    }}, {{
        name: 'xeb',
        targets: ['0_0', '1_0'],
        values: [{{
            double_val:{1 - xeb_fidelity}
        }}]
     }}]
    """,
        v2.metrics_pb2.MetricsSnapshot(),
    )

    bad_calibration_average_xeb = cirq_google.Calibration(_CALIBRATION_DATA_AVERAGE_XEB)
    with pytest.raises(
        ValueError,
        match=f'Decay constant from RB Average error: {decay_constant_average}, '
        f'decay constant from XEB Fidelity: {decay_constant_from_xeb}. '
        'If validation is disabled, XEB Fidelity will be used.',
    ):
        noise_properties_from_calibration(bad_calibration_average_xeb)

    assert np.isclose(
        noise_properties_from_calibration(bad_calibration_average_xeb, validate=False).xeb,
        xeb_fidelity,
    )

    # Calibration data with no RB error or XEB fidelity
    t1 = 2.0  # microseconds

    _CALIBRATION_DATA_T1 = Merge(
        f"""
    timestamp_ms: 1579214873,
    metrics: [{{
        name: 'single_qubit_idle_t1_micros',
        targets: ['0_0'],
        values: [{{
            double_val: {t1}
        }}]
    }}]
    """,
        v2.metrics_pb2.MetricsSnapshot(),
    )

    calibration_t1 = cirq_google.Calibration(_CALIBRATION_DATA_T1)

    assert np.isclose(noise_properties_from_calibration(calibration_t1).t1_ns, t1 * 1000)
