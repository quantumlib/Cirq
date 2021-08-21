import cirq_google
import numpy as np
from cirq.devices.noise_properties import NoiseModelFromNoiseProperties
from cirq_google.experimental.noise_models.calibration_to_noise_properties import (
    noise_properties_from_calibration,
)
import cirq
from cirq_google.experimental.noise_models.compare_generated_noise_to_metrics import (
    compare_generated_noise_to_metrics,
)
from cirq_google.api import v2
from google.protobuf.text_format import Merge
import pandas as pd


def test_compare_generated_noise_to_metrics():
    xeb_1 = 0.001
    xeb_2 = 0.004

    p00_1 = 0.001
    p00_2 = 0.002
    p00_3 = 0.003

    p11_1 = 0.004
    p11_2 = 0.005
    p11_3 = 0.006

    t1_1 = 0.5  # microseconds
    t1_2 = 0.7
    t1_3 = 0.3

    _CALIBRATION_DATA = Merge(
        f"""
    timestamp_ms: 1579214873,
    metrics: [{{
        name: 'xeb',
        targets: ['0_0', '0_1'],
        values: [{{
            double_val: {xeb_1}
        }}]
    }}, {{
        name: 'xeb',
        targets: ['0_0', '1_0'],
        values: [{{
            double_val:{xeb_2}
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
        name: 'single_qubit_p11_error',
        targets: ['0_0'],
        values: [{{
            double_val: {p11_1}
        }}]
    }}, {{
        name: 'single_qubit_p11_error',
        targets: ['0_1'],
        values: [{{
            double_val: {p11_2}
        }}]
    }}, {{
        name: 'single_qubit_p11_error',
        targets: ['1_0'],
        values: [{{
            double_val: {p11_3}
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

    calibration = cirq_google.Calibration(_CALIBRATION_DATA)
    output_df = compare_generated_noise_to_metrics(calibration, seed=1)

    # TODO check against reasonable values one clear boundaries are established
