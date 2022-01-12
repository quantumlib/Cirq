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

import cirq_google
from cirq_google.api import v2
from cirq_google.experimental.noise_models.compare_generated_noise_to_metrics import (
    compare_generated_noise_to_metrics,
)
from google.protobuf.text_format import Merge


def test_compare_generated_noise_to_metrics():
    # This test only verifies that the experiment succeeds without error.
    # Analysis of the experiment results is left to human reviewers.
    pauli_error = [0.001, 0.002, 0.003]
    incoherent_error = [0.0001, 0.0002, 0.0003]
    p00_error = [0.004, 0.005, 0.006]
    p11_error = [0.007, 0.008, 0.009]
    t1_micros = [10, 20, 30]
    syc_pauli = [0.01, 0.02]
    iswap_pauli = [0.03, 0.04]
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
    }}]
    """,
        v2.metrics_pb2.MetricsSnapshot(),
    )

    calibration = cirq_google.Calibration(_CALIBRATION_DATA)
    comparison = compare_generated_noise_to_metrics(calibration)
    print(comparison)
