import cirq_google
import numpy as np
from cirq.devices.noise_properties import NoiseProperties, NoiseModelFromNoiseProperties
from cirq_google.experimental.noise_models.calibration_to_noise_properties import noise_properties_from_calibration
import cirq
from cirq_google.experimental.noise_models.validate_noise_model_from_calibration import validate_noise_model
from cirq_google.api import v2
from google.protobuf.text_format import Merge
import pandas as pd

def test():
    xeb_1 = 0.01
    xeb_2 = 0.04

    p00_1 = 0.001
    p00_2 = 0.002
    p00_3 = 0.003

    p11_1 = 0.004
    p11_2 = 0.005
    p11_3 = 0.006

    t1_1 = 0.5 # microseconds
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
    output_df = validate_noise_model(calibration)

    # Construct expected results
    expected_p00 = np.mean([p00_1, p00_2, p00_3])
    expected_p11 = np.mean([p11_1, p11_2, p11_3])
    expected_xeb_fidelity = 1 - np.mean([xeb_1, xeb_2])
    expected_t1 = np.mean([t1_1, t1_2, t1_3]) * 1000

    num_qubits_xeb = 2
    N_xeb = 2 ** num_qubits_xeb
    expected_decay_constant = 1 - (1 - expected_xeb_fidelity) * N_xeb/ (N_xeb - 1)
    N = 2 # single qubit Hilbert space
    expected_pauli_error = (1 - expected_decay_constant) * (1 - 1 / N**2)

    expected_average_error = (1 - expected_decay_constant) * (1 - 1 / N)

    # Create noise model and simulator
    noise_prop = noise_properties_from_calibration(calibration)
    noise_model = NoiseModelFromNoiseProperties(noise_prop)

    qubits = [cirq.LineQubit(0), cirq.LineQubit(1)]

    simulator = cirq.sim.DensityMatrixSimulator(noise = noise_model, seed = 1)

    # Run experiments
    estimate_readout = cirq.experiments.estimate_single_qubit_readout_errors(simulator, qubits = [qubits[0]], repetitions = 1000)

    xeb_result = cirq.experiments.cross_entropy_benchmarking(simulator, qubits, num_circuits = 50, repetitions = 1000)
    measured_xeb = np.mean([datum.xeb_fidelity for datum in xeb_result.data])
    decay_constant = xeb_result.depolarizing_model().cycle_depolarization
    t1_results = cirq.experiments.t1_decay(simulator, qubit = qubits[0], num_points = 100, repetitions = 100, min_delay = cirq.Duration(nanos = 10), max_delay = cirq.Duration(micros = 1))

    N = 2 # Dimension of Hilbert Space
    measured_pauli_error = (1 - decay_constant) * (1 - 1 / N / N)
    measured_average_error = (1 - decay_constant) * (1 - 1 / N)

    output = []

    output.append(['p00', expected_p00, estimate_readout.zero_state_errors[cirq.LineQubit(0)]])
    output.append(['p11', expected_p11, estimate_readout.one_state_errors[cirq.LineQubit(0)]])
    output.append(['XEB Fidelity', expected_xeb_fidelity, measured_xeb])
    output.append(['T1', expected_t1, t1_results.constant])
    output.append(['Pauli Error', expected_pauli_error, measured_pauli_error])
    output.append(['Average Error', expected_average_error, measured_average_error])

    columns = ["Metric", "Initial value", "Measured value"]
    expected_df = pd.DataFrame(output, columns = columns)

    print(output_df)
    print(expected_df)
    assert np.allclose(expected_df['Initial value'], output_df['Initial value'], atol = 0.5)
    assert np.allclose(expected_df['Measured value'], output_df['Measured value'], atol = 0.5)

