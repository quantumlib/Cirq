# pylint: disable=wrong-or-nonexistent-copyright-notice
# coverage: ignore
import cirq_google
import numpy as np
from cirq.devices.noise_properties import NoiseModelFromNoiseProperties
from cirq_google.experimental.noise_models.calibration_to_noise_properties import (
    noise_properties_from_calibration,
)
import cirq
import pandas as pd


def compare_generated_noise_to_metrics(
    calibration: cirq_google.Calibration, validate: bool = True, tolerance: float = 0.01
):
    """Compares the metrics from a Calibration object to those measured from a Noise Model
       created with cirq.devices.noise_properties_from_calibration.

    Args:
    calibration: Calibration object to be turned into a Noise Model
    validate: check calibration metrics are in agreement (arg for noise_properties_from_calibration)
    tolerance: tolerance for calibration metrics (argument for noise_properties_from_calibration)

    Returns:
    df: Pandas dataframe comparing input and measured values for each calibration metric
    """
    # Create Noise Model from Calibration object
    noise_prop = noise_properties_from_calibration(calibration, validate, tolerance)
    noise_model = NoiseModelFromNoiseProperties(noise_prop)

    p00 = noise_prop.p00
    p11 = noise_prop.p11
    xeb_fidelity = noise_prop.xeb
    pauli_error = noise_prop.pauli_error
    t1_ns = noise_prop.t1_ns
    average_error = noise_prop.average_error()

    qubits = [cirq.LineQubit(0), cirq.LineQubit(1)]

    # Create simulator for experiments with noise model
    simulator = cirq.sim.DensityMatrixSimulator(noise=noise_model)

    # Experiments to measure metrics
    estimate_readout = cirq.experiments.estimate_single_qubit_readout_errors(
        simulator, qubits=[qubits[0]], repetitions=1000
    )

    xeb_result = cirq.experiments.cross_entropy_benchmarking(
        simulator, qubits, num_circuits=50, repetitions=1000
    )
    measured_xeb = np.mean([datum.xeb_fidelity for datum in xeb_result.data])
    decay_constant = xeb_result.depolarizing_model().cycle_depolarization

    output = []

    if p00 is not None:
        output.append(['p00', p00, estimate_readout.zero_state_errors[cirq.LineQubit(0)]])
    if p11 is not None:
        output.append(['p11', p11, estimate_readout.one_state_errors[cirq.LineQubit(0)]])
    if xeb_fidelity is not None:
        output.append(['XEB Fidelity', xeb_fidelity, measured_xeb])
    if t1_ns is not None:
        t1_results = cirq.experiments.t1_decay(
            simulator,
            qubit=qubits[0],
            num_points=100,
            repetitions=100,
            min_delay=cirq.Duration(nanos=10),
            max_delay=cirq.Duration(micros=1),
        )
        output.append(['T1', t1_ns, t1_results.constant])
    if pauli_error is not None:
        N = 2  # Dimension of Hilbert Space
        measured_pauli_error = (1 - decay_constant) * (1 - 1 / N / N)
        output.append(['Pauli Error', pauli_error, measured_pauli_error])
    if average_error is not None:
        N = 2  # Dimension of Hilbert Space
        measured_average_error = (1 - decay_constant) * (1 - 1 / N)
        output.append(['Average Error', average_error, measured_average_error])

    columns = ["Metric", "Initial value", "Measured value"]
    df = pd.DataFrame(output, columns=columns)
    return df
