# coverage: ignore
from typing import Any, List
import cirq_google
from cirq.devices.noise_properties import NoiseModelFromNoiseProperties
from cirq.devices.noise_utils import (
    OpIdentifier,
    decay_constant_to_pauli_error,
)
from cirq_google.experimental.noise_models.calibration_to_noise_properties import (
    noise_properties_from_calibration,
)
import cirq
import pandas as pd


def compare_generated_noise_to_metrics(calibration: cirq_google.Calibration):
    """Compares the metrics from a Calibration object to those measured from a Noise Model
       created with cirq.devices.noise_properties_from_calibration.

    Args:
        calibration: Calibration object to convert to noise

    Returns:
        Pandas dataframe comparing input and measured values for each calibration metric
    """
    # Create Noise Model from Calibration object
    noise_prop = noise_properties_from_calibration(calibration)
    noise_model = NoiseModelFromNoiseProperties(noise_prop)

    # TODO: refactor this experiment
    q0 = cirq.LineQubit(0)
    p00 = noise_prop.ro_fidelities[q0][0]
    p11 = noise_prop.ro_fidelities[q0][1]
    pauli_error = noise_prop.gate_pauli_errors[OpIdentifier(cirq.PhasedXZGate, q0)]
    t1_ns = noise_prop.T1_ns[q0]

    # # Create simulator for experiments with noise model
    simulator = cirq.sim.DensityMatrixSimulator(noise=noise_model)

    # # Experiments to measure metrics
    estimate_readout = cirq.experiments.estimate_single_qubit_readout_errors(
        simulator, qubits=[q0], repetitions=1000
    )

    xeb_result = cirq.experiments.cross_entropy_benchmarking(
        simulator, [q0], num_circuits=50, repetitions=1000
    )
    decay_constant_result = xeb_result.depolarizing_model().cycle_depolarization
    pauli_error_result = decay_constant_to_pauli_error(decay_constant_result)

    output: List[Any] = []

    output.append(['p00', p00, estimate_readout.zero_state_errors[cirq.LineQubit(0)]])
    output.append(['p11', p11, estimate_readout.one_state_errors[cirq.LineQubit(0)]])
    t1_results = cirq.experiments.t1_decay(
        simulator,
        qubit=q0,
        num_points=100,
        repetitions=100,
        min_delay=cirq.Duration(nanos=10),
        max_delay=cirq.Duration(micros=1),
    )
    output.append(['T1', t1_ns, t1_results.constant])
    output.append(['Pauli Error', pauli_error, pauli_error_result])

    columns = ["Metric", "Initial value", "Measured value"]
    df = pd.DataFrame(output, columns=columns)
    return df
