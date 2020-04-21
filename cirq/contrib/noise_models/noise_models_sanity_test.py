# Lint as: python3
"""Sanity tests for noise_models.

These tests simulate calibration on a noisy circuit; runs will take several
minutes to complete. For testing during development, set _REPETITIONS to 1.
"""

from google.protobuf.text_format import Merge

import numpy as np

import cirq
import cirq.contrib.noise_models.noise_models as ccn
from cirq.google.api import v2


# Sample calibration data for 21 Apr 2020, polled from QuantumEngine.
_CALIBRATION_DATA = Merge(
    """
    timestamp_ms: 1583870096,
    metrics: [{
        name: 'single_qubit_rb_pauli_error_per_gate',
        targets: ['5_2'],
        values: [{
            double_val: 0.0022007
        }]
    }, {
        name: 'single_qubit_rb_pauli_error_per_gate',
        targets: ['5_3'],
        values: [{
            double_val: 0.0016749
        }]
    }, {
        name: 'single_qubit_rb_pauli_error_per_gate',
        targets: ['6_2'],
        values: [{
            double_val: 0.0020182
        }]
    }, {
        name: 'single_qubit_rb_pauli_error_per_gate',
        targets: ['6_3'],
        values: [{
            double_val: 0.0015040
        }]
    }, {
        name: 'single_qubit_readout_separation_error',
        targets: ['5_2'],
        values: [{
            double_val: 0.0025676
        }]
    }, {
        name: 'single_qubit_readout_separation_error',
        targets: ['5_3'],
        values: [{
            double_val: 0.0012148
        }]
    }, {
        name: 'single_qubit_readout_separation_error',
        targets: ['6_2'],
        values: [{
            double_val: 0.0080927
        }]
    }, {
        name: 'single_qubit_readout_separation_error',
        targets: ['6_3'],
        values: [{
            double_val: 0.0035724
        }]
    }, {
        name: 'single_qubit_p00_error',
        targets: ['5_2'],
        values: [{
            double_val: 0.013000
        }]
    }, {
        name: 'single_qubit_p00_error',
        targets: ['5_3'],
        values: [{
            double_val: 0.0058000
        }]
    }, {
        name: 'single_qubit_p00_error',
        targets: ['6_2'],
        values: [{
            double_val: 0.021000
        }]
    }, {
        name: 'single_qubit_p00_error',
        targets: ['6_3'],
        values: [{
            double_val: 0.011200
        }]
    }, {
        name: 'single_qubit_p11_error',
        targets: ['5_2'],
        values: [{
            double_val: 0.047000
        }]
    }, {
        name: 'single_qubit_p11_error',
        targets: ['5_3'],
        values: [{
            double_val: 0.073600
        }]
    }, {
        name: 'single_qubit_p11_error',
        targets: ['6_2'],
        values: [{
            double_val: 0.060200
        }]
    }, {
        name: 'single_qubit_p11_error',
        targets: ['6_3'],
        values: [{
            double_val: 0.091400
        }]
    }, {
        name: 'single_qubit_idle_t1_micros',
        targets: ['5_2'],
        values: [{
            double_val: 18.091
        }]
    }, {
        name: 'single_qubit_idle_t1_micros',
        targets: ['5_3'],
        values: [{
            double_val: 6.0306
        }]
    }, {
        name: 'single_qubit_idle_t1_micros',
        targets: ['6_2'],
        values: [{
            double_val: 14.945
        }]
    }, {
        name: 'single_qubit_idle_t1_micros',
        targets: ['6_3'],
        values: [{
            double_val: 14.489
        }]
    }]
""", v2.metrics_pb2.MetricsSnapshot())


def wilson_interval(num_trials, success_ratio, std_deviations):
    """Helper method for calculating confidence intervals using Wilson score:
    https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval

    Args:
        num_trials: number of trials run.
        success_ratio: ratio of error-free runs to total runs.
        std_deviations: width of the desired interval in standard deviations.

    Returns:
        A tuple containing the Wilson interval (min, max).
    """
    base_numer = success_ratio + std_deviations ** 2 / (2 * num_trials)
    delta_numer = std_deviations * np.sqrt(
        (success_ratio*(1-success_ratio)
         + (std_deviations ** 2) / (4 * num_trials)) / num_trials)
    denom = (1 + std_deviations ** 2 / num_trials)
    return ((base_numer - delta_numer) / denom,
            (base_numer + delta_numer) / denom) 


def test_randomized_benchmark():
    calibration = cirq.google.Calibration(_CALIBRATION_DATA)
    noise_model = ccn.simple_noise_from_calibration_metrics(
        calibration=calibration,
        depol_noise=True,
        readout_error_noise=True,
        readout_decay_noise=True,
    )
    sampler = cirq.DensityMatrixSimulator(noise=noise_model, seed=1)

    print('Beginning RB tests')
    for qubit in noise_model.qubits:
        # TODO: best guess for current failure is that this is the total error,
        # not just depol effects. Error seems off by ~2x
        fidelity = 1 - calibration['single_qubit_rb_pauli_error_per_gate'][(qubit,)][0]
        gate_counts = [10, 100]
        trials = 10
        reps = 1000
        result = cirq.experiments.single_qubit_randomized_benchmarking(
            sampler=sampler,
            qubit=qubit,
            num_clifford_range=gate_counts,
            num_circuits=trials,
            repetitions=reps,
        )
        print(f'Randomized benchmark results ({qubit}): \n{result.data}')
        for i, num_gates in enumerate(gate_counts):
            assert result.data[i][0] == num_gates
            # ~2.125 gates per Clifford, plus a terminal "inverse" Clifford.
            actual_gates = 2.125 * (num_gates + 1)
            target = fidelity ** actual_gates
            # target = np.mean([
            #     fidelity ** gate_count for gate_count in result.gates[i][1]
            # ])
            print(f'fidelity: {fidelity}')
            print(f'observed: {result.data[i][1] ** (1 / actual_gates)}')
            interval = wilson_interval(trials * reps, result.data[i][1], 3)
            print(f'{target} is within {interval}')
            # assert interval[0] < target < interval[1]
    assert False


# TODO: compare with p00, p11 results as well
def test_readout_calibration():
    calibration = cirq.google.Calibration(_CALIBRATION_DATA)
    noise_model = ccn.simple_noise_from_calibration_metrics(
        calibration=calibration,
        depol_noise=False,
        readout_error_noise=True,
        readout_decay_noise=True,
    )
    sampler = cirq.DensityMatrixSimulator(noise=noise_model, seed=0)

    p00_err = {
        qubit[0]: vals[0]
        for qubit, vals in calibration['single_qubit_p00_error'].items()
    }
    p11_err = {
        qubit[0]: vals[0]
        for qubit, vals in calibration['single_qubit_p11_error'].items()
    }
    print(p00_err)
    print(p11_err)
    reps = 1000
    print('Beginning readout tests')
    result = cirq.experiments.estimate_single_qubit_readout_errors(
        sampler=sampler, qubits=noise_model.qubits, repetitions=reps)
    print(f'Readout error results: {result}')
    for qubit in noise_model.qubits:
        observed_p00 = result.zero_state_errors[qubit]
        observed_p11 = result.one_state_errors[qubit]
        p00_interval = wilson_interval(reps, observed_p00, 3)
        p11_interval = wilson_interval(reps, observed_p11, 3)

        expected_p00 = p00_err[qubit]
        expected_p11 = p11_err[qubit]
        print(f'p00: {expected_p00} is within {p00_interval}')
        print(f'p11: {expected_p11} is within {p11_interval}')
        # assert p00_interval[0] < expected_p00 < p00_interval[1]
        # assert p11_interval[0] < expected_p11 < p11_interval[1]
    assert False
