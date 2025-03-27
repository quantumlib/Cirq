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
import pytest
import sympy

import cirq
import cirq_google.engine.runtime_estimator as runtime_estimator


def _assert_about_equal(actual: float, expected: float):
    """Assert that two times are within 25% of the expected time.

    Used to test the estimator with noisy data from actual devices.
    """
    assert expected * 0.75 < actual < expected * 1.25


@pytest.mark.parametrize("reps,expected", [(1000, 2.25), (16000, 2.9), (64000, 4.6), (128000, 7.4)])
def test_estimate_run_time_vary_reps(reps, expected):
    """Test various run times.
    Values taken from Weber November 2021."""
    qubits = cirq.GridQubit.rect(2, 5)
    circuit = cirq.testing.random_circuit(qubits, n_moments=10, op_density=1.0)
    runtime = runtime_estimator.estimate_run_time(circuit, repetitions=reps)
    _assert_about_equal(runtime, expected)


@pytest.mark.parametrize(
    "depth, width, reps, expected",
    [
        (10, 80, 32000, 3.7),
        (10, 160, 32000, 4.5),
        (10, 320, 32000, 6.6),
        (10, 10, 32000, 3.5),
        (20, 10, 32000, 4.6),
        (30, 10, 32000, 5.9),
        (40, 10, 32000, 7.7),
        (50, 10, 32000, 9.4),
        (10, 10, 256000, 11.4),
        (40, 40, 256000, 26.8),
        (40, 160, 256000, 32.8),
        (40, 80, 32000, 12.1),
        (2, 40, 256000, 11.3),
        (2, 160, 256000, 11.4),
        (2, 640, 256000, 13.3),
        (2, 1280, 256000, 16.5),
        (2, 2560, 256000, 23.5),
        (10, 160, 256000, 18.2),
        (20, 160, 256000, 24.7),
        (30, 160, 256000, 30.8),
        (10, 1280, 256000, 38.6),
        (10, 1280, 1000, 18.7),
        (10, 1280, 256000, 38.6),
    ],
)
def test_estimate_run_time(depth, width, reps, expected):
    """Test various run times.
    Values taken from Weber November 2021."""
    qubits = cirq.GridQubit.rect(8, 8)
    circuit = cirq.testing.random_circuit(qubits[:depth], n_moments=width, op_density=1.0)
    runtime = runtime_estimator.estimate_run_time(circuit, repetitions=reps)
    _assert_about_equal(runtime, expected)


@pytest.mark.parametrize(
    "depth, width, reps, sweeps, expected",
    [
        (10, 10, 1000, 1, 2.3),
        (10, 10, 1000, 2, 2.8),
        (10, 10, 1000, 4, 3.2),
        (10, 10, 1000, 8, 4.1),
        (10, 10, 1000, 16, 6.1),
        (10, 10, 1000, 32, 10.2),
        (10, 10, 1000, 64, 19.2),
        (40, 10, 1000, 2, 6.0),
        (40, 10, 1000, 4, 7.2),
        (40, 10, 1000, 8, 10.9),
        (40, 10, 1000, 16, 17.2),
        (40, 10, 1000, 32, 32.2),
        (40, 10, 1000, 64, 61.4),
        (40, 10, 1000, 128, 107.5),
        (40, 160, 32000, 32, 249.7),
        (30, 40, 32000, 32, 171.0),
        (40, 40, 32000, 32, 206.9),
        (40, 80, 32000, 16, 90.4),
        (40, 80, 32000, 8, 58.7),
        (40, 80, 8000, 32, 80.1),
        (20, 40, 32000, 32, 69.8),
        (30, 40, 32000, 32, 170.9),
        (40, 40, 32000, 32, 215.4),
        (2, 40, 16000, 16, 10.5),
        (2, 640, 16000, 16, 16.9),
        (2, 1280, 16000, 16, 22.6),
        (2, 2560, 16000, 16, 38.9),
    ],
)
def test_estimate_run_sweep_time(depth, width, sweeps, reps, expected):
    """Test various run times.
    Values taken from Weber November 2021."""
    qubits = cirq.GridQubit.rect(8, 8)
    circuit = cirq.testing.random_circuit(qubits[:depth], n_moments=width, op_density=1.0)
    params = cirq.Linspace('t', 0, 1, sweeps)
    runtime = runtime_estimator.estimate_run_sweep_time(circuit, params, repetitions=reps)
    _assert_about_equal(runtime, expected)


@pytest.mark.parametrize("num_qubits", [54, 72, 100, 150, 200])
def test_many_qubits(num_qubits: int) -> None:
    """Regression test

    Make sure that high numbers of qubits do not
    slow the rep rate down to below zero.
    """
    qubits = cirq.LineQubit.range(num_qubits)
    sweeps_10 = cirq.Linspace('t', 0, 1, 10)
    circuit = cirq.Circuit(*[cirq.X(q) ** sympy.Symbol('t') for q in qubits], cirq.measure(*qubits))
    sweep_runtime = runtime_estimator.estimate_run_sweep_time(circuit, sweeps_10, repetitions=10000)
    assert sweep_runtime > 0


def test_estimate_run_batch_time():
    qubits = cirq.GridQubit.rect(4, 5)
    circuit = cirq.testing.random_circuit(qubits[:19], n_moments=40, op_density=1.0)
    circuit2 = cirq.testing.random_circuit(qubits[:19], n_moments=40, op_density=1.0)
    circuit3 = cirq.testing.random_circuit(qubits, n_moments=40, op_density=1.0)
    sweeps_10 = cirq.Linspace('t', 0, 1, 10)
    sweeps_20 = cirq.Linspace('t', 0, 1, 20)
    sweeps_30 = cirq.Linspace('t', 0, 1, 30)
    sweeps_40 = cirq.Linspace('t', 0, 1, 40)

    # 2 batches with same qubits is the same time as a combined sweep
    sweep_runtime = runtime_estimator.estimate_run_sweep_time(circuit, sweeps_30, repetitions=1000)
    batch_runtime = runtime_estimator.estimate_run_batch_time(
        [circuit, circuit2], [sweeps_10, sweeps_20], repetitions=1000
    )
    assert sweep_runtime == batch_runtime

    # 2 batches with same qubits and 1 batch with different qubits
    # Should be equal to combining the first two batches
    three_batches = runtime_estimator.estimate_run_batch_time(
        [circuit, circuit2, circuit3], [sweeps_10, sweeps_20, sweeps_10], repetitions=1000
    )
    two_batches = runtime_estimator.estimate_run_batch_time(
        [circuit, circuit3], [sweeps_30, sweeps_10], repetitions=1000
    )
    assert three_batches == two_batches
    # The last batch cannot be combined since it has different qubits
    sweep_runtime = runtime_estimator.estimate_run_sweep_time(circuit, sweeps_40, repetitions=1000)
    assert three_batches > sweep_runtime


def test_estimate_run_batch_time_average_depths():
    qubits = cirq.GridQubit.rect(4, 5)
    circuit_depth_20 = cirq.testing.random_circuit(qubits, n_moments=20, op_density=1.0)
    circuit_depth_30 = cirq.testing.random_circuit(qubits, n_moments=30, op_density=1.0)
    circuit_depth_40 = cirq.testing.random_circuit(qubits, n_moments=40, op_density=1.0)
    sweeps_10 = cirq.Linspace('t', 0, 1, 10)
    sweeps_20 = cirq.Linspace('t', 0, 1, 20)

    depth_20_and_40 = runtime_estimator.estimate_run_batch_time(
        [circuit_depth_20, circuit_depth_40], [sweeps_10, sweeps_10], repetitions=1000
    )
    depth_30 = runtime_estimator.estimate_run_sweep_time(
        circuit_depth_30, sweeps_20, repetitions=1000
    )
    depth_40 = runtime_estimator.estimate_run_sweep_time(
        circuit_depth_40, sweeps_20, repetitions=1000
    )
    assert depth_20_and_40 == depth_30
    assert depth_20_and_40 < depth_40
