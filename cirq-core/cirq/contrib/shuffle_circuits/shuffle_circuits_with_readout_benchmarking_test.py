# Copyright 2025 The Cirq Developers
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

import numpy as np
import cirq

from cirq.experiments.single_qubit_readout_calibration_test import NoisySingleQubitReadoutSampler
from cirq.experiments import random_quantum_circuit_generation as rqcg
from cirq.experiments import SingleQubitReadoutCalibrationResult
from cirq.study import ResultDict


def test_shuffled_circuits_with_readout_benchmarking_errors_no_noise():
    """Test shuffled circuits with readout benchmarking with no noise from sampler."""
    qubits = cirq.LineQubit.range(5)

    # Generate random input circuits
    input_circuits = []
    input_circuits += rqcg.generate_library_of_2q_circuits(
        n_library_circuits=5, two_qubit_gate=cirq.ISWAP**0.5, q0=qubits[0], q1=qubits[2]
    )
    input_circuits += rqcg.generate_library_of_2q_circuits(
        n_library_circuits=5, two_qubit_gate=cirq.CNOT**0.5, q0=qubits[1], q1=qubits[3]
    )
    input_circuits += rqcg.generate_library_of_2q_circuits(
        n_library_circuits=5, two_qubit_gate=cirq.CNOT**0.5, q0=qubits[0], q1=qubits[4]
    )
    input_circuits += rqcg.generate_library_of_2q_circuits(
        n_library_circuits=5, two_qubit_gate=cirq.ISWAP**0.5, q0=qubits[2], q1=qubits[4]
    )
    for circuit in input_circuits:
        circuit.append(cirq.measure(*qubits, key="m"))

    sampler = cirq.Simulator()
    circuit_repetitions = 1
    # allow passing a seed
    rng = 123
    readout_repetitions = 1000

    measurements, readout_calibration_results = (
        cirq.contrib.shuffle_circuits.run_shuffled_with_readout_benchmarking(
            input_circuits,
            sampler,
            circuit_repetitions,
            rng,
            num_random_bitstrings=100,
            readout_repetitions=readout_repetitions,
        )
    )

    for measurement in measurements:
        assert isinstance(measurement, ResultDict)

    for qlist, readout_calibration_result in readout_calibration_results.items():
        assert isinstance(qlist, tuple)
        assert all(isinstance(q, cirq.Qid) for q in qlist)
        assert isinstance(readout_calibration_result, SingleQubitReadoutCalibrationResult)

        assert readout_calibration_result.zero_state_errors == {q: 0 for q in qubits}
        assert readout_calibration_result.one_state_errors == {q: 0 for q in qubits}
        assert readout_calibration_result.repetitions == readout_repetitions
        assert isinstance(readout_calibration_result.timestamp, float)


def test_shuffled_circuits_with_readout_benchmarking_errors_with_noise():
    """Test shuffled circuits with readout benchmarking with noise from sampler."""
    qubits = cirq.LineQubit.range(6)

    # Generate random input circuits
    input_circuits = []
    input_circuits += rqcg.generate_library_of_2q_circuits(
        n_library_circuits=5, two_qubit_gate=cirq.ISWAP**0.5, q0=qubits[0], q1=qubits[1]
    )
    input_circuits += rqcg.generate_library_of_2q_circuits(
        n_library_circuits=5, two_qubit_gate=cirq.CNOT**0.5, q0=qubits[1], q1=qubits[3]
    )
    input_circuits += rqcg.generate_library_of_2q_circuits(
        n_library_circuits=5, two_qubit_gate=cirq.CNOT**0.5, q0=qubits[0], q1=qubits[4]
    )
    input_circuits += rqcg.generate_library_of_2q_circuits(
        n_library_circuits=5, two_qubit_gate=cirq.ISWAP**0.5, q0=qubits[2], q1=qubits[4]
    )
    input_circuits += rqcg.generate_library_of_2q_circuits(
        n_library_circuits=5, two_qubit_gate=cirq.ISWAP**0.5, q0=qubits[2], q1=qubits[5]
    )
    for circuit in input_circuits:
        circuit.append(cirq.measure(*qubits, key="m"))

    sampler = NoisySingleQubitReadoutSampler(p0=0.1, p1=0.2, seed=1234)
    circuit_repetitions = 1
    rng = np.random.default_rng()
    readout_repetitions = 1000

    measurements, readout_calibration_results = (
        cirq.contrib.shuffle_circuits.run_shuffled_with_readout_benchmarking(
            input_circuits,
            sampler,
            circuit_repetitions,
            rng,
            num_random_bitstrings=100,
            readout_repetitions=readout_repetitions,
        )
    )

    for measurement in measurements:
        assert isinstance(measurement, ResultDict)

    for qlist, readout_calibration_result in readout_calibration_results.items():
        assert isinstance(qlist, tuple)
        assert all(isinstance(q, cirq.Qid) for q in qlist)
        assert isinstance(readout_calibration_result, SingleQubitReadoutCalibrationResult)

        for error in readout_calibration_result.zero_state_errors.values():
            assert 0.08 < error < 0.12
        for error in readout_calibration_result.one_state_errors.values():
            assert 0.18 < error < 0.22
        assert readout_calibration_result.repetitions == readout_repetitions
        assert isinstance(readout_calibration_result.timestamp, float)


def test_shuffled_circuits_with_readout_benchmarking_errors_with_noise_and_input_qubits():
    """Test shuffled circuits with readout benchmarking with noise from sampler and input qubits."""
    qubits = cirq.LineQubit.range(6)
    readout_qubits = qubits[:4]

    # Generate random input circuits
    input_circuits = []
    input_circuits += rqcg.generate_library_of_2q_circuits(
        n_library_circuits=5, two_qubit_gate=cirq.ISWAP**0.5, q0=qubits[0], q1=qubits[1]
    )
    input_circuits += rqcg.generate_library_of_2q_circuits(
        n_library_circuits=5, two_qubit_gate=cirq.CNOT**0.5, q0=qubits[1], q1=qubits[2]
    )
    input_circuits += rqcg.generate_library_of_2q_circuits(
        n_library_circuits=5, two_qubit_gate=cirq.CNOT**0.5, q0=qubits[0], q1=qubits[2]
    )
    input_circuits += rqcg.generate_library_of_2q_circuits(
        n_library_circuits=5, two_qubit_gate=cirq.ISWAP**0.5, q0=qubits[4], q1=qubits[3]
    )
    input_circuits += rqcg.generate_library_of_2q_circuits(
        n_library_circuits=5, two_qubit_gate=cirq.ISWAP**0.5, q0=qubits[2], q1=qubits[5]
    )
    for circuit in input_circuits:
        circuit.append(cirq.measure(*qubits, key="m"))

    sampler = NoisySingleQubitReadoutSampler(p0=0.1, p1=0.3, seed=1234)
    circuit_repetitions = 1
    rng = np.random.default_rng()
    readout_repetitions = 1000

    measurements, readout_calibration_results = (
        cirq.contrib.shuffle_circuits.run_shuffled_with_readout_benchmarking(
            input_circuits,
            sampler,
            circuit_repetitions,
            rng,
            num_random_bitstrings=100,
            readout_repetitions=readout_repetitions,
            qubits=readout_qubits,
        )
    )

    for measurement in measurements:
        assert isinstance(measurement, ResultDict)

    for qlist, readout_calibration_result in readout_calibration_results.items():
        assert isinstance(qlist, tuple)
        assert all(isinstance(q, cirq.Qid) for q in qlist)
        assert isinstance(readout_calibration_result, SingleQubitReadoutCalibrationResult)

        for error in readout_calibration_result.zero_state_errors.values():
            assert 0.08 < error < 0.12
        for error in readout_calibration_result.one_state_errors.values():
            assert 0.28 < error < 0.32
        assert readout_calibration_result.repetitions == readout_repetitions
        assert isinstance(readout_calibration_result.timestamp, float)


def test_shuffled_circuits_with_readout_benchmarking_errors_with_noise_and_lists_input_qubits():
    """Test shuffled circuits with readout benchmarking with noise from sampler and input qubits."""
    qubits_1 = cirq.LineQubit.range(3)
    qubits_2 = cirq.LineQubit.range(4)

    readout_qubits = [qubits_1, qubits_2]

    # Generate random input circuits and append measurements
    input_circuit_1 = rqcg.generate_library_of_2q_circuits(
        n_library_circuits=5, two_qubit_gate=cirq.ISWAP**0.5, q0=qubits_1[0], q1=qubits_1[1]
    )
    for circuit in input_circuit_1:
        circuit.append(cirq.Circuit(cirq.measure(*qubits_1, key="m")))

    input_circuit_2 = rqcg.generate_library_of_2q_circuits(
        n_library_circuits=5, two_qubit_gate=cirq.CNOT**0.5, q0=qubits_1[1], q1=qubits_1[2]
    )
    for circuit in input_circuit_2:
        circuit.append(cirq.Circuit(cirq.measure(*qubits_1, key="m")))

    input_circuit_3 = rqcg.generate_library_of_2q_circuits(
        n_library_circuits=5, two_qubit_gate=cirq.CNOT**0.5, q0=qubits_2[0], q1=qubits_2[3]
    )
    for circuit in input_circuit_3:
        circuit.append(cirq.Circuit(cirq.measure(*qubits_2, key="m")))

    input_circuit_4 = rqcg.generate_library_of_2q_circuits(
        n_library_circuits=5, two_qubit_gate=cirq.ISWAP**0.5, q0=qubits_2[1], q1=qubits_1[2]
    )
    for circuit in input_circuit_4:
        circuit.append(cirq.Circuit(cirq.measure(*qubits_2, key="m")))

    input_circuits = input_circuit_1 + input_circuit_2 + input_circuit_3 + input_circuit_4

    sampler = NoisySingleQubitReadoutSampler(p0=0.1, p1=0.3, seed=1234)
    circuit_repetitions = 1
    rng = np.random.default_rng()
    readout_repetitions = 1000

    measurements, readout_calibration_results = (
        cirq.contrib.shuffle_circuits.run_shuffled_with_readout_benchmarking(
            input_circuits,
            sampler,
            circuit_repetitions,
            rng,
            num_random_bitstrings=100,
            readout_repetitions=readout_repetitions,
            qubits=readout_qubits,
        )
    )

    for measurement in measurements:
        assert isinstance(measurement, ResultDict)

    for qlist, readout_calibration_result in readout_calibration_results.items():
        assert isinstance(qlist, tuple)
        assert all(isinstance(q, cirq.Qid) for q in qlist)
        assert isinstance(readout_calibration_result, SingleQubitReadoutCalibrationResult)

        for error in readout_calibration_result.zero_state_errors.values():
            assert 0.08 < error < 0.12
        for error in readout_calibration_result.one_state_errors.values():
            assert 0.28 < error < 0.32
        assert readout_calibration_result.repetitions == readout_repetitions
        assert isinstance(readout_calibration_result.timestamp, float)


def test_empty_input_circuits():
    """Test that the input circuits are empty."""
    with pytest.raises(ValueError, match="Input circuits must not be empty."):
        cirq.contrib.shuffle_circuits.run_shuffled_with_readout_benchmarking(
            [],
            cirq.ZerosSampler(),
            circuit_repetitions=10,
            rng_or_seed=np.random.default_rng(456),
            num_random_bitstrings=5,
            readout_repetitions=100,
        )


def test_non_circuit_input():
    """Test that the input circuits are not of type cirq.Circuit."""
    q = cirq.LineQubit(0)
    with pytest.raises(ValueError, match="Input circuits must be of type cirq.Circuit."):
        cirq.contrib.shuffle_circuits.run_shuffled_with_readout_benchmarking(
            [q],
            cirq.ZerosSampler(),
            circuit_repetitions=10,
            rng_or_seed=np.random.default_rng(456),
            num_random_bitstrings=5,
            readout_repetitions=100,
        )


def test_no_measurements():
    """Test that the input circuits don't have measurements."""
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q))
    with pytest.raises(ValueError, match="Input circuits must have measurements."):
        cirq.contrib.shuffle_circuits.run_shuffled_with_readout_benchmarking(
            [circuit],
            cirq.ZerosSampler(),
            circuit_repetitions=10,
            rng_or_seed=np.random.default_rng(456),
            num_random_bitstrings=5,
            readout_repetitions=100,
        )


def test_zero_circuit_repetitions():
    """Test that the circuit repetitions are zero."""
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q), cirq.measure(q))
    with pytest.raises(ValueError, match="Must provide non-zero circuit_repetitions."):
        cirq.contrib.shuffle_circuits.run_shuffled_with_readout_benchmarking(
            [circuit],
            cirq.ZerosSampler(),
            circuit_repetitions=0,
            rng_or_seed=np.random.default_rng(456),
            num_random_bitstrings=5,
            readout_repetitions=100,
        )


def test_mismatch_circuit_repetitions():
    """Test that the number of circuit repetitions don't match the number of input circuits."""
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q), cirq.measure(q))
    with pytest.raises(
        ValueError,
        match="Number of circuit_repetitions must match the number of" + " input circuits.",
    ):
        cirq.contrib.shuffle_circuits.run_shuffled_with_readout_benchmarking(
            [circuit],
            cirq.ZerosSampler(),
            circuit_repetitions=[10, 20],
            rng_or_seed=np.random.default_rng(456),
            num_random_bitstrings=5,
            readout_repetitions=100,
        )


def test_zero_num_random_bitstrings():
    """Test that the number of random bitstrings is zero."""
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q), cirq.measure(q))
    with pytest.raises(ValueError, match="Must provide non-zero num_random_bitstrings."):
        cirq.contrib.shuffle_circuits.run_shuffled_with_readout_benchmarking(
            [circuit],
            cirq.ZerosSampler(),
            circuit_repetitions=10,
            rng_or_seed=np.random.default_rng(456),
            num_random_bitstrings=0,
            readout_repetitions=100,
        )


def test_zero_readout_repetitions():
    """Test that the readout repetitions is zero."""
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q), cirq.measure(q))
    with pytest.raises(
        ValueError, match="Must provide non-zero readout_repetitions for readout" + " calibration."
    ):
        cirq.contrib.shuffle_circuits.run_shuffled_with_readout_benchmarking(
            [circuit],
            cirq.ZerosSampler(),
            circuit_repetitions=10,
            rng_or_seed=np.random.default_rng(456),
            num_random_bitstrings=5,
            readout_repetitions=0,
        )


def test_rng_type_mismatch():
    """Test that the rng is not a numpy random generator or a seed."""
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q), cirq.measure(q))
    with pytest.raises(ValueError, match="Must provide a numpy random generator or a seed"):
        cirq.contrib.shuffle_circuits.run_shuffled_with_readout_benchmarking(
            [circuit],
            cirq.ZerosSampler(),
            circuit_repetitions=10,
            rng_or_seed="not a random generator or seed",
            num_random_bitstrings=5,
            readout_repetitions=100,
        )
