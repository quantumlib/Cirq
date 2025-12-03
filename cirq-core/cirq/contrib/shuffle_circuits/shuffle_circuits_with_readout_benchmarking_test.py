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

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import pytest
import sympy

import cirq
import cirq.contrib.shuffle_circuits.shuffle_circuits_with_readout_benchmarking as sc_readout
from cirq.experiments import (
    random_quantum_circuit_generation as rqcg,
    SingleQubitReadoutCalibrationResult,
)
from cirq.experiments.single_qubit_readout_calibration_test import NoisySingleQubitReadoutSampler
from cirq.study import ResultDict


def _create_test_circuits(qubits: Sequence[cirq.Qid], n_circuits: int) -> list[cirq.Circuit]:
    """Helper function to generate circuits for testing."""
    if len(qubits) < 2:
        raise ValueError(
            "Need at least two qubits to generate two-qubit circuits."
        )  # pragma: no cover
    two_qubit_gates = [cirq.ISWAP**0.5, cirq.CNOT**0.5]
    input_circuits = []
    qubit_pairs = list(itertools.combinations(qubits, 2))
    num_pairs = len(qubit_pairs)
    for i in range(n_circuits):
        gate = two_qubit_gates[i % len(two_qubit_gates)]
        q0, q1 = qubit_pairs[i % num_pairs]
        circuits = rqcg.generate_library_of_2q_circuits(
            n_library_circuits=5, two_qubit_gate=gate, q0=q0, q1=q1
        )
        for circuit in circuits:
            circuit.append(cirq.measure(*qubits, key="m"))
        input_circuits.extend(circuits)
    return input_circuits


def _create_test_circuits_with_sweep(
    qubits: Sequence[cirq.Qid], n_circuits: int
) -> tuple[list[cirq.Circuit], list[cirq.ParamResolver]]:
    """Helper function to generate sweep circuits for testing."""
    if len(qubits) < 2:
        raise ValueError(
            "Need at least two qubits to generate two-qubit circuits."
        )  # pragma: no cover
    theta_symbol = sympy.Symbol('theta')
    phi_symbol = sympy.Symbol('phi')

    two_qubit_gates = [cirq.ISWAP, cirq.CNOT]

    input_circuits = []
    sweep_params: list[cirq.ParamResolver] = []
    qubit_pairs = list(itertools.combinations(qubits, 2))
    num_pairs = len(qubit_pairs)
    for i in range(n_circuits):
        gate = two_qubit_gates[i % len(two_qubit_gates)]
        q0, q1 = qubit_pairs[i % num_pairs]
        circuits = rqcg.generate_library_of_2q_circuits(
            n_library_circuits=5, two_qubit_gate=gate, q0=q0, q1=q1
        )
        for circuit in circuits:
            circuit += cirq.Circuit(cirq.X(q0) ** theta_symbol, cirq.Y(q1) ** phi_symbol)
            circuit.append(cirq.measure(*qubits, key="m"))
            sweep_params.append(cirq.ParamResolver({'theta': 0, 'phi': 1}))
        input_circuits.extend(circuits)

    return input_circuits, sweep_params


def _circuits_with_readout_benchmarking_errors_shuffled(
    sampler: cirq.Sampler,
    input_circuits: list[cirq.Circuit],
    qubits: Sequence[cirq.Qid] | Sequence[Sequence[cirq.Qid]],
    parameters: sc_readout.ReadoutBenchmarkingParams,
    rng_or_seed: np.random.Generator | int,
):
    measurements, readout_calibration_results = (
        sc_readout.run_shuffled_circuits_with_readout_benchmarking(
            sampler, input_circuits, parameters, qubits, rng_or_seed
        )
    )

    for measurement in measurements:
        assert isinstance(measurement, ResultDict)
    return readout_calibration_results


def _circuits_with_readout_benchmarking_errors_sweep(
    sampler: cirq.Sampler,
    input_circuits: list[cirq.Circuit],
    qubits: Sequence[cirq.Qid] | Sequence[Sequence[cirq.Qid]],
    sweep_params: list[cirq.ParamResolver],
    parameters: sc_readout.ReadoutBenchmarkingParams,
    rng_or_seed: np.random.Generator | int,
):
    sweep_measurements, readout_calibration_results = (
        sc_readout.run_sweep_with_readout_benchmarking(
            sampler, input_circuits, sweep_params, parameters, qubits, rng_or_seed
        )
    )

    for measurement_group in sweep_measurements:
        for single_sweep_measurement in measurement_group:
            assert isinstance(single_sweep_measurement, ResultDict)
    return readout_calibration_results


@pytest.mark.parametrize("mode", ["shuffled", "sweep"])
def test_circuits_with_readout_benchmarking_errors_no_noise(mode: str) -> None:
    """Test shuffled/sweep circuits with readout benchmarking with no noise from sampler."""
    qubits = cirq.LineQubit.range(5)

    sampler = cirq.Simulator()
    circuit_repetitions = 1
    num_random_bitstrings = 100
    readout_repetitions = 1000

    readout_benchmarking_params = sc_readout.ReadoutBenchmarkingParams(
        circuit_repetitions=circuit_repetitions,
        num_random_bitstrings=num_random_bitstrings,
        readout_repetitions=readout_repetitions,
    )
    # allow passing a seed
    rng = 123

    if mode == "shuffled":
        input_circuits = _create_test_circuits(qubits, 3)
        readout_calibration_results = _circuits_with_readout_benchmarking_errors_shuffled(
            sampler, input_circuits, qubits, readout_benchmarking_params, rng
        )

    elif mode == "sweep":
        input_circuits, sweep_params = _create_test_circuits_with_sweep(qubits, 3)

        readout_calibration_results = _circuits_with_readout_benchmarking_errors_sweep(
            sampler, input_circuits, qubits, sweep_params, readout_benchmarking_params, rng
        )

    for qlist, readout_calibration_result in readout_calibration_results.items():
        assert isinstance(qlist, tuple)
        assert all(isinstance(q, cirq.Qid) for q in qlist)
        assert isinstance(readout_calibration_result, SingleQubitReadoutCalibrationResult)

        assert readout_calibration_result.zero_state_errors == {q: 0 for q in qubits}
        assert readout_calibration_result.one_state_errors == {q: 0 for q in qubits}
        assert readout_calibration_result.repetitions == readout_repetitions
        assert isinstance(readout_calibration_result.timestamp, float)


@pytest.mark.parametrize("mode", ["shuffled", "sweep"])
def test_circuits_with_readout_benchmarking_errors_with_noise(mode: str) -> None:
    """Test shuffled/sweep circuits with readout benchmarking with noise from sampler."""
    qubits = cirq.LineQubit.range(6)
    sampler = NoisySingleQubitReadoutSampler(p0=0.1, p1=0.2, seed=1234)
    circuit_repetitions = 1
    rng = np.random.default_rng()
    readout_repetitions = 1000
    num_random_bitstrings = 100

    readout_benchmarking_params = sc_readout.ReadoutBenchmarkingParams(
        circuit_repetitions=circuit_repetitions,
        num_random_bitstrings=num_random_bitstrings,
        readout_repetitions=readout_repetitions,
    )

    if mode == "shuffled":
        input_circuits = _create_test_circuits(qubits, 6)

        readout_calibration_results = _circuits_with_readout_benchmarking_errors_shuffled(
            sampler, input_circuits, qubits, readout_benchmarking_params, rng
        )

    elif mode == "sweep":
        input_circuits, sweep_params = _create_test_circuits_with_sweep(qubits, 6)

        readout_calibration_results = _circuits_with_readout_benchmarking_errors_sweep(
            sampler, input_circuits, qubits, sweep_params, readout_benchmarking_params, rng
        )

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


@pytest.mark.parametrize("mode", ["shuffled", "sweep"])
def test_circuits_with_readout_benchmarking_errors_with_noise_and_input_qubits(mode: str) -> None:
    """Test shuffled/sweep circuits with readout benchmarking with
    noise from sampler and input qubits."""
    qubits = cirq.LineQubit.range(6)
    readout_qubits = qubits[:4]

    sampler = NoisySingleQubitReadoutSampler(p0=0.1, p1=0.3, seed=1234)
    circuit_repetitions = 1
    rng = np.random.default_rng()
    readout_repetitions = 1000
    num_random_bitstrings = 100

    readout_benchmarking_params = sc_readout.ReadoutBenchmarkingParams(
        circuit_repetitions=circuit_repetitions,
        num_random_bitstrings=num_random_bitstrings,
        readout_repetitions=readout_repetitions,
    )

    if mode == "shuffled":
        input_circuits = _create_test_circuits(qubits, 6)

        readout_calibration_results = _circuits_with_readout_benchmarking_errors_shuffled(
            sampler, input_circuits, readout_qubits, readout_benchmarking_params, rng
        )

    elif mode == "sweep":
        input_circuits, sweep_params = _create_test_circuits_with_sweep(qubits, 6)

        readout_calibration_results = _circuits_with_readout_benchmarking_errors_sweep(
            sampler, input_circuits, readout_qubits, sweep_params, readout_benchmarking_params, rng
        )

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


@pytest.mark.parametrize("mode", ["shuffled", "sweep"])
def test_circuits_with_readout_benchmarking_errors_with_noise_and_lists_input_qubits(
    mode: str,
) -> None:
    """Test shuffled/sweep circuits with readout benchmarking with noise
    from sampler and input qubits."""
    qubits_1 = cirq.LineQubit.range(3)
    qubits_2 = cirq.LineQubit.range(4)
    readout_qubits = [qubits_1, qubits_2]

    sampler = NoisySingleQubitReadoutSampler(p0=0.1, p1=0.3, seed=1234)
    circuit_repetitions = 1
    rng = np.random.default_rng()
    readout_repetitions = 1000
    num_random_bitstrings = 100

    readout_benchmarking_params = sc_readout.ReadoutBenchmarkingParams(
        circuit_repetitions=circuit_repetitions,
        num_random_bitstrings=num_random_bitstrings,
        readout_repetitions=readout_repetitions,
    )

    if mode == "shuffled":
        input_circuits = _create_test_circuits(qubits_1, 6) + _create_test_circuits(qubits_2, 4)

        readout_calibration_results = _circuits_with_readout_benchmarking_errors_shuffled(
            sampler, input_circuits, readout_qubits, readout_benchmarking_params, rng
        )

    elif mode == "sweep":
        input_circuits, sweep_params = _create_test_circuits_with_sweep(qubits_1, 6)
        additional_circuits, additional_sweep_params = _create_test_circuits_with_sweep(qubits_2, 4)
        input_circuits += additional_circuits
        sweep_params += additional_sweep_params

        readout_calibration_results = _circuits_with_readout_benchmarking_errors_sweep(
            sampler, input_circuits, readout_qubits, sweep_params, readout_benchmarking_params, rng
        )

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


@pytest.mark.parametrize("mode", ["shuffled", "sweep"])
def test_can_handle_zero_random_bitstring(mode: str) -> None:
    """Test shuffled/sweep circuits without readout benchmarking."""
    qubits_1 = cirq.LineQubit.range(3)
    qubits_2 = cirq.LineQubit.range(4)
    readout_qubits = [qubits_1, qubits_2]

    sampler = NoisySingleQubitReadoutSampler(p0=0.1, p1=0.3, seed=1234)
    circuit_repetitions = 1
    rng = np.random.default_rng()
    readout_repetitions = 1000
    num_random_bitstrings = 0

    readout_benchmarking_params = sc_readout.ReadoutBenchmarkingParams(
        circuit_repetitions=circuit_repetitions,
        num_random_bitstrings=num_random_bitstrings,
        readout_repetitions=readout_repetitions,
    )

    if mode == "shuffled":
        input_circuits = _create_test_circuits(qubits_1, 6) + _create_test_circuits(qubits_2, 4)

        readout_calibration_results = _circuits_with_readout_benchmarking_errors_shuffled(
            sampler, input_circuits, readout_qubits, readout_benchmarking_params, rng
        )

    elif mode == "sweep":
        input_circuits, sweep_params = _create_test_circuits_with_sweep(qubits_1, 6)
        additional_circuits, additional_sweep_params = _create_test_circuits_with_sweep(qubits_2, 4)
        input_circuits += additional_circuits
        sweep_params += additional_sweep_params

        readout_calibration_results = _circuits_with_readout_benchmarking_errors_sweep(
            sampler, input_circuits, readout_qubits, sweep_params, readout_benchmarking_params, rng
        )

    # Check that the readout_calibration_results is empty
    assert len(readout_calibration_results.items()) == 0


@pytest.mark.parametrize("mode", ["shuffled", "sweep"])
def test_circuits_with_readout_benchmarking_no_qubits_arg_empty_rng(mode: str) -> None:
    """Test benchmarking when the `qubits` argument is not provided."""
    qubits = cirq.LineQubit.range(3)
    sampler = NoisySingleQubitReadoutSampler(p0=0.1, p1=0.2, seed=1234)
    circuit_repetitions = 1
    readout_repetitions = 1000
    num_random_bitstrings = 100

    readout_benchmarking_params = sc_readout.ReadoutBenchmarkingParams(
        circuit_repetitions=circuit_repetitions,
        num_random_bitstrings=num_random_bitstrings,
        readout_repetitions=readout_repetitions,
    )

    if mode == "shuffled":
        input_circuits = _create_test_circuits(qubits, 3)
        measurements, readout_calibration_results = (
            sc_readout.run_shuffled_circuits_with_readout_benchmarking(
                sampler, input_circuits, readout_benchmarking_params, None, None
            )
        )
        assert len(measurements) == len(input_circuits)
    else:  # mode == "sweep"
        input_circuits, sweep_params = _create_test_circuits_with_sweep(qubits, 3)
        sweep_measurements, readout_calibration_results = (
            sc_readout.run_sweep_with_readout_benchmarking(
                sampler, input_circuits, sweep_params, readout_benchmarking_params, None, None
            )
        )
        assert len(sweep_measurements) == len(input_circuits)

    # When qubits is None, all qubits from input circuits are benchmarked as one group.
    assert len(readout_calibration_results) == 1
    qlist, result = list(readout_calibration_results.items())[0]
    assert isinstance(qlist, tuple)
    assert set(qlist) == set(qubits)
    assert isinstance(result, SingleQubitReadoutCalibrationResult)
    for error in result.zero_state_errors.values():
        assert 0.08 < error < 0.12
    for error in result.one_state_errors.values():
        assert 0.18 < error < 0.22
    assert result.repetitions == readout_repetitions


def test_deprecated_run_shuffled_with_readout_benchmarking() -> None:
    """Test that the deprecated function works correctly and is covered."""
    qubits = cirq.LineQubit.range(3)
    input_circuits = _create_test_circuits(qubits, 3)
    sampler = NoisySingleQubitReadoutSampler(p0=0.1, p1=0.2, seed=1234)
    circuit_repetitions = 1
    readout_repetitions = 1000
    num_random_bitstrings = 100

    # Test with an integer seed.
    with cirq.testing.assert_deprecated(deadline='v1.8', count=1):
        measurements_seed, results_seed = sc_readout.run_shuffled_with_readout_benchmarking(
            input_circuits=input_circuits,
            sampler=sampler,
            circuit_repetitions=circuit_repetitions,
            rng_or_seed=123,
            num_random_bitstrings=num_random_bitstrings,
            readout_repetitions=readout_repetitions,
            qubits=qubits,
        )
    assert len(measurements_seed) == len(input_circuits)
    qlist, result = list(results_seed.items())[0]
    assert tuple(qubits) == qlist
    for error in result.zero_state_errors.values():
        assert 0.08 < error < 0.12
    for error in result.one_state_errors.values():
        assert 0.18 < error < 0.22

    # Test with qubits=None to cover the auto-detection branch.
    with cirq.testing.assert_deprecated(deadline='v1.8', count=1):
        _, results_none = sc_readout.run_shuffled_with_readout_benchmarking(
            input_circuits=input_circuits,
            sampler=sampler,
            circuit_repetitions=circuit_repetitions,
            rng_or_seed=123,
            num_random_bitstrings=num_random_bitstrings,
            readout_repetitions=readout_repetitions,
            qubits=None,
        )
    qlist_none, _ = list(results_none.items())[0]
    assert set(qlist_none) == set(qubits)

    # Test circuit_repetitions must be > 0
    with cirq.testing.assert_deprecated(deadline="v1.8", count=1):
        with pytest.raises(ValueError, match="Must provide non-zero circuit_repetitions."):
            sc_readout.run_shuffled_with_readout_benchmarking(
                input_circuits,
                sampler,
                circuit_repetitions=0,
                num_random_bitstrings=5,
                readout_repetitions=100,
                rng_or_seed=123,
            )

    # Test num_random_bitstrings must be >= 0
    with cirq.testing.assert_deprecated(deadline="v1.8", count=1):
        with pytest.raises(ValueError, match="Must provide zero or more num_random_bitstrings."):
            sc_readout.run_shuffled_with_readout_benchmarking(
                input_circuits,
                sampler,
                circuit_repetitions=10,
                num_random_bitstrings=-1,
                readout_repetitions=100,
                rng_or_seed=123,
            )

    # Test readout_repetitions must be > 0
    with cirq.testing.assert_deprecated(deadline="v1.8", count=1):
        with pytest.raises(
            ValueError, match="Must provide non-zero readout_repetitions for readout calibration."
        ):
            sc_readout.run_shuffled_with_readout_benchmarking(
                input_circuits,
                sampler,
                circuit_repetitions=10,
                num_random_bitstrings=1,
                readout_repetitions=0,
                rng_or_seed=123,
            )


def test_empty_input_circuits() -> None:
    """Test that the input circuits are empty."""
    readout_benchmarking_params = sc_readout.ReadoutBenchmarkingParams(
        circuit_repetitions=10, num_random_bitstrings=5, readout_repetitions=100
    )
    with pytest.raises(ValueError, match="Input circuits must not be empty."):
        sc_readout.run_shuffled_circuits_with_readout_benchmarking(
            cirq.ZerosSampler(),
            [],
            readout_benchmarking_params,
            rng_or_seed=np.random.default_rng(456),
        )


def test_non_circuit_input() -> None:
    """Test that the input circuits are not of type cirq.Circuit."""
    q = cirq.LineQubit(0)
    readout_benchmarking_params = sc_readout.ReadoutBenchmarkingParams(
        circuit_repetitions=10, num_random_bitstrings=5, readout_repetitions=100
    )
    with pytest.raises(ValueError, match="Input circuits must be of type cirq.Circuit."):
        sc_readout.run_shuffled_circuits_with_readout_benchmarking(
            cirq.ZerosSampler(),
            [q],  # type: ignore[list-item]
            readout_benchmarking_params,
            rng_or_seed=np.random.default_rng(456),
        )


def test_no_measurements() -> None:
    """Test that the input circuits don't have measurements."""
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q))
    readout_benchmarking_params = sc_readout.ReadoutBenchmarkingParams(
        circuit_repetitions=10, num_random_bitstrings=5, readout_repetitions=100
    )
    with pytest.raises(ValueError, match="Input circuits must have measurements."):
        sc_readout.run_shuffled_circuits_with_readout_benchmarking(
            cirq.ZerosSampler(),
            [circuit],
            readout_benchmarking_params,
            rng_or_seed=np.random.default_rng(456),
        )


def test_zero_circuit_repetitions() -> None:
    """Test that the circuit repetitions are zero."""
    with pytest.raises(ValueError, match="Must provide non-zero circuit_repetitions."):
        sc_readout.ReadoutBenchmarkingParams(
            circuit_repetitions=0, num_random_bitstrings=5, readout_repetitions=100
        )


def test_mismatch_circuit_repetitions() -> None:
    """Test that the number of circuit repetitions don't match the number of input circuits."""
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q), cirq.measure(q))
    readout_benchmarking_params = sc_readout.ReadoutBenchmarkingParams(
        circuit_repetitions=[10, 20], num_random_bitstrings=5, readout_repetitions=100
    )
    with pytest.raises(
        ValueError,
        match="Number of circuit_repetitions must match the number of" + " input circuits.",
    ):
        sc_readout.run_shuffled_circuits_with_readout_benchmarking(
            cirq.ZerosSampler(),
            [circuit],
            readout_benchmarking_params,
            rng_or_seed=np.random.default_rng(456),
        )


def test_zero_num_random_bitstrings() -> None:
    """Test that the number of random bitstrings is smaller than zero."""
    with pytest.raises(ValueError, match="Must provide zero or more num_random_bitstrings."):
        sc_readout.ReadoutBenchmarkingParams(
            circuit_repetitions=10, num_random_bitstrings=-1, readout_repetitions=100
        )


def test_zero_readout_repetitions() -> None:
    """Test that the readout repetitions is zero."""
    with pytest.raises(
        ValueError, match="Must provide non-zero readout_repetitions for readout" + " calibration."
    ):
        sc_readout.ReadoutBenchmarkingParams(
            circuit_repetitions=10, num_random_bitstrings=5, readout_repetitions=0
        )


def test_empty_sweep_params() -> None:
    """Test that the sweep params are empty."""
    q = cirq.LineQubit(5)
    circuit = cirq.Circuit(cirq.H(q))
    readout_benchmarking_params = sc_readout.ReadoutBenchmarkingParams(
        circuit_repetitions=10, num_random_bitstrings=5, readout_repetitions=100
    )
    with pytest.raises(ValueError, match="Sweep parameters must not be empty."):
        sc_readout.run_sweep_with_readout_benchmarking(
            cirq.ZerosSampler(),
            [circuit],
            [],
            readout_benchmarking_params,
            rng_or_seed=np.random.default_rng(456),
        )
