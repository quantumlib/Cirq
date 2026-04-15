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
import random
from collections.abc import Sequence

import numpy as np
import pytest

import cirq
from cirq.contrib.paulistring import (
    CircuitToPauliStringsParameters,
    measure_pauli_strings,
    PostFilteringSymmetryCalibrationResult,
)
from cirq.experiments import SingleQubitReadoutCalibrationResult
from cirq.experiments.single_qubit_readout_calibration_test import NoisySingleQubitReadoutSampler


def _create_ghz(number_of_qubits: int, qubits: Sequence[cirq.Qid]) -> cirq.Circuit:
    ghz_circuit = cirq.Circuit(
        cirq.H(qubits[0]),
        *[cirq.CNOT(qubits[i - 1], qubits[i]) for i in range(1, number_of_qubits)],
    )
    return ghz_circuit


def _generate_random_pauli_string(
    qubits: Sequence[cirq.Qid], enable_coeff: bool = False, allow_pauli_i: bool = True
):
    pauli_ops = [cirq.I, cirq.X, cirq.Y, cirq.Z]

    if not allow_pauli_i:
        pauli_ops = [cirq.X, cirq.Y, cirq.Z]

    operators = {q: random.choice(pauli_ops) for q in qubits}
    # Ensure at least one non-identity.
    operators[random.choice(qubits)] = random.choice(pauli_ops[1:])

    if enable_coeff:
        coefficient = (2 * random.random() - 1) * 100
        return coefficient * cirq.PauliString(operators)
    return cirq.PauliString(operators)


def _generate_qwc_paulis(
    input_pauli: cirq.PauliString, num_output: int, exclude_input_pauli: bool = False
) -> Sequence[cirq.PauliString]:
    """Generates PauliStrings that are Qubit-Wise Commuting (QWC)
    with the input_pauli.

    All operations in input_pauli must not be pauli I.
    """
    allowed_paulis_per_qubit = []
    qubits = input_pauli.qubits

    for qubit in qubits:
        pauli_op = input_pauli.get(qubit, cirq.I)

        allowed_pauli_op = []
        if pauli_op == cirq.I:
            allowed_pauli_op = [cirq.I, cirq.X, cirq.Y, cirq.Z]  # pragma: no cover
        elif pauli_op == cirq.X:
            allowed_pauli_op = [cirq.I, cirq.X]
        elif pauli_op == cirq.Y:
            allowed_pauli_op = [cirq.I, cirq.Y]
        elif pauli_op == cirq.Z:
            allowed_pauli_op = [cirq.I, cirq.Z]

        allowed_paulis_per_qubit.append(allowed_pauli_op)

    qwc_paulis: list[cirq.PauliString] = []

    for pauli_combination in itertools.product(*allowed_paulis_per_qubit):
        pauli_dict = {}
        for i, qid in enumerate(qubits):
            pauli_dict[qid] = pauli_combination[i]

        qwc_pauli: cirq.PauliString = cirq.PauliString(pauli_dict)
        if exclude_input_pauli and qwc_pauli == input_pauli:
            continue  # pragma: no cover
        if all(q == cirq.I for q in qwc_pauli):
            continue
        qwc_paulis.append(qwc_pauli)

    return qwc_paulis if num_output > len(qwc_paulis) else random.sample(qwc_paulis, num_output)


def _ideal_expectation_based_on_pauli_string(
    pauli_string: cirq.PauliString, final_state_vector: np.ndarray
) -> float:
    return pauli_string.expectation_from_state_vector(
        final_state_vector, qubit_map={q: i for i, q in enumerate(pauli_string.qubits)}
    )


@pytest.mark.parametrize("use_sweep", [True, False])
def test_pauli_string_measurement_errors_no_noise(use_sweep: bool) -> None:
    """Test that the mitigated expectation is close to the ideal expectation
    based on the Pauli string"""

    qubits = cirq.LineQubit.range(5)
    circuit = cirq.FrozenCircuit(_create_ghz(5, qubits))
    sampler = cirq.Simulator()

    circuits_to_pauli: dict[cirq.FrozenCircuit, Sequence[cirq.PauliString]] = {}
    circuits_to_pauli[circuit] = [_generate_random_pauli_string(qubits) for _ in range(3)]

    circuits_with_pauli_expectations = measure_pauli_strings(
        circuits_to_pauli, sampler, 300, 300, 300, 1234, use_sweep
    )

    for circuit_with_pauli_expectations in circuits_with_pauli_expectations:
        assert isinstance(circuit_with_pauli_expectations.circuit, cirq.FrozenCircuit)

        expected_val_simulation = sampler.simulate(
            circuit_with_pauli_expectations.circuit.unfreeze()
        )
        final_state_vector = expected_val_simulation.final_state_vector

        for pauli_string_measurement_results in circuit_with_pauli_expectations.results:
            # Since there is no noise, the mitigated and unmitigated expectations should be the same
            assert np.isclose(
                pauli_string_measurement_results.mitigated_expectation,
                pauli_string_measurement_results.unmitigated_expectation,
            )
            assert np.isclose(
                pauli_string_measurement_results.mitigated_expectation,
                _ideal_expectation_based_on_pauli_string(
                    pauli_string_measurement_results.pauli_string, final_state_vector
                ),
                atol=10 * pauli_string_measurement_results.mitigated_stddev,
            )
            assert isinstance(
                pauli_string_measurement_results.calibration_result,
                SingleQubitReadoutCalibrationResult,
            )
            assert (
                pauli_string_measurement_results.calibration_result.zero_state_errors
                == dict.fromkeys(pauli_string_measurement_results.pauli_string.qubits, 0)
            )
            assert (
                pauli_string_measurement_results.calibration_result.one_state_errors
                == dict.fromkeys(pauli_string_measurement_results.pauli_string.qubits, 0)
            )


@pytest.mark.parametrize("use_sweep", [True, False])
def test_group_pauli_string_measurement_errors_no_noise_with_coefficient(use_sweep: bool) -> None:
    """Test that the mitigated expectation is close to the ideal expectation
    based on the group of Pauli strings"""

    qubits = cirq.LineQubit.range(5)
    circuit = cirq.FrozenCircuit(_create_ghz(5, qubits))
    sampler = cirq.Simulator()

    circuits_to_pauli: dict[cirq.FrozenCircuit, list[Sequence[cirq.PauliString]]] = {}
    circuits_to_pauli[circuit] = [
        _generate_qwc_paulis(
            _generate_random_pauli_string(qubits, enable_coeff=True, allow_pauli_i=False), 10, True
        )
        for _ in range(3)
    ]
    circuits_to_pauli[circuit].append([cirq.PauliString(dict.fromkeys(qubits, cirq.X))])

    circuits_with_pauli_expectations = measure_pauli_strings(
        circuits_to_pauli, sampler, 300, 300, 300, 1234, use_sweep
    )

    for circuit_with_pauli_expectations in circuits_with_pauli_expectations:
        assert isinstance(circuit_with_pauli_expectations.circuit, cirq.FrozenCircuit)

        expected_val_simulation = sampler.simulate(
            circuit_with_pauli_expectations.circuit.unfreeze()
        )
        final_state_vector = expected_val_simulation.final_state_vector

        for pauli_string_measurement_results in circuit_with_pauli_expectations.results:
            # Since there is no noise, the mitigated and unmitigated expectations should be the same
            assert np.isclose(
                pauli_string_measurement_results.mitigated_expectation,
                pauli_string_measurement_results.unmitigated_expectation,
            )
            assert np.isclose(
                pauli_string_measurement_results.mitigated_expectation,
                _ideal_expectation_based_on_pauli_string(
                    pauli_string_measurement_results.pauli_string, final_state_vector
                ),
                atol=10 * pauli_string_measurement_results.mitigated_stddev,
            )
            assert isinstance(
                pauli_string_measurement_results.calibration_result,
                SingleQubitReadoutCalibrationResult,
            )
            assert (
                pauli_string_measurement_results.calibration_result.zero_state_errors
                == dict.fromkeys(pauli_string_measurement_results.pauli_string.qubits, 0)
            )
            assert (
                pauli_string_measurement_results.calibration_result.one_state_errors
                == dict.fromkeys(pauli_string_measurement_results.pauli_string.qubits, 0)
            )


@pytest.mark.parametrize("use_sweep", [True, False])
def test_pauli_string_measurement_errors_with_noise(use_sweep: bool) -> None:
    """Test that the mitigated expectation is close to the ideal expectation
    based on the Pauli string"""
    qubits = cirq.LineQubit.range(7)
    circuit = cirq.FrozenCircuit(_create_ghz(7, qubits))
    sampler = NoisySingleQubitReadoutSampler(p0=0.01, p1=0.05, seed=1234)
    simulator = cirq.Simulator()

    circuits_to_pauli: dict[cirq.FrozenCircuit, Sequence[cirq.PauliString]] = {}
    circuits_to_pauli[circuit] = [_generate_random_pauli_string(qubits) for _ in range(3)]

    circuits_with_pauli_expectations = measure_pauli_strings(
        circuits_to_pauli, sampler, 300, 300, 300, np.random.default_rng(), use_sweep
    )

    for circuit_with_pauli_expectations in circuits_with_pauli_expectations:
        assert isinstance(circuit_with_pauli_expectations.circuit, cirq.FrozenCircuit)

        expected_val_simulation = simulator.simulate(
            circuit_with_pauli_expectations.circuit.unfreeze()
        )
        final_state_vector = expected_val_simulation.final_state_vector

        for pauli_string_measurement_results in circuit_with_pauli_expectations.results:
            assert np.isclose(
                pauli_string_measurement_results.mitigated_expectation,
                _ideal_expectation_based_on_pauli_string(
                    pauli_string_measurement_results.pauli_string, final_state_vector
                ),
                atol=10 * pauli_string_measurement_results.mitigated_stddev,
            )

            assert isinstance(
                pauli_string_measurement_results.calibration_result,
                SingleQubitReadoutCalibrationResult,
            )

            for (
                error
            ) in pauli_string_measurement_results.calibration_result.zero_state_errors.values():
                assert 0.008 < error < 0.012
            for (
                error
            ) in pauli_string_measurement_results.calibration_result.one_state_errors.values():
                assert 0.045 < error < 0.055


@pytest.mark.parametrize("use_sweep", [True, False])
def test_group_pauli_string_measurement_errors_with_noise(use_sweep: bool) -> None:
    """Test that the mitigated expectation is close to the ideal expectation
    based on the group Pauli strings"""
    qubits = cirq.LineQubit.range(7)
    circuit = cirq.FrozenCircuit(_create_ghz(7, qubits))
    sampler = NoisySingleQubitReadoutSampler(p0=0.01, p1=0.05, seed=1234)
    simulator = cirq.Simulator()

    circuits_to_pauli: dict[cirq.FrozenCircuit, Sequence[Sequence[cirq.PauliString]]] = {}
    circuits_to_pauli[circuit] = [
        _generate_qwc_paulis(
            _generate_random_pauli_string(qubits, enable_coeff=True, allow_pauli_i=False), 5
        )
    ]

    circuits_with_pauli_expectations = measure_pauli_strings(
        circuits_to_pauli, sampler, 300, 300, 300, np.random.default_rng(), use_sweep
    )

    for circuit_with_pauli_expectations in circuits_with_pauli_expectations:
        assert isinstance(circuit_with_pauli_expectations.circuit, cirq.FrozenCircuit)

        expected_val_simulation = simulator.simulate(
            circuit_with_pauli_expectations.circuit.unfreeze()
        )
        final_state_vector = expected_val_simulation.final_state_vector

        for pauli_string_measurement_results in circuit_with_pauli_expectations.results:
            assert np.isclose(
                pauli_string_measurement_results.mitigated_expectation,
                _ideal_expectation_based_on_pauli_string(
                    pauli_string_measurement_results.pauli_string, final_state_vector
                ),
                atol=10 * pauli_string_measurement_results.mitigated_stddev,
            )

            assert isinstance(
                pauli_string_measurement_results.calibration_result,
                SingleQubitReadoutCalibrationResult,
            )

            for (
                error
            ) in pauli_string_measurement_results.calibration_result.zero_state_errors.values():
                assert 0.008 < error < 0.012
            for (
                error
            ) in pauli_string_measurement_results.calibration_result.one_state_errors.values():
                assert 0.045 < error < 0.055


@pytest.mark.parametrize("use_sweep", [True, False])
def test_many_circuits_mixed_mitigation_types(use_sweep: bool) -> None:
    """Test mixed input: some circuits using confusion matrices, some using symmetries.

    This test specifically includes a QWC group with multiple Pauli strings to
    ensure the processing logic handles nested groups correctly.
    """
    qubits_1 = cirq.LineQubit.range(3)
    qubits_2 = cirq.LineQubit.range(5)
    qubits_3 = cirq.LineQubit.range(2)

    # Circuit 1 is a Standard GHZ without symmetry.
    circuit_1 = cirq.FrozenCircuit(_create_ghz(3, qubits_1))
    params_1 = CircuitToPauliStringsParameters(
        circuit=circuit_1, pauli_strings=((cirq.PauliString(cirq.Z(qubits_1[0])),),)
    )

    # Circuit 2 is a GHZ with a known symmetry (Z0*Z1 = 1)
    circuit_2 = cirq.FrozenCircuit(_create_ghz(5, qubits_2))
    symmetry: cirq.PauliString = cirq.PauliString(cirq.Z(qubits_2[0]) * cirq.Z(qubits_2[1]))

    pauli_group: tuple[cirq.PauliString, ...] = (
        cirq.PauliString(cirq.Z(qubits_2[0])),
        2.0 * cirq.PauliString(cirq.Z(qubits_2[1])),
    )

    params_sym = CircuitToPauliStringsParameters(
        circuit=circuit_2, pauli_strings=(pauli_group,), postselection_symmetries=((symmetry, 1),)
    )

    # Circuit 3 is a |+>|+> state with a PauliSum symmetry (X0 + X1 = 2).
    circuit_3 = cirq.FrozenCircuit(cirq.H(qubits_3[0]), cirq.H(qubits_3[1]))
    symmetry_sum = cirq.X(qubits_3[0]) + cirq.X(qubits_3[1])
    params_sym_sum = CircuitToPauliStringsParameters(
        circuit=circuit_3,
        pauli_strings=((2.0 * cirq.PauliString(cirq.X(qubits_3[0])),),),
        postselection_symmetries=((symmetry_sum, 2),),
    )

    sampler = NoisySingleQubitReadoutSampler(p0=0.01, p1=0.02, seed=1234)
    simulator = cirq.Simulator()

    results = measure_pauli_strings(
        [params_1, params_sym, params_sym_sum],
        sampler,
        pauli_repetitions=500,
        readout_repetitions=500,
        num_random_bitstrings=100,
        rng_or_seed=1234,
        use_sweep=use_sweep,
    )

    assert len(results) == 3

    for circuit_res in results:
        # For Circuit 2, we expect exactly 2 results because the group had 2 Pauli strings
        if circuit_res.circuit == circuit_2:
            assert len(circuit_res.results) == 2

        # Simulate the ideal circuit to extract the ground truth state vector
        expected_val_simulation = simulator.simulate(circuit_res.circuit.unfreeze())
        final_state_vector = expected_val_simulation.final_state_vector

        for res in circuit_res.results:
            # Calculate the ideal expectation directly from the state vector and Pauli string
            ideal_expectation = _ideal_expectation_based_on_pauli_string(
                res.pauli_string, final_state_vector
            )

            # Assert the mitigated result falls statistically close to the ideal simulation
            assert np.isclose(
                res.mitigated_expectation, ideal_expectation, atol=10 * res.mitigated_stddev
            )

            # Maintain type validations based on whether symmetries were used
            if circuit_res.circuit in [circuit_2, circuit_3]:
                assert isinstance(res.calibration_result, PostFilteringSymmetryCalibrationResult)
            else:
                assert isinstance(res.calibration_result, SingleQubitReadoutCalibrationResult)


@pytest.mark.parametrize("use_sweep", [True, False])
def test_allow_group_pauli_measurement_without_readout_mitigation(use_sweep: bool) -> None:
    """Test that the function allows to measure without error mitigation"""
    qubits = cirq.LineQubit.range(7)
    circuit = cirq.FrozenCircuit(_create_ghz(7, qubits))
    sampler = NoisySingleQubitReadoutSampler(p0=0.01, p1=0.005, seed=1234)

    circuits_to_pauli: dict[cirq.FrozenCircuit, list[Sequence[cirq.PauliString]]] = {}
    circuits_to_pauli[circuit] = [
        _generate_qwc_paulis(_generate_random_pauli_string(qubits, True), 2, True),
        _generate_qwc_paulis(_generate_random_pauli_string(qubits), 4),
        _generate_qwc_paulis(_generate_random_pauli_string(qubits), 6),
    ]

    circuits_with_pauli_expectations = measure_pauli_strings(
        circuits_to_pauli, sampler, 300, 300, 0, np.random.default_rng(), use_sweep
    )

    for circuit_with_pauli_expectations in circuits_with_pauli_expectations:
        assert isinstance(circuit_with_pauli_expectations.circuit, cirq.FrozenCircuit)

        for pauli_string_measurement_results in circuit_with_pauli_expectations.results:
            # Since there's no mitigation, the mitigated and unmitigated expectations
            # should be the same
            assert np.isclose(
                pauli_string_measurement_results.mitigated_expectation,
                pauli_string_measurement_results.unmitigated_expectation,
            )
            assert pauli_string_measurement_results.calibration_result is None


@pytest.mark.parametrize("use_sweep", [True, False])
@pytest.mark.parametrize(
    "insert_strategy", [cirq.InsertStrategy.INLINE, cirq.InsertStrategy.EARLIEST]
)
def test_many_circuits_with_coefficient(
    use_sweep: bool, insert_strategy: cirq.InsertStrategy
) -> None:
    """Test that the mitigated expectation is close to the ideal expectation
    based on the Pauli string for multiple circuits"""
    qubits_1 = cirq.LineQubit.range(3)
    qubits_2 = [
        cirq.GridQubit(0, 1),
        cirq.GridQubit(1, 1),
        cirq.GridQubit(1, 0),
        cirq.GridQubit(1, 2),
        cirq.GridQubit(2, 1),
    ]
    qubits_3 = cirq.LineQubit.range(8)

    circuit_1 = cirq.FrozenCircuit(_create_ghz(3, qubits_1))
    circuit_2 = cirq.FrozenCircuit(_create_ghz(5, qubits_2))
    circuit_3 = cirq.FrozenCircuit(_create_ghz(8, qubits_3))

    circuits_to_pauli: dict[cirq.FrozenCircuit, list[cirq.PauliString]] = {}
    circuits_to_pauli[circuit_1] = [_generate_random_pauli_string(qubits_1, True) for _ in range(3)]
    circuits_to_pauli[circuit_2] = [_generate_random_pauli_string(qubits_2, True) for _ in range(3)]
    circuits_to_pauli[circuit_3] = [_generate_random_pauli_string(qubits_3, True) for _ in range(3)]

    sampler = NoisySingleQubitReadoutSampler(p0=0.03, p1=0.05, seed=1234)
    simulator = cirq.Simulator()

    circuits_with_pauli_expectations = measure_pauli_strings(
        circuits_to_pauli,
        sampler,
        300,
        300,
        300,
        np.random.default_rng(),
        use_sweep,
        insert_strategy,
    )

    for circuit_with_pauli_expectations in circuits_with_pauli_expectations:
        assert isinstance(circuit_with_pauli_expectations.circuit, cirq.FrozenCircuit)

        expected_val_simulation = simulator.simulate(
            circuit_with_pauli_expectations.circuit.unfreeze()
        )
        final_state_vector = expected_val_simulation.final_state_vector

        for pauli_string_measurement_results in circuit_with_pauli_expectations.results:
            assert np.isclose(
                pauli_string_measurement_results.mitigated_expectation,
                _ideal_expectation_based_on_pauli_string(
                    pauli_string_measurement_results.pauli_string, final_state_vector
                ),
                atol=10 * pauli_string_measurement_results.mitigated_stddev,
            )
            assert isinstance(
                pauli_string_measurement_results.calibration_result,
                SingleQubitReadoutCalibrationResult,
            )
            for (
                error
            ) in pauli_string_measurement_results.calibration_result.zero_state_errors.values():
                assert 0.025 < error < 0.035
            for (
                error
            ) in pauli_string_measurement_results.calibration_result.one_state_errors.values():
                assert 0.045 < error < 0.055


@pytest.mark.parametrize("use_sweep", [True, False])
def test_many_group_pauli_in_circuits_with_coefficient(use_sweep: bool) -> None:
    """Test that the mitigated expectation is close to the ideal expectation
    based on the Pauli string for multiple circuits"""
    qubits_1 = cirq.LineQubit.range(3)
    qubits_2 = [
        cirq.GridQubit(0, 1),
        cirq.GridQubit(1, 1),
        cirq.GridQubit(1, 0),
        cirq.GridQubit(1, 2),
        cirq.GridQubit(2, 1),
    ]
    qubits_3 = cirq.LineQubit.range(8)

    circuit_1 = cirq.FrozenCircuit(_create_ghz(3, qubits_1))
    circuit_2 = cirq.FrozenCircuit(_create_ghz(5, qubits_2))
    circuit_3 = cirq.FrozenCircuit(_create_ghz(8, qubits_3))

    circuits_to_pauli: dict[cirq.FrozenCircuit, list[Sequence[cirq.PauliString]]] = {}
    circuits_to_pauli[circuit_1] = [
        _generate_qwc_paulis(
            _generate_random_pauli_string(qubits_1, enable_coeff=True, allow_pauli_i=False), 2
        )
    ]

    circuits_to_pauli[circuit_2] = [
        _generate_qwc_paulis(
            _generate_random_pauli_string(qubits_2, enable_coeff=True, allow_pauli_i=False), 2
        )
    ]

    circuits_to_pauli[circuit_3] = [
        _generate_qwc_paulis(
            _generate_random_pauli_string(qubits_3, enable_coeff=True, allow_pauli_i=False), 2
        )
    ]

    sampler = NoisySingleQubitReadoutSampler(p0=0.03, p1=0.05, seed=1234)
    simulator = cirq.Simulator()

    circuits_with_pauli_expectations = measure_pauli_strings(
        circuits_to_pauli,
        sampler,
        300,
        300,
        300,
        np.random.default_rng(),
        use_sweep,
        measure_on_full_support=True,
    )

    for circuit_with_pauli_expectations in circuits_with_pauli_expectations:
        assert isinstance(circuit_with_pauli_expectations.circuit, cirq.FrozenCircuit)

        expected_group_count = len(circuits_to_pauli[circuit_with_pauli_expectations.circuit][0])

        assert len(circuit_with_pauli_expectations.results) == expected_group_count, (
            f"Expected {expected_group_count} results (groups) for circuit, "
            f"but got {len(circuit_with_pauli_expectations.results)}."
        )

        expected_val_simulation = simulator.simulate(
            circuit_with_pauli_expectations.circuit.unfreeze()
        )
        final_state_vector = expected_val_simulation.final_state_vector

        for pauli_string_measurement_results in circuit_with_pauli_expectations.results:
            assert np.isclose(
                pauli_string_measurement_results.mitigated_expectation,
                _ideal_expectation_based_on_pauli_string(
                    pauli_string_measurement_results.pauli_string, final_state_vector
                ),
                atol=10 * pauli_string_measurement_results.mitigated_stddev,
            )
            assert isinstance(
                pauli_string_measurement_results.calibration_result,
                SingleQubitReadoutCalibrationResult,
            )
            for (
                error
            ) in pauli_string_measurement_results.calibration_result.zero_state_errors.values():
                assert 0.025 < error < 0.035
            for (
                error
            ) in pauli_string_measurement_results.calibration_result.one_state_errors.values():
                assert 0.045 < error < 0.055


def test_coefficient_not_real_number() -> None:
    """Test that the coefficient of input pauli string is not real.
    Should return error in this case"""
    qubits_1 = cirq.LineQubit.range(3)
    random_pauli_string = _generate_random_pauli_string(qubits_1, True) * (3 + 4j)
    circuit_1 = cirq.FrozenCircuit(_create_ghz(3, qubits_1))

    circuits_to_pauli: dict[cirq.FrozenCircuit, list[cirq.PauliString]] = {}
    circuits_to_pauli[circuit_1] = [
        random_pauli_string,
        _generate_random_pauli_string(qubits_1, True),
        _generate_random_pauli_string(qubits_1, True),
    ]

    with pytest.raises(
        ValueError,
        match="Cannot compute expectation value of a "
        "non-Hermitian PauliString. Coefficient must be real.",
    ):
        measure_pauli_strings(
            circuits_to_pauli, cirq.Simulator(), 300, 300, 300, np.random.default_rng()
        )


def test_empty_input_circuits_to_pauli_mapping() -> None:
    """Test that the input circuits are empty."""

    with pytest.raises(ValueError, match="Input circuits_to_pauli parameter must not be empty"):
        measure_pauli_strings([], cirq.Simulator(), 300, 300, 300, np.random.default_rng())


def test_invalid_input_container_type() -> None:
    """Test that passing an invalid container type raises TypeError."""
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.FrozenCircuit(_create_ghz(2, qubits))

    invalid_input = {circuit}

    qubits_to_pauli: dict[tuple, list[cirq.PauliString]] = {}
    qubits_to_pauli[tuple(qubits)] = [cirq.PauliString(dict.fromkeys(qubits, cirq.X))]
    with pytest.raises(TypeError, match="Input must be a dict or a list"):
        measure_pauli_strings(
            invalid_input, cirq.Simulator(), 100, 100, 100, np.random.default_rng()  # type: ignore
        )


def test_circuit_parameters_validation_errors() -> None:
    """Test validation errors specific to CircuitToPauliStringsParameters attributes."""
    q0 = cirq.LineQubit(0)
    valid_circuit = cirq.FrozenCircuit(cirq.Circuit(cirq.X(q0)))
    valid_pauli: list[list[cirq.PauliString]] = [[cirq.PauliString(cirq.Z(q0))]]

    sampler = cirq.Simulator()
    rng = np.random.default_rng()

    # Test empty circuit
    params_empty_circuit = CircuitToPauliStringsParameters(
        circuit=cirq.FrozenCircuit(),  # Empty
        pauli_strings=valid_pauli,
        postselection_symmetries=[],
    )
    with pytest.raises(ValueError, match="Circuit must not be empty"):
        measure_pauli_strings([params_empty_circuit], sampler, 10, 10, 10, rng)

    # Test Invalid Type for Circuit
    params_invalid_circuit_type = CircuitToPauliStringsParameters(
        circuit="NotACircuit",  # type: ignore
        pauli_strings=valid_pauli,
        postselection_symmetries=[],
    )
    with pytest.raises(TypeError, match="Expected circuit to be FrozenCircuit"):
        measure_pauli_strings([params_invalid_circuit_type], sampler, 10, 10, 10, rng)

    # Test Invalid Type in Pauli Strings
    params_invalid_type = CircuitToPauliStringsParameters(
        circuit=valid_circuit, pauli_strings=[["NotAPauliString"]], postselection_symmetries=[]
    )
    with pytest.raises(
        TypeError, match=r"Expected all elements to be Sequence\[Sequence\[ops.PauliString\]\]"
    ):
        measure_pauli_strings([params_invalid_type], sampler, 10, 10, 10, rng)


def test_all_pauli_strings_are_pauli_i() -> None:
    """Test that all input pauli are pauli I"""
    qubits_1 = cirq.LineQubit.range(5)
    qubits_2 = [
        cirq.GridQubit(0, 1),
        cirq.GridQubit(1, 1),
        cirq.GridQubit(1, 0),
        cirq.GridQubit(1, 2),
        cirq.GridQubit(2, 1),
    ]

    circuit_1 = cirq.FrozenCircuit(_create_ghz(5, qubits_1))
    circuit_2 = cirq.FrozenCircuit(_create_ghz(5, qubits_2))

    circuits_to_pauli: dict[cirq.FrozenCircuit, list[cirq.PauliString]] = {}
    circuits_to_pauli[circuit_1] = [
        cirq.PauliString(dict.fromkeys(qubits_1, cirq.I)),
        cirq.PauliString(dict.fromkeys(qubits_1, cirq.X)),
    ]
    circuits_to_pauli[circuit_2] = [cirq.PauliString(dict.fromkeys(qubits_2, cirq.X))]

    with pytest.raises(
        ValueError,
        match="Empty Pauli strings or Pauli strings consisting "
        "only of Pauli I are not allowed. Please provide "
        "valid input Pauli strings.",
    ):
        measure_pauli_strings(
            circuits_to_pauli, cirq.Simulator(), 300, 300, 300, np.random.default_rng()
        )


def test_zero_pauli_repetitions() -> None:
    """Test that the pauli repetitions are zero."""
    qubits = cirq.LineQubit.range(5)

    circuit = cirq.FrozenCircuit(_create_ghz(5, qubits))

    circuits_to_pauli: dict[cirq.FrozenCircuit, list[cirq.PauliString]] = {}
    circuits_to_pauli[circuit] = [cirq.PauliString(dict.fromkeys(qubits, cirq.X))]
    with pytest.raises(ValueError, match="Must provide positive pauli_repetitions."):
        measure_pauli_strings(
            circuits_to_pauli, cirq.Simulator(), 0, 300, 300, np.random.default_rng()
        )


def test_negative_num_random_bitstrings() -> None:
    """Test that the number of random bitstrings is smaller than zero."""
    qubits = cirq.LineQubit.range(5)

    circuit = cirq.FrozenCircuit(_create_ghz(5, qubits))

    circuits_to_pauli: dict[cirq.FrozenCircuit, list[cirq.PauliString]] = {}
    circuits_to_pauli[circuit] = [cirq.PauliString(dict.fromkeys(qubits, cirq.X))]
    with pytest.raises(ValueError, match="Must provide zero or more num_random_bitstrings."):
        measure_pauli_strings(
            circuits_to_pauli, cirq.Simulator(), 300, 300, -1, np.random.default_rng()
        )


def test_zero_readout_repetitions() -> None:
    """Test that the readout repetitions is zero."""
    qubits = cirq.LineQubit.range(5)

    circuit = cirq.FrozenCircuit(_create_ghz(5, qubits))

    circuits_to_pauli: dict[cirq.FrozenCircuit, list[cirq.PauliString]] = {}
    circuits_to_pauli[circuit] = [cirq.PauliString(dict.fromkeys(qubits, cirq.X))]
    with pytest.raises(
        ValueError, match="Must provide positive readout_repetitions for readout" + " calibration."
    ):
        measure_pauli_strings(
            circuits_to_pauli, cirq.Simulator(), 300, 0, 300, np.random.default_rng()
        )


def test_rng_type_mismatch() -> None:
    """Test that the rng is not a numpy random generator or a seed."""
    qubits = cirq.LineQubit.range(5)

    circuit = cirq.FrozenCircuit(_create_ghz(5, qubits))

    circuits_to_pauli: dict[cirq.FrozenCircuit, list[cirq.PauliString]] = {}
    circuits_to_pauli[circuit] = [cirq.PauliString(dict.fromkeys(qubits, cirq.X))]
    with pytest.raises(ValueError, match="Must provide a numpy random generator or a seed"):
        measure_pauli_strings(
            circuits_to_pauli, cirq.Simulator(), 300, 300, 300, "test"  # type: ignore[arg-type]
        )


def test_group_paulis_are_not_qwc() -> None:
    """Test that the group paulis are not qwc."""
    qubits = cirq.LineQubit.range(5)

    circuit = cirq.FrozenCircuit(_create_ghz(5, qubits))

    pauli_str1: cirq.PauliString = cirq.PauliString({qubits[0]: cirq.X, qubits[1]: cirq.Y})
    pauli_str2: cirq.PauliString = cirq.PauliString({qubits[0]: cirq.Y})

    circuits_to_pauli: dict[cirq.FrozenCircuit, list[cirq.PauliString]] = {}
    circuits_to_pauli[circuit] = [[pauli_str1, pauli_str2]]  # type: ignore
    with pytest.raises(ValueError, match="is not Qubit-Wise Commuting."):
        measure_pauli_strings(
            circuits_to_pauli, cirq.Simulator(), 300, 300, 300, np.random.default_rng()
        )


def test_empty_group_paulis_not_allowed() -> None:
    """Test that the group paulis are empty"""
    qubits = cirq.LineQubit.range(5)

    circuit = cirq.FrozenCircuit(_create_ghz(5, qubits))

    circuits_to_pauli: dict[cirq.FrozenCircuit, list[cirq.PauliString]] = {}
    circuits_to_pauli[circuit] = [[]]  # type: ignore
    with pytest.raises(ValueError, match="Empty group of Pauli strings is not allowed"):
        measure_pauli_strings(
            circuits_to_pauli, cirq.Simulator(), 300, 300, 300, np.random.default_rng()
        )


def test_postselection_symmetry_validation_and_logic() -> None:
    """Test validation and QWC logic for post-selection symmetries."""
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.FrozenCircuit(cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1)))

    # Target Pauli String to measure: Z0 * Z1
    target_paulis: Sequence[Sequence[cirq.PauliString]] = [
        [cirq.PauliString(cirq.Z(q0) * cirq.Z(q1))]
    ]

    sampler = cirq.Simulator()
    rng = np.random.default_rng()

    # Test Valid PauliSum Symmetry
    # Z0 commutes with Z0*Z1. Z1 commutes with Z0*Z1.
    valid_pauli_sum_sym = cirq.PauliSum.from_pauli_strings(
        [cirq.PauliString(cirq.Z(q0)), cirq.PauliString(cirq.Z(q1))]
    )
    valid_pauli_sym: cirq.PauliString = cirq.PauliString(cirq.Z(q0))
    good_symmetries: Sequence[tuple[cirq.PauliString | cirq.PauliSum, int]] = [
        (valid_pauli_sum_sym, 1),
        (valid_pauli_sym, 1),
    ]
    params_valid_sum = CircuitToPauliStringsParameters(
        circuit=circuit, pauli_strings=target_paulis, postselection_symmetries=good_symmetries
    )
    results = measure_pauli_strings([params_valid_sum], sampler, 10, 1, 0, rng)
    assert len(results) == 1
    assert len(results[0].results) == 1

    # Test PauliSum with Non-QWC Terms
    # X0 and Z0 do not commute. This is an invalid PauliSum *structure* for this context.
    invalid_structure_sum = cirq.PauliSum.from_pauli_strings(
        [cirq.PauliString(cirq.X(q0)), cirq.PauliString(cirq.Z(q0))]
    )
    params_bad_sum_structure = CircuitToPauliStringsParameters(
        circuit=circuit,
        pauli_strings=target_paulis,
        postselection_symmetries=[(invalid_structure_sum, 1)],
    )
    with pytest.raises(ValueError, match="Terms are not Qubit-Wise Commuting."):
        measure_pauli_strings([params_bad_sum_structure], sampler, 10, 10, 0, rng)

    # Test Invalid Symmetry Type
    params_bad_type = CircuitToPauliStringsParameters(
        circuit=circuit,
        pauli_strings=target_paulis,
        postselection_symmetries=[("NotASymmetry", 1)],  # type: ignore
    )
    with pytest.raises(
        TypeError, match="Postselection symmetry keys must be cirq.PauliString or cirq.PauliSum"
    ):
        measure_pauli_strings([params_bad_type], sampler, 10, 10, 0, rng)

    # Test PauliSum NOT Commuting with Target
    # X0 does not commute with Z0*Z1.
    non_commuting_sum = cirq.PauliSum.from_pauli_strings([cirq.PauliString(cirq.X(q0))])
    non_commuting_symmetries: Sequence[tuple[cirq.PauliString | cirq.PauliSum, int]] = [
        (non_commuting_sum, 1)
    ]
    params_non_commute = CircuitToPauliStringsParameters(
        circuit=circuit,
        pauli_strings=target_paulis,
        postselection_symmetries=non_commuting_symmetries,
    )
    with pytest.raises(ValueError, match="are not commuting with all Pauli"):
        measure_pauli_strings([params_non_commute], sampler, 10, 10, 0, rng)


@pytest.mark.parametrize("use_sweep", [False, True])
def test_sampler_receives_correct_circuits(use_sweep: bool) -> None:
    """Test that the sampler receives circuits with correct measurement qubits."""

    from unittest.mock import MagicMock

    from cirq.study.result import ResultDict

    qubits = cirq.LineQubit.range(5)
    circuit = cirq.FrozenCircuit(_create_ghz(5, qubits))
    pauli_qubits = qubits[1:4]  # Q1, Q2, Q3
    pauli_str: cirq.PauliString = cirq.PauliString(dict.fromkeys(pauli_qubits, cirq.X))

    # Test standard Pauli String without Symmetries
    circuits_to_pauli = {circuit: [pauli_str]}
    mock_sampler = MagicMock()

    mock_res = ResultDict(params=cirq.ParamResolver({}), measurements={"result": np.zeros((1, 3))})
    mock_sampler.run_batch.return_value = [[mock_res]]
    mock_sampler.run_sweep.return_value = [mock_res]

    measure_pauli_strings(circuits_to_pauli, mock_sampler, 10, 10, 0, 1234, use_sweep=use_sweep)

    # Determine which mock method to check
    if use_sweep:
        called_circuits = [call.kwargs['program'] for call in mock_sampler.run_sweep.call_args_list]
    else:
        called_circuits = [
            c for batch in mock_sampler.run_batch.call_args_list for c in batch.args[0]
        ]

    for called_circuit in called_circuits:
        measured = {
            q
            for op in called_circuit.all_operations()
            if isinstance(op.gate, cirq.MeasurementGate)
            for q in op.qubits
        }
        assert measured == set(pauli_qubits)

    # Now test with symmetries
    sym_qubit = qubits[0]
    params = CircuitToPauliStringsParameters(
        circuit=circuit,
        pauli_strings=[(pauli_str,)],
        postselection_symmetries=[(cirq.PauliString(cirq.Z(sym_qubit)), 1)],
    )

    mock_sampler_sym = MagicMock()
    mock_res_sym = ResultDict(
        params=cirq.ParamResolver({}), measurements={"result": np.zeros((1, 4))}
    )
    mock_sampler_sym.run_batch.return_value = [[mock_res_sym]]
    mock_sampler_sym.run_sweep.return_value = [mock_res_sym]

    measure_pauli_strings([params], mock_sampler_sym, 10, 10, 0, 1234, use_sweep=use_sweep)

    if use_sweep:
        called_circuits_sym = [
            call.kwargs['program'] for call in mock_sampler_sym.run_sweep.call_args_list
        ]
    else:
        called_circuits_sym = [
            c for batch in mock_sampler_sym.run_batch.call_args_list for c in batch.args[0]
        ]

    expected_qubits = set(pauli_qubits) | {sym_qubit}
    for called_circuit in called_circuits_sym:
        measured = {
            q
            for op in called_circuit.all_operations()
            if isinstance(op.gate, cirq.MeasurementGate)
            for q in op.qubits
        }
        assert measured == expected_qubits
