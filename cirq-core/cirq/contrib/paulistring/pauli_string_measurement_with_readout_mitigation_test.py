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
from typing import Sequence

import numpy as np
import pytest

import cirq
from cirq.contrib.paulistring import measure_pauli_strings
from cirq.experiments import SingleQubitReadoutCalibrationResult
from cirq.experiments.single_qubit_readout_calibration_test import NoisySingleQubitReadoutSampler
from cirq.contrib.paulistring.pauli_string_measurement_with_readout_mitigation import (
    CircuitToPauliStringsParameters,
)


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
) -> list[cirq.PauliString]:
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


def _commute_or_identity(
    op1: cirq.Pauli | cirq.IdentityGate, op2: cirq.Pauli | cirq.IdentityGate
) -> bool:
    if op1 == cirq.I or op2 == cirq.I:
        return True
    return op1 == op2


def _are_two_pauli_strings_qubit_wise_commuting(
    pauli_str1: cirq.PauliString,
    pauli_str2: cirq.PauliString,
    all_qubits: list[cirq.Qid] | frozenset[cirq.Qid],
) -> bool:
    for qubit in all_qubits:
        op1 = pauli_str1.get(qubit, default=cirq.I)
        op2 = pauli_str2.get(qubit, default=cirq.I)

        if not _commute_or_identity(op1, op2):
            return False
    return True


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

    circuits_to_pauli: list[CircuitToPauliStringsParameters] = []
    circuits_to_pauli.append(
        CircuitToPauliStringsParameters(
            circuit=circuit,
            pauli_strings=[_generate_random_pauli_string(qubits) for _ in range(3)],
            postselection_symmetries={},
        )
    )
    circuits_with_pauli_expectations = measure_pauli_strings(
        circuits_to_pauli, sampler, 1000, 1000, 1000, 1000, use_sweep
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
            assert pauli_string_measurement_results.calibration_result.zero_state_errors == {
                q: 0 for q in pauli_string_measurement_results.pauli_string.qubits
            }
            assert pauli_string_measurement_results.calibration_result.one_state_errors == {
                q: 0 for q in pauli_string_measurement_results.pauli_string.qubits
            }


@pytest.mark.parametrize("use_sweep", [True, False])
def test_group_pauli_string_measurement_errors_no_noise_with_coefficient(use_sweep: bool) -> None:
    """Test that the mitigated expectation is close to the ideal expectation
    based on the group of Pauli strings"""

    qubits = cirq.LineQubit.range(5)
    circuit = cirq.FrozenCircuit(_create_ghz(5, qubits))
    sampler = cirq.Simulator()

    circuits_to_pauli: dict[cirq.FrozenCircuit, list[list[cirq.PauliString]]] = {}
    circuits_to_pauli[circuit] = [
        _generate_qwc_paulis(
            _generate_random_pauli_string(qubits, enable_coeff=True, allow_pauli_i=False), 10, True
        )
    )

    circuits_with_pauli_expectations = measure_pauli_strings(
        circuits_to_pauli, sampler, 1000, 1000, 1000, 500, use_sweep
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
            assert pauli_string_measurement_results.calibration_result.zero_state_errors == {
                q: 0 for q in pauli_string_measurement_results.pauli_string.qubits
            }
            assert pauli_string_measurement_results.calibration_result.one_state_errors == {
                q: 0 for q in pauli_string_measurement_results.pauli_string.qubits
            }


@pytest.mark.parametrize("use_sweep", [True, False])
def test_pauli_string_measurement_errors_with_noise(use_sweep: bool) -> None:
    """Test that the mitigated expectation is close to the ideal expectation
    based on the Pauli string"""
    qubits = cirq.LineQubit.range(7)
    circuit = cirq.FrozenCircuit(_create_ghz(7, qubits))
    sampler = NoisySingleQubitReadoutSampler(p0=0.01, p1=0.005, seed=1234)
    simulator = cirq.Simulator()

    circuits_to_pauli: list[CircuitToPauliStringsParameters] = []
    circuits_to_pauli.append(
        CircuitToPauliStringsParameters(
            circuit=circuit,
            pauli_strings=[_generate_random_pauli_string(qubits) for _ in range(3)],
            postselection_symmetries={},
        )
    )

    circuits_with_pauli_expectations = measure_pauli_strings(
        circuits_to_pauli, sampler, 1000, 1000, 1000, np.random.default_rng(), use_sweep
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
                assert 0.0045 < error < 0.0055


@pytest.mark.parametrize("use_sweep", [True, False])
def test_group_pauli_string_measurement_errors_with_noise(use_sweep: bool) -> None:
    """Test that the mitigated expectation is close to the ideal expectation
    based on the group Pauli strings"""
    qubits = cirq.LineQubit.range(7)
    circuit = cirq.FrozenCircuit(_create_ghz(7, qubits))
    sampler = NoisySingleQubitReadoutSampler(p0=0.01, p1=0.005, seed=1234)
    simulator = cirq.Simulator()

    circuits_to_pauli: list[CircuitToPauliStringsParameters] = []
    circuits_to_pauli.append(
        CircuitToPauliStringsParameters(
            circuit=circuit,
            pauli_strings=[
                _generate_qwc_paulis(
                    _generate_random_pauli_string(qubits, enable_coeff=True, allow_pauli_i=False), 5
                )
            ],
            postselection_symmetries={},
        )
    )

    circuits_with_pauli_expectations = measure_pauli_strings(
        circuits_to_pauli, sampler, 1000, 1000, 1000, np.random.default_rng(), use_sweep
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
                assert 0.0045 < error < 0.0055


@pytest.mark.parametrize("use_sweep", [True, False])
def test_many_circuits_input_measurement_with_noise(use_sweep: bool) -> None:
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

    simulator = cirq.Simulator()

    circuits_to_pauli: list[CircuitToPauliStringsParameters] = []
    circuits_to_pauli.append(
        CircuitToPauliStringsParameters(
            circuit=circuit_1,
            pauli_strings=[_generate_random_pauli_string(qubits_1) for _ in range(3)],
            postselection_symmetries={},
        )
    )

    circuits_to_pauli.append(
        CircuitToPauliStringsParameters(
            circuit=circuit_2,
            pauli_strings=[
                _generate_qwc_paulis(cirq.PauliString({q: cirq.X for q in qubits_2}), 5)
            ],
            postselection_symmetries={},
        )
    )

    circuits_to_pauli.append(
        CircuitToPauliStringsParameters(
            circuit=circuit_3,
            pauli_strings=[_generate_random_pauli_string(qubits_3[2:]) for _ in range(3)],
            postselection_symmetries={
                cirq.PauliString({cirq.Z(qubits_3[0]), cirq.Z(qubits_3[1])}): 1
            },
        )
    )

    sampler = NoisySingleQubitReadoutSampler(p0=0.003, p1=0.005, seed=1234)
    simulator = cirq.Simulator()

    circuits_with_pauli_expectations = measure_pauli_strings(
        circuits_to_pauli, sampler, 1000, 1000, 1000, np.random.default_rng(), use_sweep
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
            if isinstance(
                pauli_string_measurement_results.calibration_result,
                SingleQubitReadoutCalibrationResult,
            )
            for (
                error
            ) in pauli_string_measurement_results.calibration_result.zero_state_errors.values():
                assert 0.0025 < error < 0.0035
            for (
                error
            ) in pauli_string_measurement_results.calibration_result.one_state_errors.values():
                assert 0.0045 < error < 0.0055


@pytest.mark.parametrize("use_sweep", [True, False])
def test_allow_group_pauli_measurement_without_readout_mitigation(use_sweep: bool) -> None:
    """Test that the function allows to measure without error mitigation"""
    qubits = cirq.LineQubit.range(7)
    circuit = cirq.FrozenCircuit(_create_ghz(7, qubits))
    sampler = NoisySingleQubitReadoutSampler(p0=0.01, p1=0.005, seed=1234)

    circuits_to_pauli: list[CircuitToPauliStringsParameters] = []
    circuits_to_pauli.append(
        CircuitToPauliStringsParameters(
            circuit=circuit,
            pauli_strings=[
                _generate_qwc_paulis(_generate_random_pauli_string(qubits, True), 2, True),
                _generate_qwc_paulis(_generate_random_pauli_string(qubits), 4),
                _generate_qwc_paulis(_generate_random_pauli_string(qubits), 6),
            ],
            postselection_symmetries={},
        )
    )

    circuits_with_pauli_expectations = measure_pauli_strings(
        circuits_to_pauli, sampler, 1000, 1000, 0, np.random.default_rng(), use_sweep
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

    circuits_to_pauli: list[CircuitToPauliStringsParameters] = []

    circuits_to_pauli.append(
        CircuitToPauliStringsParameters(
            circuit=circuit_1,
            pauli_strings=[_generate_random_pauli_string(qubits_1, True) for _ in range(3)],
            postselection_symmetries={},
        )
    )
    circuits_to_pauli.append(
        CircuitToPauliStringsParameters(
            circuit=circuit_2,
            pauli_strings=[
                _generate_random_pauli_string(
                    [q for q in qubits_2 if (q != qubits_2[1] and q != qubits_2[3])], True
                )
                for _ in range(3)
            ],
            postselection_symmetries={
                cirq.PauliString({cirq.Z(qubits_2[1]), cirq.Z(qubits_2[3])}): 1
            },
        )
    )

    circuits_to_pauli.append(
        CircuitToPauliStringsParameters(
            circuit=circuit_3,
            pauli_strings=[_generate_random_pauli_string(qubits_3, True) for _ in range(3)],
            postselection_symmetries={},
        )
    )

    sampler = NoisySingleQubitReadoutSampler(p0=0.003, p1=0.005, seed=1234)
    simulator = cirq.Simulator()

    circuits_with_pauli_expectations = measure_pauli_strings(
        circuits_to_pauli,
        sampler,
        1000,
        1000,
        1000,
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
            if isinstance(
                pauli_string_measurement_results.calibration_result,
                SingleQubitReadoutCalibrationResult,
            )
            for (
                error
            ) in pauli_string_measurement_results.calibration_result.zero_state_errors.values():
                assert 0.0025 < error < 0.0035
            for (
                error
            ) in pauli_string_measurement_results.calibration_result.one_state_errors.values():
                assert 0.0045 < error < 0.0055


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

    circuits_to_pauli: list[CircuitToPauliStringsParameters] = []
    circuits_to_pauli.append(
        CircuitToPauliStringsParameters(
            circuit=circuit_1,
            pauli_strings=[
                # [cirq.PauliString(cirq.Z(qubits_1[0]), cirq.Z(qubits_1[1]))]
                [cirq.PauliString(cirq.X(qubits_1[0]), cirq.X(qubits_1[1]), cirq.X(qubits_1[2]))]
            ],
            postselection_symmetries={
                # cirq.PauliString(cirq.X(qubits_1[2]), cirq.X(qubits_1[3])): 1
                cirq.PauliString({cirq.X(qubits_1[0]), cirq.X(qubits_1[1]), cirq.X(qubits_1[2])}): 1
            },
        )
    )
    circuits_to_pauli.append(
        CircuitToPauliStringsParameters(
            circuit=circuit_2,
            pauli_strings=[
                _generate_qwc_paulis(
                    _generate_random_pauli_string(qubits_2, enable_coeff=True, allow_pauli_i=False),
                    5,
                )
            ],
            postselection_symmetries={},
        )
    )
    circuits_to_pauli.append(
        CircuitToPauliStringsParameters(
            circuit=circuit_3,
            pauli_strings=[
                _generate_qwc_paulis(
                    _generate_random_pauli_string(qubits_3, enable_coeff=True, allow_pauli_i=False),
                    6,
                )
            ],
            postselection_symmetries={},
        )
    )

    sampler = NoisySingleQubitReadoutSampler(p0=0.003, p1=0.005, seed=1234)
    simulator = cirq.Simulator()

    circuits_with_pauli_expectations = measure_pauli_strings(
        circuits_to_pauli, sampler, 1000, 1000, 1000, np.random.default_rng(), use_sweep
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
            if isinstance(
                pauli_string_measurement_results.calibration_result,
                SingleQubitReadoutCalibrationResult,
            )
            for (
                error
            ) in pauli_string_measurement_results.calibration_result.zero_state_errors.values():
                assert 0.0025 < error < 0.035
            for (
                error
            ) in pauli_string_measurement_results.calibration_result.one_state_errors.values():
                assert 0.0045 < error < 0.0055


def test_coefficient_not_real_number() -> None:
    """Test that the coefficient of input pauli string is not real.
    Should return error in this case"""
    qubits_1 = cirq.LineQubit.range(3)
    random_pauli_string = _generate_random_pauli_string(qubits_1, True) * (3 + 4j)
    circuit_1 = cirq.FrozenCircuit(_create_ghz(3, qubits_1))

    circuits_to_pauli: list[CircuitToPauliStringsParameters] = []
    circuits_to_pauli.append(
        CircuitToPauliStringsParameters(
            circuit=circuit_1,
            pauli_strings=[
                random_pauli_string,
                _generate_random_pauli_string(qubits_1, True),
                _generate_random_pauli_string(qubits_1, True),
            ],
            postselection_symmetries={},
        )
    )

    with pytest.raises(
        ValueError,
        match="Cannot compute expectation value of a "
        "non-Hermitian PauliString. Coefficient must be real.",
    ):
        measure_pauli_strings(
            cirq.Simulator(), circuits_to_pauli, 1000, 1000, 1000, np.random.default_rng()
        )


def test_zero_pauli_repetitions() -> None:
    """Test that the pauli repetitions are zero."""
    qubits = cirq.LineQubit.range(5)

    circuit = cirq.FrozenCircuit(_create_ghz(5, qubits))

    circuits_to_pauli: dict[cirq.FrozenCircuit, list[cirq.PauliString]] = {}
    circuits_to_pauli[circuit] = [cirq.PauliString({q: cirq.X for q in qubits})]
    with pytest.raises(ValueError, match="Must provide positive pauli_repetitions."):
        measure_pauli_strings(
            circuits_to_pauli, cirq.Simulator(), 0, 1000, 1000, np.random.default_rng()
        )


def test_negative_num_random_bitstrings() -> None:
    """Test that the number of random bitstrings is smaller than zero."""
    qubits = cirq.LineQubit.range(5)

    circuit = cirq.FrozenCircuit(_create_ghz(5, qubits))

    circuits_to_pauli: dict[cirq.FrozenCircuit, list[cirq.PauliString]] = {}
    circuits_to_pauli[circuit] = [cirq.PauliString({q: cirq.X for q in qubits})]
    with pytest.raises(ValueError, match="Must provide zero or more num_random_bitstrings."):
        measure_pauli_strings(
            circuits_to_pauli, cirq.Simulator(), 1000, 1000, -1, np.random.default_rng()
        )


def test_zero_readout_repetitions() -> None:
    """Test that the readout repetitions is zero."""
    qubits = cirq.LineQubit.range(5)

    circuit = cirq.FrozenCircuit(_create_ghz(5, qubits))

    circuits_to_pauli: dict[cirq.FrozenCircuit, list[cirq.PauliString]] = {}
    circuits_to_pauli[circuit] = [cirq.PauliString({q: cirq.X for q in qubits})]
    with pytest.raises(
        ValueError, match="Must provide positive readout_repetitions for readout" + " calibration."
    ):
        measure_pauli_strings(
            circuits_to_pauli, cirq.Simulator(), 1000, 0, 1000, np.random.default_rng()
        )


def test_rng_type_mismatch() -> None:
    """Test that the rng is not a numpy random generator or a seed."""
    qubits = cirq.LineQubit.range(5)

    circuit = cirq.FrozenCircuit(_create_ghz(5, qubits))

    circuits_to_pauli: dict[cirq.FrozenCircuit, list[cirq.PauliString]] = {}
    circuits_to_pauli[circuit] = [cirq.PauliString({q: cirq.X for q in qubits})]
    with pytest.raises(ValueError, match="Must provide a numpy random generator or a seed"):
        measure_pauli_strings(
            circuits_to_pauli, cirq.Simulator(), 1000, 1000, 1000, "test"  # type: ignore[arg-type]
        )


def test_pauli_type_mismatch() -> None:
    """Test that the input paulis are not a sequence of PauliStrings."""
    qubits = cirq.LineQubit.range(5)

    circuit = cirq.FrozenCircuit(_create_ghz(5, qubits))

    circuits_to_pauli: dict[cirq.FrozenCircuit, int] = {}
    circuits_to_pauli[circuit] = 1
    with pytest.raises(
        TypeError,
        match="Expected all elements to be either a sequence of PauliStrings or sequences of"
        " ops.PauliStrings. Got <class 'int'> instead.",
    ):
        measure_pauli_strings(
            circuits_to_pauli, cirq.Simulator(), 1000, 1000, 1000, 1  # type: ignore[arg-type]
        )


def test_group_paulis_are_not_qwc() -> None:
    """Test that the group paulis are not qwc."""
    qubits = cirq.LineQubit.range(5)

    circuit = cirq.FrozenCircuit(_create_ghz(5, qubits))

    pauli_str1: cirq.PauliString = cirq.PauliString({qubits[0]: cirq.X, qubits[1]: cirq.Y})
    pauli_str2: cirq.PauliString = cirq.PauliString({qubits[0]: cirq.Y})

    circuits_to_pauli: dict[cirq.FrozenCircuit, list[cirq.PauliString]] = {}
    circuits_to_pauli[circuit] = [[pauli_str1, pauli_str2]]  # type: ignore
    with pytest.raises(
        ValueError, match="The group of Pauli strings are not Qubit-Wise Commuting with each other."
    ):
        measure_pauli_strings(
            circuits_to_pauli, cirq.Simulator(), 1000, 1000, 1000, np.random.default_rng()
        )


def test_empty_group_paulis_not_allowed() -> None:
    """Test that the group paulis are empty"""
    qubits = cirq.LineQubit.range(5)

    circuit = cirq.FrozenCircuit(_create_ghz(5, qubits))

    circuits_to_pauli: dict[cirq.FrozenCircuit, list[cirq.PauliString]] = {}
    circuits_to_pauli[circuit] = [[]]  # type: ignore
    with pytest.raises(ValueError, match="Empty group of Pauli strings is not allowed"):
        measure_pauli_strings(
            circuits_to_pauli, cirq.Simulator(), 1000, 1000, 1000, np.random.default_rng()
        )


def test_group_paulis_type_mismatch() -> None:
    """Test that the group paulis type is not correct"""
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

    circuits_to_pauli: dict[cirq.FrozenCircuit, list[list[cirq.PauliString]]] = {}
    circuits_to_pauli[circuit_1] = [
        _generate_qwc_paulis(
            _generate_random_pauli_string(qubits_1, enable_coeff=True, allow_pauli_i=False), 6
        )
        for _ in range(3)
    ]
    circuits_to_pauli[circuit_2] = [_generate_random_pauli_string(qubits_2, True) for _ in range(3)]
    circuits_to_pauli[circuit_3] = [_generate_random_pauli_string(qubits_3, True) for _ in range(3)]

    with pytest.raises(
        TypeError,
        match="Expected all elements to be sequences of ops.PauliString, "
        "but found <class 'cirq.ops.pauli_string.PauliString'>.",
    ):
        measure_pauli_strings(
            circuits_to_pauli, cirq.Simulator(), 1000, 1000, 1000, np.random.default_rng()
        )
