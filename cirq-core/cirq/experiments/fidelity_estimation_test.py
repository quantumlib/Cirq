# Copyright 2019 The Cirq Developers
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
import itertools
import math
from typing import Sequence

import numpy as np
import pytest

import cirq


def sample_noisy_bitstrings(
    circuit: cirq.Circuit, qubit_order: Sequence[cirq.Qid], depolarization: float, repetitions: int
) -> np.ndarray:
    assert 0 <= depolarization <= 1
    dim = np.product(circuit.qid_shape())
    n_incoherent = int(depolarization * repetitions)
    n_coherent = repetitions - n_incoherent
    incoherent_samples = np.random.randint(dim, size=n_incoherent)
    circuit_with_measurements = cirq.Circuit(circuit, cirq.measure(*qubit_order, key='m'))
    r = cirq.sample(circuit_with_measurements, repetitions=n_coherent)
    coherent_samples = r.data['m'].to_numpy()
    return np.concatenate((coherent_samples, incoherent_samples))


def make_random_quantum_circuit(qubits: Sequence[cirq.Qid], depth: int) -> cirq.Circuit:
    SQ_GATES = [cirq.X ** 0.5, cirq.Y ** 0.5, cirq.T]
    circuit = cirq.Circuit()
    cz_start = 0
    for q in qubits:
        circuit.append(cirq.H(q))
    for _ in range(depth):
        for q in qubits:
            random_gate = SQ_GATES[np.random.randint(len(SQ_GATES))]
            circuit.append(random_gate(q))
        for q0, q1 in zip(
            itertools.islice(qubits, cz_start, None, 2),
            itertools.islice(qubits, cz_start + 1, None, 2),
        ):
            circuit.append(cirq.CNOT(q0, q1))
        cz_start = 1 - cz_start
    for q in qubits:
        circuit.append(cirq.H(q))
    return circuit


@pytest.mark.parametrize(
    'depolarization, estimator',
    itertools.product(
        (0.0, 0.2, 0.7, 1.0),
        (
            cirq.hog_score_xeb_fidelity_from_probabilities,
            cirq.linear_xeb_fidelity_from_probabilities,
            cirq.log_xeb_fidelity_from_probabilities,
        ),
    ),
)
def test_xeb_fidelity(depolarization, estimator):
    prng_state = np.random.get_state()
    np.random.seed(0)

    fs = []
    for _ in range(10):
        qubits = cirq.LineQubit.range(5)
        circuit = make_random_quantum_circuit(qubits, depth=12)
        bitstrings = sample_noisy_bitstrings(circuit, qubits, depolarization, repetitions=5000)

        f = cirq.xeb_fidelity(circuit, bitstrings, qubits, estimator=estimator)
        amplitudes = cirq.final_state_vector(circuit)
        f2 = cirq.xeb_fidelity(
            circuit, bitstrings, qubits, amplitudes=amplitudes, estimator=estimator
        )
        assert np.abs(f - f2) < 1e-6

        fs.append(f)

    estimated_fidelity = np.mean(fs)
    expected_fidelity = 1 - depolarization
    assert np.isclose(estimated_fidelity, expected_fidelity, atol=0.04)

    np.random.set_state(prng_state)


def test_linear_and_log_xeb_fidelity():
    prng_state = np.random.get_state()
    np.random.seed(0)

    depolarization = 0.5

    fs_log = []
    fs_lin = []
    for _ in range(10):
        qubits = cirq.LineQubit.range(5)
        circuit = make_random_quantum_circuit(qubits, depth=12)
        bitstrings = sample_noisy_bitstrings(
            circuit, qubits, depolarization=depolarization, repetitions=5000
        )

        f_log = cirq.log_xeb_fidelity(circuit, bitstrings, qubits)
        f_lin = cirq.linear_xeb_fidelity(circuit, bitstrings, qubits)

        fs_log.append(f_log)
        fs_lin.append(f_lin)

    assert np.isclose(np.mean(fs_log), 1 - depolarization, atol=0.01)
    assert np.isclose(np.mean(fs_lin), 1 - depolarization, atol=0.09)

    np.random.set_state(prng_state)


def test_xeb_fidelity_invalid_qubits():
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))
    bitstrings = sample_noisy_bitstrings(circuit, (q0, q1, q2), 0.9, 10)
    with pytest.raises(ValueError):
        cirq.xeb_fidelity(circuit, bitstrings, (q0, q2))


def test_xeb_fidelity_invalid_bitstrings():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))
    bitstrings = [0, 1, 2, 3, 4]
    with pytest.raises(ValueError):
        cirq.xeb_fidelity(circuit, bitstrings, (q0, q1))


def test_xeb_fidelity_tuple_input():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))
    bitstrings = [0, 1, 2]
    f1 = cirq.xeb_fidelity(circuit, bitstrings, (q0, q1))
    f2 = cirq.xeb_fidelity(circuit, tuple(bitstrings), (q0, q1))
    assert f1 == f2


def test_least_squares_xeb_fidelity_from_expectations():
    prng_state = np.random.get_state()
    np.random.seed(0)

    depolarization = 0.5

    n_qubits = 5
    dim = 2 ** n_qubits
    n_circuits = 10
    qubits = cirq.LineQubit.range(n_qubits)

    measured_expectations_lin = []
    exact_expectations_lin = []
    measured_expectations_log = []
    exact_expectations_log = []
    uniform_expectations_log = []
    for _ in range(n_circuits):
        circuit = make_random_quantum_circuit(qubits, depth=12)
        bitstrings = sample_noisy_bitstrings(
            circuit, qubits, depolarization=depolarization, repetitions=5000
        )
        amplitudes = cirq.final_state_vector(circuit)
        probabilities = np.abs(amplitudes) ** 2

        measured_expectations_lin.append(dim * np.mean(probabilities[bitstrings]))
        exact_expectations_lin.append(dim * np.sum(probabilities ** 2))

        measured_expectations_log.append(np.mean(np.log(dim * probabilities[bitstrings])))
        exact_expectations_log.append(np.sum(probabilities * np.log(dim * probabilities)))
        uniform_expectations_log.append(np.mean(np.log(dim * probabilities)))

    f_lin, r_lin = cirq.experiments.least_squares_xeb_fidelity_from_expectations(
        measured_expectations_lin, exact_expectations_lin, [1.0] * n_circuits
    )
    f_log, r_log = cirq.experiments.least_squares_xeb_fidelity_from_expectations(
        measured_expectations_log, exact_expectations_log, uniform_expectations_log
    )

    assert np.isclose(f_lin, 1 - depolarization, atol=0.01)
    assert np.isclose(f_log, 1 - depolarization, atol=0.01)
    np.testing.assert_allclose(np.sum(np.array(r_lin) ** 2), 0.0, atol=1e-2)
    np.testing.assert_allclose(np.sum(np.array(r_log) ** 2), 0.0, atol=1e-2)

    np.random.set_state(prng_state)


def test_least_squares_xeb_fidelity_from_expectations_bad_length():
    with pytest.raises(ValueError) as exception_info:
        _ = cirq.experiments.least_squares_xeb_fidelity_from_expectations([1.0], [1.0], [1.0, 2.0])
    assert '1, 1, and 2' in str(exception_info.value)


def test_least_squares_xeb_fidelity_from_probabilities():
    prng_state = np.random.get_state()
    np.random.seed(0)

    depolarization = 0.5

    n_qubits = 5
    dim = 2 ** n_qubits
    n_circuits = 10
    qubits = cirq.LineQubit.range(n_qubits)

    all_probabilities = []
    observed_probabilities = []
    for _ in range(n_circuits):
        circuit = make_random_quantum_circuit(qubits, depth=12)
        bitstrings = sample_noisy_bitstrings(
            circuit, qubits, depolarization=depolarization, repetitions=5000
        )
        amplitudes = cirq.final_state_vector(circuit)
        probabilities = np.abs(amplitudes) ** 2

        all_probabilities.append(probabilities)
        observed_probabilities.append(probabilities[bitstrings])

    f_lin, r_lin = cirq.least_squares_xeb_fidelity_from_probabilities(
        dim, observed_probabilities, all_probabilities, None, True
    )
    f_log_np, r_log_np = cirq.least_squares_xeb_fidelity_from_probabilities(
        dim, observed_probabilities, all_probabilities, np.log, True
    )
    f_log_math, r_log_math = cirq.least_squares_xeb_fidelity_from_probabilities(
        dim, observed_probabilities, all_probabilities, math.log, False
    )

    assert np.isclose(f_lin, 1 - depolarization, atol=0.01)
    assert np.isclose(f_log_np, 1 - depolarization, atol=0.01)
    assert np.isclose(f_log_math, 1 - depolarization, atol=0.01)
    np.testing.assert_allclose(np.sum(np.array(r_lin) ** 2), 0.0, atol=1e-2)
    np.testing.assert_allclose(np.sum(np.array(r_log_np) ** 2), 0.0, atol=1e-2)
    np.testing.assert_allclose(np.sum(np.array(r_log_math) ** 2), 0.0, atol=1e-2)

    np.random.set_state(prng_state)
