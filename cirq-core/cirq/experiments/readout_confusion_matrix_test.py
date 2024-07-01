# Copyright 2022 The Cirq Developers
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

import numpy as np
import cirq
import pytest

from cirq.experiments.single_qubit_readout_calibration_test import NoisySingleQubitReadoutSampler
from cirq.experiments.readout_confusion_matrix import TensoredConfusionMatrices


def add_readout_error(
    measurements: np.ndarray,
    zero_errors: np.ndarray,
    one_errors: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add readout errors to measured (or simulated) bitstrings.

    Args:
        measurements: The bitstrings to which we will add readout errors. measurements[i,j] is the
                      ith bitstring, qubit j.
        zero_errors: zero_errors[i] is the probability of a 0->1 readout error on qubit i.
        one_errors: one_errors[i] is the probability of a 1->0 readout error on qubit i.
        rng: The pseudorandom number generator to use.

    Returns:
        New measurements but with readout errors added.
    """
    num_bitstrs, n = measurements.shape
    assert len(zero_errors) == len(one_errors) == n
    # compute the probability that each bit is 1 after adding readout errors:
    p1 = measurements * (1 - one_errors) + (1 - measurements) * zero_errors
    r = rng.random((num_bitstrs, n))
    noisy_measurements = r < p1
    return noisy_measurements.astype(int)


def get_expected_cm(num_qubits: int, p0: float, p1: float):
    expected_cm = np.zeros((2**num_qubits,) * 2)
    for i in range(2**num_qubits):
        for j in range(2**num_qubits):
            p = 1.0
            for k in range(num_qubits):
                b0 = (i >> k) & 1
                b1 = (j >> k) & 1
                if b0 == 0:
                    p *= p0 * b1 + (1 - p0) * (1 - b1)
                else:
                    p *= p1 * (1 - b1) + (1 - p1) * b1
            expected_cm[i][j] = p
    return expected_cm


@pytest.mark.parametrize('p0, p1', [(0, 0), (0.2, 0.4), (0.5, 0.5), (0.6, 0.3), (1.0, 1.0)])
def test_measure_confusion_matrix_with_noise(p0, p1):
    sampler = NoisySingleQubitReadoutSampler(p0, p1, seed=1234)
    num_qubits = 4
    qubits = cirq.LineQubit.range(num_qubits)
    expected_cm = get_expected_cm(num_qubits, p0, p1)
    qubits_small = qubits[:2]
    expected_cm_small = get_expected_cm(2, p0, p1)
    repetitions = 12_000
    # Build entire confusion matrix by running 2 ** 4 = 16 circuits.
    readout_cm = cirq.measure_confusion_matrix(sampler, qubits, repetitions=repetitions)
    assert readout_cm.repetitions == repetitions
    for q, expected in zip([None, qubits_small], [expected_cm, expected_cm_small]):
        np.testing.assert_allclose(readout_cm.confusion_matrix(q), expected, atol=1e-2)
        np.testing.assert_allclose(
            readout_cm.confusion_matrix(q) @ readout_cm.correction_matrix(q),
            np.eye(expected.shape[0]),
            atol=1e-2,
        )

    # Build a tensored confusion matrix using smaller single qubit confusion matrices.
    # This works because the error is uncorrelated and requires only 4 * 2 = 8 circuits.
    readout_cm = cirq.measure_confusion_matrix(
        sampler, [[q] for q in qubits], repetitions=repetitions
    )
    assert readout_cm.repetitions == repetitions
    for q, expected in zip([None, qubits_small], [expected_cm, expected_cm_small]):
        np.testing.assert_allclose(readout_cm.confusion_matrix(q), expected, atol=1e-2)
        np.testing.assert_allclose(
            readout_cm.confusion_matrix(q) @ readout_cm.correction_matrix(q),
            np.eye(expected.shape[0]),
            atol=1e-2,
        )

    # Apply corrections to sampled probabilities using readout_cm.
    qs = qubits_small
    circuit = cirq.Circuit(cirq.H.on_each(*qs), cirq.measure(*qs))
    reps = 100_000
    sampled_result = cirq.get_state_histogram(sampler.run(circuit, repetitions=reps)) / reps
    expected_result = [1 / 4] * 4

    def l2norm(result: np.ndarray):
        return np.sum((expected_result - result) ** 2)

    corrected_result = readout_cm.apply(sampled_result, qs)
    assert l2norm(corrected_result) <= l2norm(sampled_result)


def test_from_measurement():
    qubits = cirq.LineQubit.range(3)
    confuse_02 = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]])
    confuse_1 = np.array([[0, 1], [1, 0]])
    op = cirq.measure(
        *qubits,
        key='a',
        invert_mask=(True, False),
        confusion_map={(0, 2): confuse_02, (1,): confuse_1},
    )
    tcm = cirq.TensoredConfusionMatrices.from_measurement(op.gate, op.qubits)
    expected_tcm = cirq.TensoredConfusionMatrices(
        [confuse_02, confuse_1], ((qubits[0], qubits[2]), (qubits[1],)), repetitions=0, timestamp=0
    )
    assert tcm == expected_tcm

    no_cm_op = cirq.measure(*qubits, key='a')
    with pytest.raises(ValueError, match="Measurement has no confusion matrices"):
        _ = cirq.TensoredConfusionMatrices.from_measurement(no_cm_op.gate, no_cm_op.qubits)


def test_readout_confusion_matrix_raises():
    num_qubits = 2
    confusion_matrix = get_expected_cm(num_qubits, 0.1, 0.2)
    qubits = cirq.LineQubit.range(4)
    with pytest.raises(ValueError, match=r"measure_qubits cannot be empty"):
        _ = cirq.TensoredConfusionMatrices([], [], repetitions=0, timestamp=0)

    with pytest.raises(ValueError, match=r"len\(confusion_matrices\)"):
        _ = cirq.TensoredConfusionMatrices(
            [confusion_matrix], [qubits[:2], qubits[2:]], repetitions=0, timestamp=0
        )

    with pytest.raises(ValueError, match="Shape mismatch for confusion matrix"):
        _ = cirq.TensoredConfusionMatrices(confusion_matrix, qubits, repetitions=0, timestamp=0)

    with pytest.raises(ValueError, match="Repeated qubits not allowed"):
        _ = cirq.TensoredConfusionMatrices(
            [confusion_matrix, confusion_matrix],
            [qubits[:2], qubits[1:3]],
            repetitions=0,
            timestamp=0,
        )

    readout_cm = cirq.TensoredConfusionMatrices(
        [confusion_matrix, confusion_matrix], [qubits[:2], qubits[2:]], repetitions=0, timestamp=0
    )

    with pytest.raises(ValueError, match="should be a subset of"):
        _ = readout_cm.confusion_matrix([cirq.NamedQubit("a")])

    with pytest.raises(ValueError, match="should be a subset of"):
        _ = readout_cm.correction_matrix([cirq.NamedQubit("a")])

    with pytest.raises(ValueError, match="result.shape .* should be"):
        _ = readout_cm.apply(np.asarray([100]), qubits[:2])

    with pytest.raises(ValueError, match="method.* should be"):
        _ = readout_cm.apply(np.asarray([1 / 16] * 16), method='l1norm')


def test_readout_confusion_matrix_repr_and_equality():
    mat1 = cirq.testing.random_orthogonal(4, random_state=1234)
    mat2 = cirq.testing.random_orthogonal(2, random_state=1234)
    q = cirq.LineQubit.range(3)
    a = cirq.TensoredConfusionMatrices([mat1, mat2], [q[:2], q[2:]], repetitions=0, timestamp=0)
    b = cirq.TensoredConfusionMatrices(mat1, q[:2], repetitions=0, timestamp=0)
    c = cirq.TensoredConfusionMatrices(mat2, q[2:], repetitions=0, timestamp=0)
    for x in [a, b, c]:
        cirq.testing.assert_equivalent_repr(x)
        assert cirq.approx_eq(x, x)
        assert x._approx_eq_(mat1, 1e-6) is NotImplemented
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(a, a)
    eq.add_equality_group(b, b)
    eq.add_equality_group(c, c)


def _sample_ghz(n: int, repetitions: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a GHZ state in the z basis.
    Args:
        n: The number of qubits.
        repetitions: The number of repetitions.
        rng: The pseudorandom number generator to use.
    Returns:
        An array of the measurement outcomes.
    """
    return np.tile(rng.integers(0, 2, size=repetitions), (n, 1)).T


def _add_noise_and_mitigate_ghz(
    n: int,
    repetitions: int,
    zero_errors: np.ndarray,
    one_errors: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float, float]:
    """Add readout error to GHZ-like bitstrings and measure <ZZZ...> with and
    without readout error mitigation.
    Args:
        n: The number of qubits.
        repetitions: The number of repetitions.
        zero_errors: zero_errors[i] is the probability of a 0->1 readout error on qubit i.
        one_errors: one_errors[i] is the probability of a 1->0 readout error on qubit i.
        rng: The pseudorandom number generator to use.
    Returns:
        A tuple of:
        - The mitigated expectation value of <ZZZ...>
        - The statistical uncertainty of the previous output
        - The unmitigated expectation value of <ZZZ...>
        - The statstical uncertainty of the previous output
    """
    if rng is None:
        rng = np.random.default_rng(0)
    confusion_matrices = [
        np.array([[1 - e0, e1], [e0, 1 - e1]]) for e0, e1 in zip(zero_errors, one_errors)
    ]
    qubits = cirq.LineQubit.range(n)
    tcm = TensoredConfusionMatrices(
        confusion_matrices, [[q] for q in qubits], repetitions=0, timestamp=0.0
    )

    measurements = _sample_ghz(n, repetitions, rng)
    noisy_measurements = add_readout_error(measurements, zero_errors, one_errors, rng)
    # unmitigated:
    p1 = np.mean(np.sum(noisy_measurements, axis=1) % 2)
    z = 1 - 2 * np.mean(p1)
    dz = 2 * np.sqrt(p1 * (1 - p1) / repetitions)
    # return mitigated and unmitigated:
    return (*tcm.readout_mitigation_pauli_uncorrelated(qubits, noisy_measurements), z, dz)


def test_uncorrelated_readout_mitigation_pauli():
    n_all = np.arange(2, 35)
    z_all_mit = []
    dz_all_mit = []
    z_all_raw = []
    dz_all_raw = []
    repetitions = 10_000
    for n in n_all:
        e0 = np.ones(n) * 0.005
        e1 = np.ones(n) * 0.03
        z_mit, dz_mit, z_raw, dz_raw = _add_noise_and_mitigate_ghz(n, repetitions, e0, e1)
        z_all_mit.append(z_mit)
        dz_all_mit.append(dz_mit)
        z_all_raw.append(z_raw)
        dz_all_raw.append(dz_raw)

    for n, z, dz in zip(n_all, z_all_mit, dz_all_mit):
        ideal = 1.0 if n % 2 == 0 else 0.0
        assert np.isclose(z, ideal, atol=4 * dz)
