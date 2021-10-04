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

"""Tests for superoperator_protocol.py."""

from typing import Sequence, Tuple

import numpy as np
import pytest

import cirq

LOCAL_DEFAULT = np.array([])


def test_superoperator_no_methods():
    class NoMethod:
        pass

    with pytest.raises(TypeError, match="no _superoperator_ method"):
        _ = cirq.superoperator(NoMethod())

    assert cirq.superoperator(NoMethod(), None) is None
    assert cirq.superoperator(NoMethod(), NotImplemented) is NotImplemented
    assert cirq.superoperator(NoMethod(), LOCAL_DEFAULT) is LOCAL_DEFAULT
    assert np.all(cirq.superoperator(NoMethod(), np.eye(4)) == np.eye(4))

    assert not cirq.has_superoperator(NoMethod())


def test_superoperator_returns_not_implemented():
    class ReturnsNotImplemented:
        def _superoperator_(self):
            return NotImplemented

    with pytest.raises(TypeError, match="returned NotImplemented"):
        _ = cirq.superoperator(ReturnsNotImplemented())

    assert cirq.superoperator(ReturnsNotImplemented(), None) is None
    assert cirq.superoperator(ReturnsNotImplemented(), NotImplemented) is NotImplemented
    assert cirq.superoperator(ReturnsNotImplemented(), LOCAL_DEFAULT) is LOCAL_DEFAULT
    assert np.all(cirq.superoperator(ReturnsNotImplemented(), np.eye(4)) == np.eye(4))

    assert not cirq.has_superoperator(ReturnsNotImplemented())


def test_kraus_returns_not_implemented():
    class ReturnsNotImplemented:
        def _kraus_(self):
            return NotImplemented

    with pytest.raises(TypeError, match="no Kraus representation"):
        _ = cirq.superoperator(ReturnsNotImplemented())

    assert cirq.superoperator(ReturnsNotImplemented(), None) is None
    assert cirq.superoperator(ReturnsNotImplemented(), NotImplemented) is NotImplemented
    assert cirq.superoperator(ReturnsNotImplemented(), LOCAL_DEFAULT) is LOCAL_DEFAULT
    assert np.all(cirq.superoperator(ReturnsNotImplemented(), np.eye(4)) == np.eye(4))

    assert not cirq.has_superoperator(ReturnsNotImplemented())


def test_explicit_superoperator():
    s = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])

    class ReturnsSuperoperator:
        def _superoperator_(self) -> np.ndarray:
            return s

    assert cirq.superoperator(ReturnsSuperoperator()) is s
    assert cirq.superoperator(ReturnsSuperoperator(), None) is s
    assert cirq.superoperator(ReturnsSuperoperator(), NotImplemented) is s
    assert cirq.superoperator(ReturnsSuperoperator(), np.eye(4)) is s
    assert cirq.superoperator(ReturnsSuperoperator(), LOCAL_DEFAULT) is s

    assert cirq.has_superoperator(ReturnsSuperoperator())


def test_superoperator_fallback_to_kraus():
    g = 0.1
    k = (np.diag([1, np.sqrt(1 - g)]), np.array([[0, np.sqrt(g)], [0, 0]]))

    class ReturnsKraus:
        def _kraus_(self) -> Sequence[np.ndarray]:
            return k

    s = np.array(
        [[1, 0, 0, g], [0, np.sqrt(1 - g), 0, 0], [0, 0, np.sqrt(1 - g), 0], [0, 0, 0, 1 - g]]
    )

    assert np.allclose(cirq.superoperator(ReturnsKraus()), s)
    assert np.allclose(cirq.superoperator(ReturnsKraus(), None), s)
    assert np.allclose(cirq.superoperator(ReturnsKraus(), NotImplemented), s)
    assert np.allclose(cirq.superoperator(ReturnsKraus(), (1,)), s)
    assert np.allclose(cirq.superoperator(ReturnsKraus(), LOCAL_DEFAULT), s)

    assert cirq.has_superoperator(ReturnsKraus())


def test_superoperator_fallback_to_mixture():
    x, y, z = [cirq.unitary(g) for g in (cirq.X, cirq.Y, cirq.Z)]
    m = ((0.2, x), (0.3, y), (0.5, z))

    class ReturnsMixture:
        def _mixture_(self) -> Sequence[Tuple[float, np.ndarray]]:
            return m

    s = 0.2 * np.kron(x, x) + 0.3 * np.kron(y, y.conj()) + 0.5 * np.kron(z, z)

    assert np.allclose(cirq.superoperator(ReturnsMixture()), s)
    assert np.allclose(cirq.superoperator(ReturnsMixture(), None), s)
    assert np.allclose(cirq.superoperator(ReturnsMixture(), NotImplemented), s)
    assert np.allclose(cirq.superoperator(ReturnsMixture(), (1,)), s)
    assert np.allclose(cirq.superoperator(ReturnsMixture(), LOCAL_DEFAULT), s)

    assert cirq.has_superoperator(ReturnsMixture())


def test_superoperator_fallback_to_unitary():
    u = cirq.unitary(cirq.Y)

    class ReturnsUnitary:
        def _unitary_(self) -> np.ndarray:
            return u

    s = np.kron(u, u.conj())

    assert np.allclose(cirq.superoperator(ReturnsUnitary()), s)
    assert np.allclose(cirq.superoperator(ReturnsUnitary(), None), s)
    assert np.allclose(cirq.superoperator(ReturnsUnitary(), NotImplemented), s)
    assert np.allclose(cirq.superoperator(ReturnsUnitary(), (1,)), s)
    assert np.allclose(cirq.superoperator(ReturnsUnitary(), LOCAL_DEFAULT), s)

    assert cirq.has_superoperator(ReturnsUnitary())


class HasSuperoperator:
    def _has_superoperator_(self) -> bool:
        return True


class HasKraus(cirq.SingleQubitGate):
    def _has_kraus_(self) -> bool:
        return True


class HasMixture(cirq.SingleQubitGate):
    def _has_mixture_(self) -> bool:
        return True


class HasUnitary(cirq.SingleQubitGate):
    def _has_unitary_(self) -> bool:
        return True


class HasKrausWhenDecomposed(cirq.SingleQubitGate):
    def __init__(self, decomposed_cls):
        self.decomposed_cls = decomposed_cls

    def _decompose_(self, qubits):
        return [self.decomposed_cls().on(q) for q in qubits]


@pytest.mark.parametrize('cls', [HasSuperoperator, HasKraus, HasMixture, HasUnitary])
def test_has_superoperator(cls):
    assert cirq.has_superoperator(cls())


@pytest.mark.parametrize('decomposed_cls', [HasKraus, HasMixture, HasUnitary])
def test_has_superoperator_when_decomposed(decomposed_cls):
    op = HasKrausWhenDecomposed(decomposed_cls).on(cirq.NamedQubit('test'))
    assert cirq.has_superoperator(op)
    assert not cirq.has_superoperator(op, allow_decompose=False)


def apply_superoperator(superoperator: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Computes output density matrix from superoperator and input density matrix."""
    vectorized_input = np.reshape(rho, np.prod(rho.shape))
    vectorized_output = superoperator @ vectorized_input
    return np.reshape(vectorized_output, rho.shape)


@pytest.mark.parametrize('op', (cirq.I, cirq.X, cirq.Y, cirq.Z, cirq.H, cirq.amplitude_damp(0.1)))
@pytest.mark.parametrize(
    'rho',
    (
        np.diag([1, 0]),
        np.diag([0, 1]),
        np.array([[0.5, 0.5], [0.5, 0.5]]),
        np.array([[0.5, -0.5], [-0.5, 0.5]]),
        np.array([[0.5, -0.5j], [0.5j, 0.5]]),
    ),
)
def test_compare_superoperator_action_to_simulation_one_qubit(op, rho):
    superoperator = cirq.superoperator(op)
    actual_state = apply_superoperator(superoperator, rho)

    sim = cirq.DensityMatrixSimulator()
    q = cirq.NamedQubit('q')
    circuit = cirq.Circuit(op.on(q))
    initial_state = np.array(rho, dtype=np.complex64)
    expected_state = sim.simulate(circuit, initial_state=initial_state).final_density_matrix

    assert np.allclose(actual_state, expected_state)


@pytest.mark.parametrize('op', (cirq.CNOT, cirq.ISWAP, cirq.depolarize(0.2, n_qubits=2)))
@pytest.mark.parametrize(
    'rho',
    (
        np.diag([1, 0, 0, 0]),
        np.diag([0, 1, 0, 0]),
        np.diag([0, 0, 1, 0]),
        np.diag([0, 0, 0, 1]),
        np.array([[0.5, 0.5, 0, 0], [0.5, 0.5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        np.array([[0.5, -0.5j, 0, 0], [0.5j, 0.5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        np.array([[0.5, 0, 0.5j, 0], [0, 0, 0, 0], [-0.5j, 0, 0.5, 0], [0, 0, 0, 0]]),
    ),
)
def test_compare_superoperator_action_to_simulation_two_qubits(op, rho):
    superoperator = cirq.superoperator(op)
    actual_state = apply_superoperator(superoperator, rho)

    sim = cirq.DensityMatrixSimulator()
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(op.on(q0, q1))
    initial_state = np.array(rho, dtype=np.complex64)
    expected_state = sim.simulate(circuit, initial_state=initial_state).final_density_matrix

    assert np.allclose(actual_state, expected_state)
