# Copyright 2018 The Cirq Developers
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
import pytest

import cirq


def test_unitary():
    m = np.array([[0, 1], [1, 0]])
    m2 = np.array([[0, 0, 0, 1],
                   [0, 0, 1, 0],
                   [0, 1, 0, 0],
                   [1, 0, 0, 0]])  # corresponds to X on two qubits
    m3 = [[0, 0, 0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 1],
          [0, 0, 0, 0, 0, 0, 1, 0],
          [0, 1, 0, 0, 0, 0, 0, 0],
          [1, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 1, 0, 0, 0, 0, 0]]  # corresponds to X on qubit 1 and 3
    d = np.array([])

    class NoMethod:
        pass

    class ReturnsNotImplemented(cirq.Gate):
        def _has_unitary_(self):
            return NotImplemented
        def _unitary_(self):
            return NotImplemented
        def num_qubits(self):
            return None # coverage: ignore

    class ReturnsMatrix:
        def _unitary_(self) -> np.ndarray:
            return m
        def num_qubits(self):
            return 1 # coverage: ignore

    class FullyImplemented(cirq.Gate):
        def __init__(self, unitary_value):
            self.unitary_value = unitary_value
        def _has_unitary_(self) -> bool:
            return self.unitary_value
        def _unitary_(self) -> np.ndarray:
            return m
        def num_qubits(self):
            return 1 # coverage: ignore

    class Decomposable(cirq.Operation):
        qubits = ()
        with_qubits = NotImplemented
        def __init__(self, qubits, unitary_value):
            self.qubits = qubits
            self.unitary_value = unitary_value
        def _decompose_(self):
            for q in self.qubits:
                yield FullyImplemented(self.unitary_value)(q)

    class DecomposableOrder(cirq.Operation):
        qubits = ()
        with_qubits = NotImplemented
        def __init__(self, qubits):
            self.qubits = qubits
        def _decompose_(self):
            yield FullyImplemented(True)(self.qubits[2])
            yield FullyImplemented(True)(self.qubits[0])

    class DecomposableNoUnitary(cirq.Operation):
        qubits = ()
        with_qubits = NotImplemented
        def __init__(self, qubits):
            self.qubits = qubits
        def _decompose_(self):
            for q in self.qubits:
                yield ReturnsNotImplemented()(q)

    class DummyOperation(cirq.Operation):
        qubits = ()
        with_qubits = NotImplemented
        def __init__(self, qubits):
            self.qubits = qubits
        def _decompose_(self):
            return ()

    with pytest.raises(TypeError, match='no _unitary_ method'):
        _ = cirq.unitary(NoMethod())
    with pytest.raises(TypeError, match='returned NotImplemented'):
        _ = cirq.unitary(ReturnsNotImplemented())
    assert cirq.unitary(ReturnsMatrix()) is m

    assert cirq.unitary(NoMethod(), None) is None
    assert cirq.unitary(ReturnsNotImplemented(), None) is None
    assert cirq.unitary(ReturnsMatrix(), None) is m

    assert cirq.unitary(NoMethod(), NotImplemented) is NotImplemented
    assert cirq.unitary(ReturnsNotImplemented(),
                        NotImplemented) is NotImplemented
    assert cirq.unitary(ReturnsMatrix(), NotImplemented) is m

    assert cirq.unitary(NoMethod(), 1) == 1
    assert cirq.unitary(ReturnsNotImplemented(), 1) == 1
    assert cirq.unitary(ReturnsMatrix(), 1) is m

    assert cirq.unitary(NoMethod(), d) is d
    assert cirq.unitary(ReturnsNotImplemented(), d) is d
    assert cirq.unitary(ReturnsMatrix(), d) is m
    assert cirq.unitary(FullyImplemented(True), d) is m

    # Test _has_unitary_
    assert not cirq.has_unitary(NoMethod())
    assert not cirq.has_unitary(ReturnsNotImplemented())
    assert cirq.has_unitary(ReturnsMatrix())
    # Explicit function should override
    assert cirq.has_unitary(FullyImplemented(True))
    assert not cirq.has_unitary(FullyImplemented(False))

    # Test if decomposed operations _has_unitary_
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')

    assert cirq.has_unitary(Decomposable((a, b), True))
    assert not cirq.has_unitary(Decomposable((a,), False))
    assert cirq.has_unitary(DummyOperation((a,)))
    assert cirq.has_unitary(DummyOperation((a, b)))

    # Test _decompose_and_get_unitary
    from cirq.protocols.unitary import _decompose_and_get_unitary
    np.testing.assert_allclose(_decompose_and_get_unitary(
        Decomposable((a,), True)), m)
    np.testing.assert_allclose(_decompose_and_get_unitary(
        Decomposable((a, b), True)), m2)
    np.testing.assert_allclose(_decompose_and_get_unitary(
        DecomposableOrder((a, b, c))), m3)
    np.testing.assert_allclose(_decompose_and_get_unitary(
        DummyOperation((a,))), np.eye(2))
    np.testing.assert_allclose(_decompose_and_get_unitary(
        DummyOperation((a, b))), np.eye(4))

    # Test if decomposed operations has _unitary_
    np.testing.assert_allclose(cirq.unitary(Decomposable((a,), True)), m)
    np.testing.assert_allclose(cirq.unitary(Decomposable((a, b), True)), m2)
    np.testing.assert_allclose(cirq.unitary(DecomposableOrder((a, b, c))), m3)
    np.testing.assert_allclose(cirq.unitary(DummyOperation((a,))), np.eye(2))
    np.testing.assert_allclose(cirq.unitary(DummyOperation((a, b))), np.eye(4))

    assert cirq.unitary(DecomposableNoUnitary((a,)), None) is None
