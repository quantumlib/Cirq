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

"""Tests for kraus_protocol.py."""

from typing import Iterable, Sequence, Tuple

import numpy as np
import pytest

import cirq


LOCAL_DEFAULT = [np.array([])]


def test_kraus_no_methods():
    class NoMethod:
        pass

    with pytest.raises(TypeError, match='no _kraus_ or _mixture_ or _unitary_ method'):
        _ = cirq.kraus(NoMethod())

    assert cirq.kraus(NoMethod(), None) is None
    assert cirq.kraus(NoMethod, NotImplemented) is NotImplemented
    assert cirq.kraus(NoMethod(), (1,)) == (1,)
    assert cirq.kraus(NoMethod(), LOCAL_DEFAULT) is LOCAL_DEFAULT

    assert not cirq.has_kraus(NoMethod())


def assert_not_implemented(val):
    with pytest.raises(TypeError, match='returned NotImplemented'):
        _ = cirq.kraus(val)

    assert cirq.kraus(val, None) is None
    assert cirq.kraus(val, NotImplemented) is NotImplemented
    assert cirq.kraus(val, (1,)) == (1,)
    assert cirq.kraus(val, LOCAL_DEFAULT) is LOCAL_DEFAULT

    assert not cirq.has_kraus(val)


def test_kraus_returns_not_implemented():
    class ReturnsNotImplemented:
        def _kraus_(self):
            return NotImplemented

    assert_not_implemented(ReturnsNotImplemented())


def test_mixture_returns_not_implemented():
    class ReturnsNotImplemented:
        def _mixture_(self):
            return NotImplemented

    assert_not_implemented(ReturnsNotImplemented())


def test_unitary_returns_not_implemented():
    class ReturnsNotImplemented:
        def _unitary_(self):
            return NotImplemented

    with pytest.raises(TypeError, match='returned NotImplemented'):
        _ = cirq.kraus(ReturnsNotImplemented())
    assert cirq.kraus(ReturnsNotImplemented(), None) is None
    assert cirq.kraus(ReturnsNotImplemented(), NotImplemented) is NotImplemented
    assert cirq.kraus(ReturnsNotImplemented(), (1,)) == (1,)
    assert cirq.kraus(ReturnsNotImplemented(), LOCAL_DEFAULT) is LOCAL_DEFAULT


def test_explicit_kraus():
    a0 = np.array([[0, 0], [1, 0]])
    a1 = np.array([[1, 0], [0, 0]])
    c = (a0, a1)

    class ReturnsKraus:
        def _kraus_(self) -> Sequence[np.ndarray]:
            return c

    assert cirq.kraus(ReturnsKraus()) is c
    assert cirq.kraus(ReturnsKraus(), None) is c
    assert cirq.kraus(ReturnsKraus(), NotImplemented) is c
    assert cirq.kraus(ReturnsKraus(), (1,)) is c
    assert cirq.kraus(ReturnsKraus(), LOCAL_DEFAULT) is c

    assert cirq.has_kraus(ReturnsKraus())


def test_kraus_fallback_to_mixture():
    m = ((0.3, cirq.unitary(cirq.X)), (0.4, cirq.unitary(cirq.Y)), (0.3, cirq.unitary(cirq.Z)))

    class ReturnsMixture:
        def _mixture_(self) -> Iterable[Tuple[float, np.ndarray]]:
            return m

    c = (
        np.sqrt(0.3) * cirq.unitary(cirq.X),
        np.sqrt(0.4) * cirq.unitary(cirq.Y),
        np.sqrt(0.3) * cirq.unitary(cirq.Z),
    )

    np.testing.assert_equal(cirq.kraus(ReturnsMixture()), c)
    np.testing.assert_equal(cirq.kraus(ReturnsMixture(), None), c)
    np.testing.assert_equal(cirq.kraus(ReturnsMixture(), NotImplemented), c)
    np.testing.assert_equal(cirq.kraus(ReturnsMixture(), (1,)), c)
    np.testing.assert_equal(cirq.kraus(ReturnsMixture(), LOCAL_DEFAULT), c)

    assert cirq.has_kraus(ReturnsMixture())


def test_kraus_fallback_to_unitary():
    u = np.array([[1, 0], [1, 0]])

    class ReturnsUnitary:
        def _unitary_(self) -> np.ndarray:
            return u

    np.testing.assert_equal(cirq.kraus(ReturnsUnitary()), (u,))
    np.testing.assert_equal(cirq.kraus(ReturnsUnitary(), None), (u,))
    np.testing.assert_equal(cirq.kraus(ReturnsUnitary(), NotImplemented), (u,))
    np.testing.assert_equal(cirq.kraus(ReturnsUnitary(), (1,)), (u,))
    np.testing.assert_equal(cirq.kraus(ReturnsUnitary(), LOCAL_DEFAULT), (u,))

    assert cirq.has_kraus(ReturnsUnitary())


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


@pytest.mark.parametrize('cls', [HasKraus, HasMixture, HasUnitary])
def test_has_kraus(cls):
    assert cirq.has_kraus(cls())


@pytest.mark.parametrize('decomposed_cls', [HasKraus, HasMixture, HasUnitary])
def test_has_kraus_when_decomposed(decomposed_cls):
    op = HasKrausWhenDecomposed(decomposed_cls).on(cirq.NamedQubit('test'))
    assert cirq.has_kraus(op)
    assert not cirq.has_kraus(op, allow_decompose=False)
