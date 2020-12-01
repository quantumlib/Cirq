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

"""Tests for channel.py."""

from typing import Iterable, Sequence, Tuple

import numpy as np
import pytest

import cirq


LOCAL_DEFAULT = [np.array([])]


def test_channel_no_methods():
    class NoMethod:
        pass

    with pytest.raises(TypeError, match='no _channel_ or _mixture_ or _unitary_ method'):
        _ = cirq.channel(NoMethod())

    assert cirq.channel(NoMethod(), None) is None
    assert cirq.channel(NoMethod, NotImplemented) is NotImplemented
    assert cirq.channel(NoMethod(), (1,)) == (1,)
    assert cirq.channel(NoMethod(), LOCAL_DEFAULT) is LOCAL_DEFAULT

    assert not cirq.has_channel(NoMethod())


def assert_not_implemented(val):
    with pytest.raises(TypeError, match='returned NotImplemented'):
        _ = cirq.channel(val)

    assert cirq.channel(val, None) is None
    assert cirq.channel(val, NotImplemented) is NotImplemented
    assert cirq.channel(val, (1,)) == (1,)
    assert cirq.channel(val, LOCAL_DEFAULT) is LOCAL_DEFAULT

    assert not cirq.has_channel(val)


def test_channel_returns_not_implemented():
    class ReturnsNotImplemented:
        def _channel_(self):
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
        _ = cirq.channel(ReturnsNotImplemented())
    assert cirq.channel(ReturnsNotImplemented(), None) is None
    assert cirq.channel(ReturnsNotImplemented(), NotImplemented) is NotImplemented
    assert cirq.channel(ReturnsNotImplemented(), (1,)) == (1,)
    assert cirq.channel(ReturnsNotImplemented(), LOCAL_DEFAULT) is LOCAL_DEFAULT


def test_channel():
    a0 = np.array([[0, 0], [1, 0]])
    a1 = np.array([[1, 0], [0, 0]])
    c = (a0, a1)

    class ReturnsChannel:
        def _channel_(self) -> Sequence[np.ndarray]:
            return c

    assert cirq.channel(ReturnsChannel()) is c
    assert cirq.channel(ReturnsChannel(), None) is c
    assert cirq.channel(ReturnsChannel(), NotImplemented) is c
    assert cirq.channel(ReturnsChannel(), (1,)) is c
    assert cirq.channel(ReturnsChannel(), LOCAL_DEFAULT) is c

    assert cirq.has_channel(ReturnsChannel())


def test_channel_fallback_to_mixture():
    m = ((0.3, cirq.unitary(cirq.X)), (0.4, cirq.unitary(cirq.Y)), (0.3, cirq.unitary(cirq.Z)))

    class ReturnsMixture:
        def _mixture_(self) -> Iterable[Tuple[float, np.ndarray]]:
            return m

    c = (
        np.sqrt(0.3) * cirq.unitary(cirq.X),
        np.sqrt(0.4) * cirq.unitary(cirq.Y),
        np.sqrt(0.3) * cirq.unitary(cirq.Z),
    )

    np.allclose(cirq.channel(ReturnsMixture()), c)
    np.allclose(cirq.channel(ReturnsMixture(), None), c)
    np.allclose(cirq.channel(ReturnsMixture(), NotImplemented), c)
    np.allclose(cirq.channel(ReturnsMixture(), (1,)), c)
    np.allclose(cirq.channel(ReturnsMixture(), LOCAL_DEFAULT), c)

    assert cirq.has_channel(ReturnsMixture())


def test_channel_fallback_to_unitary():
    u = np.array([[1, 0], [1, 0]])

    class ReturnsUnitary:
        def _unitary_(self) -> np.ndarray:
            return u

    np.testing.assert_equal(cirq.channel(ReturnsUnitary()), (u,))
    np.testing.assert_equal(cirq.channel(ReturnsUnitary(), None), (u,))
    np.testing.assert_equal(cirq.channel(ReturnsUnitary(), NotImplemented), (u,))
    np.testing.assert_equal(cirq.channel(ReturnsUnitary(), (1,)), (u,))
    np.testing.assert_equal(cirq.channel(ReturnsUnitary(), LOCAL_DEFAULT), (u,))

    assert cirq.has_channel(ReturnsUnitary())


class HasChannel(cirq.SingleQubitGate):
    def _has_channel_(self) -> bool:
        return True


class HasMixture(cirq.SingleQubitGate):
    def _has_mixture_(self) -> bool:
        return True


class HasUnitary(cirq.SingleQubitGate):
    def _has_unitary_(self) -> bool:
        return True


class HasChannelWhenDecomposed(cirq.SingleQubitGate):
    def __init__(self, decomposed_cls):
        self.decomposed_cls = decomposed_cls

    def _decompose_(self, qubits):
        return [self.decomposed_cls().on(q) for q in qubits]


@pytest.mark.parametrize('cls', [HasChannel, HasMixture, HasUnitary])
def test_has_channel(cls):
    assert cirq.has_channel(cls())


@pytest.mark.parametrize('decomposed_cls', [HasChannel, HasMixture, HasUnitary])
def test_has_channel_when_decomposed(decomposed_cls):
    op = HasChannelWhenDecomposed(decomposed_cls).on(cirq.NamedQubit('test'))
    assert cirq.has_channel(op)
    assert not cirq.has_channel(op, allow_decompose=False)
