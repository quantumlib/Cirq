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

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pytest

import cirq
from cirq.protocols.apply_channel_protocol import _apply_kraus

LOCAL_DEFAULT: list[np.ndarray] = [np.array([])]


def test_kraus_no_methods() -> None:
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


def test_kraus_returns_not_implemented() -> None:
    class ReturnsNotImplemented:
        def _kraus_(self):
            return NotImplemented

    assert_not_implemented(ReturnsNotImplemented())


def test_mixture_returns_not_implemented() -> None:
    class ReturnsNotImplemented:
        def _mixture_(self):
            return NotImplemented

    assert_not_implemented(ReturnsNotImplemented())


def test_unitary_returns_not_implemented() -> None:
    class ReturnsNotImplemented:
        def _unitary_(self):
            return NotImplemented

    with pytest.raises(TypeError, match='returned NotImplemented'):
        _ = cirq.kraus(ReturnsNotImplemented())
    assert cirq.kraus(ReturnsNotImplemented(), None) is None
    assert cirq.kraus(ReturnsNotImplemented(), NotImplemented) is NotImplemented
    assert cirq.kraus(ReturnsNotImplemented(), (1,)) == (1,)
    assert cirq.kraus(ReturnsNotImplemented(), LOCAL_DEFAULT) is LOCAL_DEFAULT


def test_explicit_kraus() -> None:
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


def test_kraus_fallback_to_mixture() -> None:
    m = ((0.3, cirq.unitary(cirq.X)), (0.4, cirq.unitary(cirq.Y)), (0.3, cirq.unitary(cirq.Z)))

    class ReturnsMixture:
        def _mixture_(self) -> Iterable[tuple[float, np.ndarray]]:
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


def test_kraus_fallback_to_unitary() -> None:
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


class HasKraus(cirq.testing.SingleQubitGate):
    def _has_kraus_(self) -> bool:
        return True


class HasMixture(cirq.testing.SingleQubitGate):
    def _has_mixture_(self) -> bool:
        return True


class HasUnitary(cirq.testing.SingleQubitGate):
    def _has_unitary_(self) -> bool:
        return True


class HasKrausWhenDecomposed(cirq.testing.SingleQubitGate):
    def __init__(self, decomposed_cls):
        self.decomposed_cls = decomposed_cls

    def _decompose_(self, qubits):
        return [self.decomposed_cls().on(q) for q in qubits]


@pytest.mark.parametrize('cls', [HasKraus, HasMixture, HasUnitary])
def test_has_kraus(cls) -> None:
    assert cirq.has_kraus(cls())


@pytest.mark.parametrize('decomposed_cls', [HasKraus, HasMixture])
def test_has_kraus_when_decomposed(decomposed_cls) -> None:
    op = HasKrausWhenDecomposed(decomposed_cls).on(cirq.NamedQubit('test'))
    assert cirq.has_kraus(op)
    assert not cirq.has_kraus(op, allow_decompose=False)


def test_strat_kraus_from_apply_channel_returns_none():
    # Remove _kraus_ and _apply_channel_ methods
    class NoApplyChannelReset(cirq.ResetChannel):
        def _kraus_(self):
            return NotImplemented

        def _apply_channel_(self, args):
            return NotImplemented

    gate_no_apply = NoApplyChannelReset()
    with pytest.raises(
        TypeError,
        match="does have a _kraus_, _mixture_ or _unitary_ method, but it returned NotImplemented",
    ):
        cirq.kraus(gate_no_apply)


@pytest.mark.parametrize(
    'channel_cls,params',
    [
        (cirq.BitFlipChannel, (0.5,)),
        (cirq.PhaseFlipChannel, (0.3,)),
        (cirq.DepolarizingChannel, (0.2,)),
        (cirq.AmplitudeDampingChannel, (0.4,)),
        (cirq.PhaseDampingChannel, (0.25,)),
    ],
)
def test_kraus_fallback_to_apply_channel(channel_cls, params) -> None:
    """Kraus protocol falls back to _apply_channel_ when no _kraus_, _mixture_, or _unitary_."""
    # Create the expected channel and get its Kraus operators
    expected_channel = channel_cls(*params)
    expected_kraus = cirq.kraus(expected_channel)

    class TestChannel:
        def __init__(self, channel_cls, params):
            self.channel_cls = channel_cls
            self.params = params
            self.expected_kraus = cirq.kraus(channel_cls(*params))

        def _num_qubits_(self):
            return 1

        def _apply_channel_(self, args: cirq.ApplyChannelArgs):
            return _apply_kraus(self.expected_kraus, args)

    chan = TestChannel(channel_cls, params)
    kraus_ops = cirq.kraus(chan)

    # Compare the superoperator matrices for equivalence
    expected_super = sum(np.kron(k, k.conj()) for k in expected_kraus)
    actual_super = sum(np.kron(k, k.conj()) for k in kraus_ops)
    np.testing.assert_allclose(actual_super, expected_super, atol=1e-8)


def test_reset_channel_kraus_apply_channel_consistency():
    Reset = cirq.ResetChannel
    # Original gate
    gate = Reset()
    cirq.testing.assert_has_consistent_apply_channel(gate)
    cirq.testing.assert_consistent_channel(gate)

    # Remove _kraus_ method
    class NoKrausReset(Reset):
        def _kraus_(self):
            return NotImplemented

    gate_no_kraus = NoKrausReset()
    # Should still match the original superoperator
    np.testing.assert_allclose(cirq.kraus(gate), cirq.kraus(gate_no_kraus), atol=1e-8)


def test_kraus_channel_with_has_unitary():
    """CZSWAP has no unitary dunder method but has_unitary returns True."""
    op = cirq.CZSWAP.on(cirq.q(1), cirq.q(2))
    channels = cirq.kraus(op)
    assert len(channels) == 1
    np.testing.assert_allclose(channels[0], cirq.unitary(op))
