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

import numpy as np
import pytest

import cirq


class NoMethod:
    pass


class ReturnsNotImplemented:
    def _mixture_(self):
        return NotImplemented

    def _has_mixture_(self):
        return NotImplemented


class ReturnsValidTuple(cirq.SupportsMixture):
    def _mixture_(self):
        return ((0.4, 'a'), (0.6, 'b'))

    def _has_mixture_(self):
        return True


class ReturnsNonnormalizedTuple:
    def _mixture_(self):
        return ((0.4, 'a'), (0.4, 'b'))


class ReturnsNegativeProbability:
    def _mixture_(self):
        return ((0.4, 'a'), (-0.4, 'b'))


class ReturnsGreaterThanUnityProbability:
    def _mixture_(self):
        return ((1.2, 'a'), (0.4, 'b'))


class ReturnsMixtureButNoHasMixture:
    def _mixture_(self):
        return ((0.4, 'a'), (0.6, 'b'))


class ReturnsUnitary:
    def _unitary_(self):
        return np.ones((2, 2))

    def _has_unitary_(self):
        return True


class ReturnsNotImplementedUnitary:
    def _unitary_(self):
        return NotImplemented

    def _has_unitary_(self):
        return NotImplemented


@pytest.mark.parametrize(
    'val,mixture',
    (
        (ReturnsValidTuple(), ((0.4, 'a'), (0.6, 'b'))),
        (ReturnsNonnormalizedTuple(), ((0.4, 'a'), (0.4, 'b'))),
        (ReturnsUnitary(), ((1.0, np.ones((2, 2))),)),
    ),
)
def test_objects_with_mixture(val, mixture):
    expected_keys, expected_values = zip(*mixture)
    keys, values = zip(*cirq.mixture(val))
    np.testing.assert_almost_equal(keys, expected_keys)
    np.testing.assert_equal(values, expected_values)

    keys, values = zip(*cirq.mixture(val, ((0.3, 'a'), (0.7, 'b'))))
    np.testing.assert_almost_equal(keys, expected_keys)
    np.testing.assert_equal(values, expected_values)


@pytest.mark.parametrize(
    'val', (NoMethod(), ReturnsNotImplemented(), ReturnsNotImplementedUnitary())
)
def test_objects_with_no_mixture(val):
    with pytest.raises(TypeError, match="mixture"):
        _ = cirq.mixture(val)
    assert cirq.mixture(val, None) is None
    assert cirq.mixture(val, NotImplemented) is NotImplemented
    default = ((0.4, 'a'), (0.6, 'b'))
    assert cirq.mixture(val, default) == default


def test_has_mixture():
    assert cirq.has_mixture(ReturnsValidTuple())
    assert not cirq.has_mixture(ReturnsNotImplemented())
    assert cirq.has_mixture(ReturnsMixtureButNoHasMixture())
    assert cirq.has_mixture(ReturnsUnitary())
    assert not cirq.has_mixture(ReturnsNotImplementedUnitary())


def test_valid_mixture():
    cirq.validate_mixture(ReturnsValidTuple())


@pytest.mark.parametrize(
    'val,message',
    (
        (ReturnsNonnormalizedTuple(), '1.0'),
        (ReturnsNegativeProbability(), 'less than 0'),
        (ReturnsGreaterThanUnityProbability(), 'greater than 1'),
    ),
)
def test_invalid_mixture(val, message):
    with pytest.raises(ValueError, match=message):
        cirq.validate_mixture(val)


def test_missing_mixture():
    with pytest.raises(TypeError, match='_mixture_'):
        cirq.validate_mixture(NoMethod)
