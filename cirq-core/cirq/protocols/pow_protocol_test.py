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

import pytest

import cirq


class NoMethod:
    pass


class ReturnsNotImplemented:
    def __pow__(self, exponent):
        return NotImplemented


class ReturnsExponent:
    def __pow__(self, exponent) -> int:
        return exponent


@pytest.mark.parametrize(
    'val',
    (
        NoMethod(),
        'text',
        object(),
        ReturnsNotImplemented(),
    ),
)
def test_powerless(val):
    assert cirq.pow(val, 5, None) is None
    assert cirq.pow(val, 2, NotImplemented) is NotImplemented

    # Don't assume X**1 == X if X doesn't define __pow__.
    assert cirq.pow(val, 1, None) is None


def test_pow_error():
    with pytest.raises(TypeError, match="returned NotImplemented"):
        _ = cirq.pow(ReturnsNotImplemented(), 3)
    with pytest.raises(TypeError, match="no __pow__ method"):
        _ = cirq.pow(NoMethod(), 3)


@pytest.mark.parametrize(
    'val,exponent,out',
    (
        (ReturnsExponent(), 2, 2),
        (1, 2, 1),
        (2, 3, 8),
    ),
)
def test_pow_with_result(val, exponent, out):
    assert (
        cirq.pow(val, exponent) == cirq.pow(val, exponent, default=None) == val**exponent == out
    )
