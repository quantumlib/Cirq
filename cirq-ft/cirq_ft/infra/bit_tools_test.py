# Copyright 2023 The Cirq Developers
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

import math
import random

import pytest
from cirq_ft.infra.bit_tools import (
    float_as_fixed_width_int,
    iter_bits,
    iter_bits_fixed_point,
    iter_bits_twos_complement,
)


def test_iter_bits():
    assert list(iter_bits(0, 2)) == [0, 0]
    assert list(iter_bits(0, 3, signed=True)) == [0, 0, 0]
    assert list(iter_bits(1, 2)) == [0, 1]
    assert list(iter_bits(1, 2, signed=True)) == [0, 1]
    assert list(iter_bits(-1, 2, signed=True)) == [1, 1]
    assert list(iter_bits(2, 2)) == [1, 0]
    assert list(iter_bits(2, 3, signed=True)) == [0, 1, 0]
    assert list(iter_bits(-2, 3, signed=True)) == [1, 1, 0]
    assert list(iter_bits(3, 2)) == [1, 1]
    with pytest.raises(ValueError):
        assert list(iter_bits(4, 2)) == [1, 0, 0]
    with pytest.raises(ValueError):
        _ = list(iter_bits(-3, 4))


def test_iter_bits_twos():
    assert list(iter_bits_twos_complement(0, 4)) == [0, 0, 0, 0]
    assert list(iter_bits_twos_complement(1, 4)) == [0, 0, 0, 1]
    assert list(iter_bits_twos_complement(-2, 4)) == [1, 1, 1, 0]
    assert list(iter_bits_twos_complement(-3, 4)) == [1, 1, 0, 1]
    with pytest.raises(ValueError):
        _ = list(iter_bits_twos_complement(100, 2))


random.seed(1234)


@pytest.mark.parametrize('val', [random.uniform(-1, 1) for _ in range(10)])
@pytest.mark.parametrize('width', [*range(2, 20, 2)])
@pytest.mark.parametrize('signed', [True, False])
def test_iter_bits_fixed_point(val, width, signed):
    if (val < 0) and not signed:
        with pytest.raises(AssertionError):
            _ = [*iter_bits_fixed_point(val, width, signed=signed)]
    else:
        bits = [*iter_bits_fixed_point(val, width, signed=signed)]
        if signed:
            sign, bits = bits[0], bits[1:]
            assert sign == (1 if val < 0 else 0)
        val = abs(val)
        approx_val = math.fsum([b * (1 / 2 ** (1 + i)) for i, b in enumerate(bits)])
        unsigned_width = width - 1 if signed else width
        assert math.isclose(
            val, approx_val, abs_tol=1 / 2**unsigned_width
        ), f'{val}:{approx_val}:{width}'
        bits_from_int = [
            *iter_bits(float_as_fixed_width_int(val, unsigned_width + 1)[1], unsigned_width)
        ]
        assert bits == bits_from_int
