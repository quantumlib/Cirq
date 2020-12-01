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

import pytest

import cirq


def test_big_endian_bits_to_int():
    assert cirq.big_endian_bits_to_int([0, 1]) == 1
    assert cirq.big_endian_bits_to_int([1, 0]) == 2
    assert cirq.big_endian_bits_to_int([0, 1, 0]) == 2
    assert cirq.big_endian_bits_to_int([1, 0, 0, 1, 0]) == 18

    assert cirq.big_endian_bits_to_int([]) == 0
    assert cirq.big_endian_bits_to_int([0]) == 0
    assert cirq.big_endian_bits_to_int([0, 0]) == 0
    assert cirq.big_endian_bits_to_int([0, 0, 0]) == 0

    assert cirq.big_endian_bits_to_int([1]) == 1
    assert cirq.big_endian_bits_to_int([0, 1]) == 1
    assert cirq.big_endian_bits_to_int([0, 0, 1]) == 1


def test_big_endian_digits_to_int():
    with pytest.raises(ValueError, match=r'len\(base\)'):
        _ = cirq.big_endian_digits_to_int([1, 2, 3], base=[2, 3, 5, 7])
    with pytest.raises(ValueError, match='Out of range'):
        _ = cirq.big_endian_digits_to_int([105, 106, 107], base=4)

    assert cirq.big_endian_digits_to_int([0, 1], base=102) == 1
    assert cirq.big_endian_digits_to_int([1, 0], base=102) == 102
    assert cirq.big_endian_digits_to_int([1, 0], base=[5, 7]) == 7
    assert cirq.big_endian_digits_to_int([0, 1], base=[5, 7]) == 1
    assert cirq.big_endian_digits_to_int([1, 2, 3, 4], base=[2, 3, 5, 7]) == 200
    assert cirq.big_endian_digits_to_int([1, 2, 3, 4], base=10) == 1234

    # Use-once digit and base iterators.
    assert (
        cirq.big_endian_digits_to_int((e for e in [1, 2, 3, 4]), base=(e for e in [2, 3, 5, 7]))
        == 200
    )


def test_big_endian_int_to_bits():
    assert cirq.big_endian_int_to_bits(2, bit_count=4) == [0, 0, 1, 0]
    assert cirq.big_endian_int_to_bits(18, bit_count=8) == [0, 0, 0, 1, 0, 0, 1, 0]
    assert cirq.big_endian_int_to_bits(18, bit_count=4) == [0, 0, 1, 0]
    assert cirq.big_endian_int_to_bits(-3, bit_count=4) == [1, 1, 0, 1]


def test_big_endian_int_to_digits():
    with pytest.raises(ValueError, match='No digit count'):
        _ = cirq.big_endian_int_to_digits(0, base=10)
    with pytest.raises(ValueError, match='Inconsistent digit count'):
        _ = cirq.big_endian_int_to_digits(0, base=[], digit_count=1)
    with pytest.raises(ValueError, match='Inconsistent digit count'):
        _ = cirq.big_endian_int_to_digits(0, base=[2, 3], digit_count=20)
    with pytest.raises(ValueError, match='Out of range'):
        assert cirq.big_endian_int_to_digits(101, base=[], digit_count=0) == []
    with pytest.raises(ValueError, match='Out of range'):
        _ = cirq.big_endian_int_to_digits(10 ** 100, base=[2, 3, 4, 5, 6])

    assert cirq.big_endian_int_to_digits(0, base=[], digit_count=0) == []
    assert cirq.big_endian_int_to_digits(0, base=[]) == []
    assert cirq.big_endian_int_to_digits(11, base=3, digit_count=4) == [0, 1, 0, 2]
    assert cirq.big_endian_int_to_digits(11, base=[3, 3, 3, 3], digit_count=4) == [0, 1, 0, 2]
    assert cirq.big_endian_int_to_digits(11, base=[2, 3, 4], digit_count=3) == [0, 2, 3]

    # Use-once base iterators.
    assert cirq.big_endian_int_to_digits(11, base=(e for e in [2, 3, 4]), digit_count=3) == [
        0,
        2,
        3,
    ]
    assert cirq.big_endian_int_to_digits(11, base=(e for e in [2, 3, 4])) == [0, 2, 3]
