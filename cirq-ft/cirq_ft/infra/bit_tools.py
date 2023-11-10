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

from typing import Iterator, Tuple

import numpy as np


def iter_bits(val: int, width: int, *, signed: bool = False) -> Iterator[int]:
    """Iterate over the bits in a binary representation of `val`.

    This uses a big-endian convention where the most significant bit
    is yielded first.

    Args:
        val: The integer value. Its bitsize must fit within `width`
        width: The number of output bits.
        signed: If True, the most significant bit represents the sign of
            the number (ones complement) which is 1 if val < 0 else 0.
    Raises:
        ValueError: If `val` is negative or if `val.bit_length()` exceeds `width`.
    """
    if val.bit_length() + int(val < 0) > width:
        raise ValueError(f"{val} exceeds width {width}.")
    if val < 0 and not signed:
        raise ValueError(f"{val} is negative.")
    if signed:
        yield 1 if val < 0 else 0
        width -= 1
    for b in f'{abs(val):0{width}b}':
        yield int(b)


def iter_bits_twos_complement(val: int, width: int) -> Iterator[int]:
    """Iterate over the bits in a binary representation of `val`.

    This uses a big-endian convention where the most significant bit
    is yielded first. Allows for negative values and represents these using twos
    complement.

    Args:
        val: The integer value. Its bitsize must fit within `width`
        width: The number of output bits.

    Raises:
        ValueError: If `val.bit_length()` exceeds `2 * width + 1`.
    """
    if (val.bit_length() - 1) // 2 > width:
        raise ValueError(f"{val} exceeds width {width}.")
    mask = (1 << width) - 1
    for b in f'{val&mask:0{width}b}':
        yield int(b)


def iter_bits_fixed_point(val: float, width: int, *, signed: bool = False) -> Iterator[int]:
    r"""Represent the floating point number -1 <= val <= 1 using `width` bits.

    $$
        val = \sum_{b=0}^{width - 1} val[b] / 2^{1+b}
    $$

    Args:
        val: Floating point number in [-1, 1]
        width: The number of output bits in fixed point binary representation of `val`.
        signed: If True, the most significant bit represents the sign of
            the number (ones complement) which is 1 if val < 0 else 0.

    Raises:
        ValueError: If val is not between [0, 1] (signed=False) / [-1, 1] (signed=True).
    """
    lb = -1 if signed else 0
    assert lb <= val <= 1, f"{val} must be between [{lb}, 1]"
    if signed:
        yield 1 if val < 0 else 0
        width -= 1
        val = abs(val)
    for _ in range(width):
        val = val * 2
        out_bit = np.floor(val)
        val = val - out_bit
        yield int(out_bit)


def float_as_fixed_width_int(val: float, width: int) -> Tuple[int, int]:
    """Returns a `width` length fixed point binary representation of `val` where -1 <= val <= 1."""
    bits = [*iter_bits_fixed_point(val, width, signed=True)]
    return bits[0], int(''.join(str(b) for b in bits[1:]), 2)
