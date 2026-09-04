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

from __future__ import annotations

import numpy as np
import pytest

import cirq


@pytest.mark.parametrize('p', [0.0, 0.1, 0.6, 1.0])
def test_validate_probability_valid(p) -> None:
    assert p == cirq.validate_probability(p, 'p')


@pytest.mark.parametrize('p', [-0.1, 1.1])
def test_validate_probability_invalid(p) -> None:
    with pytest.raises(ValueError, match='p'):
        cirq.validate_probability(p, 'p')


@pytest.mark.parametrize(
    'p',
    [
        np.float16(0.25),
        np.float32(0.25),
        np.float64(0.6),
        np.int8(0),
        np.int32(1),
        np.uint8(1),
        np.uint64(0),
    ],
)
def test_validate_probability_numpy_scalars(p) -> None:
    assert cirq.validate_probability(p, 'p') == p


@pytest.mark.parametrize('p', [np.float32(-0.1), np.float64(1.1), np.int32(-1), np.int64(2)])
def test_validate_probability_numpy_scalars_invalid(p) -> None:
    with pytest.raises(ValueError, match='p'):
        cirq.validate_probability(p, 'p')
