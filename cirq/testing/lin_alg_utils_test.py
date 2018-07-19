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

from cirq.testing import (
    random_unitary,
    assert_allclose_up_to_global_phase,
)
from cirq.linalg import is_unitary


def test_random_unitary():
    u1 = random_unitary(2)
    u2 = random_unitary(2)
    assert is_unitary(u1)
    assert is_unitary(u2)
    assert not np.allclose(u1, u2)


def test_assert_allclose_up_to_global_phase():
    assert_allclose_up_to_global_phase(
        np.array([[1]]),
        np.array([[1j]]),
        atol=0)

    with pytest.raises(AssertionError):
        assert_allclose_up_to_global_phase(
            np.array([[1]]),
            np.array([[2]]),
            atol=0)

    assert_allclose_up_to_global_phase(
        np.array([[1e-8, -1, 1e-8]]),
        np.array([[1e-8, 1, 1e-8]]),
        atol=1e-6)

    with pytest.raises(AssertionError):
        assert_allclose_up_to_global_phase(
            np.array([[1e-4, -1, 1e-4]]),
            np.array([[1e-4, 1, 1e-4]]),
            atol=1e-6)

    assert_allclose_up_to_global_phase(
        np.array([[1, 2], [3, 4]]),
        np.array([[-1, -2], [-3, -4]]),
        atol=0)
