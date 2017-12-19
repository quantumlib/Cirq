# Copyright 2017 Google LLC
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

from cirq.testing.lin_alg_utils import allclose_up_to_global_phase


def test_allclose_up_to_global_phase():
    assert allclose_up_to_global_phase(
        np.mat([[1]]),
        np.mat([[1]]))
    assert allclose_up_to_global_phase(
        np.mat([[1]]),
        np.mat([[-1]]))

    assert allclose_up_to_global_phase(
        np.mat([[0]]),
        np.mat([[0]]))

    assert allclose_up_to_global_phase(
        np.mat([[1, 2]]),
        np.mat([[1j, 2j]]))

    assert allclose_up_to_global_phase(
        np.mat([[1, 2.0000000001]]),
        np.mat([[1j, 2j]]))

    with pytest.raises(AssertionError):
        assert allclose_up_to_global_phase(
            np.mat([[1]]),
            np.mat([[1, 0]]))
    with pytest.raises(AssertionError):
        assert allclose_up_to_global_phase(
            np.mat([[1]]),
            np.mat([[2]]))
    with pytest.raises(AssertionError):
        assert allclose_up_to_global_phase(
            np.mat([[1]]),
            np.mat([[2]]))
