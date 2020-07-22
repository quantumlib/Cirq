# Copyright 2020 The Cirq developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest

import cirq
from cirq.work.observable_measurement_data import _get_real_coef, _obs_vals_from_measurements


def test_get_real_coef():
    q0 = cirq.LineQubit(0)
    assert _get_real_coef(cirq.Z(q0) * 2) == 2
    assert _get_real_coef(cirq.Z(q0) * complex(2.0)) == 2
    with pytest.raises(ValueError):
        _get_real_coef(cirq.Z(q0) * 2.j)


def test_obs_vals_from_measurements():
    bitstrings = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    qubit_to_index = {a: 0, b: 1}
    obs = cirq.Z(a) * cirq.Z(b) * 10
    vals = _obs_vals_from_measurements(bitstrings, qubit_to_index, obs)
    should_be = [10, -10, -10, 10]
    np.testing.assert_equal(vals, should_be)
