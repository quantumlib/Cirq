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

import cirq


def apply_channel(rho, chan):
    res = np.zeros_like(rho)
    for kraus in chan:
        res += kraus @ rho @ np.conj(kraus).T
    return res


def assert_same_effect(rho, a, b):
    res1 = apply_channel(rho, a)
    res2 = apply_channel(rho, b)

    np.testing.assert_array_almost_equal(res1, res2)


def test_thermal_relaxation_recovers_amplitude_damping():
    for a in np.linspace(0, 1, 15):
        for b in np.linspace(0, 1, 15):
            tr = cirq.thermal_relaxation(a, b, b)

            tr2 = cirq.generalized_amplitude_damp(a, b)

            state = cirq.testing.random_superposition(2)
            rho = np.outer(state, state)

            assert_same_effect(rho, cirq.channel(tr2), cirq.channel(tr))


def test_thermal_relaxation_recovers_phase_damping():
    for a in np.linspace(0, 1, 15):
        tr = cirq.thermal_relaxation(1.0, 0.0, a)

        tr2 = cirq.phase_damp(a)

        state = cirq.testing.random_superposition(2)
        rho = np.outer(state, state)

        assert_same_effect(rho, cirq.channel(tr2), cirq.channel(tr))


def test_thermal_relaxation_simultaneous_decay_diagonal():
    rho = np.eye(2) * 0.5

    ch = cirq.thermal_relaxation(0.95, 0.1, 0.3)
    res = apply_channel(rho, cirq.channel(ch))
    expected = np.array([[0.545, 0],
                         [0, 0.455]])

    np.testing.assert_array_almost_equal(res, expected)


def test_thermal_relaxation_simultaneous_decay_full():
    rho = np.array([[0.5, 0.5],
                    [0.5, 0.5]])

    ch = cirq.thermal_relaxation(0.95, 0.1, 0.3)
    res = apply_channel(rho, cirq.channel(ch))
    expected = np.array([[0.545, 0.41833],
                         [0.41833, 0.455]])

    np.testing.assert_array_almost_equal(res, expected)


def test_thermal_relaxation_invalid_probs():
    with pytest.raises(ValueError, match='p'):
        cirq.thermal_relaxation(-5, 0.5, 0.5)

    with pytest.raises(ValueError, match='gamma'):
        cirq.thermal_relaxation(0.5, -5, 0.5)

    with pytest.raises(ValueError, match='beta'):
        cirq.thermal_relaxation(0.5, 0.5, -5)


def test_thermal_relaxation_non_cp():
    with pytest.raises(ValueError, match='CP requirement.'):
        cirq.thermal_relaxation(0.1, 0.9, 0.3)

    with pytest.raises(ValueError, match='CP requirement.'):
        cirq.thermal_relaxation(0.9, 0.5, 0.4)
