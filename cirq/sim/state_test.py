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
"""Tests for state.py"""

import itertools
import pytest

import numpy as np

import cirq


def assert_dirac_notation(vec, expected, decimals=2):
    assert cirq.dirac_notation(np.array(vec), decimals=decimals) == expected


def test_dirac_notation():
    sqrt = np.sqrt(0.5)
    exp_pi_2 = 0.5 + 0.5j
    assert_dirac_notation([1], "|⟩")
    assert_dirac_notation([sqrt, sqrt], "(0.71)|0⟩ + (0.71)|1⟩")
    assert_dirac_notation([-sqrt, sqrt], "(-0.71)|0⟩ + (0.71)|1⟩")
    assert_dirac_notation([sqrt, -sqrt], "(0.71)|0⟩ + (-0.71)|1⟩")
    assert_dirac_notation([-sqrt, -sqrt], "(-0.71)|0⟩ + (-0.71)|1⟩")
    assert_dirac_notation([sqrt, 1j * sqrt], "(0.71)|0⟩ + (0.71j)|1⟩")
    assert_dirac_notation([sqrt, exp_pi_2], "(0.71)|0⟩ + (0.5+0.5j)|1⟩")
    assert_dirac_notation([exp_pi_2, -sqrt], "(0.5+0.5j)|0⟩ + (-0.71)|1⟩")
    assert_dirac_notation([exp_pi_2, 0.5 - 0.5j],
                          "(0.5+0.5j)|0⟩ + (0.5-0.5j)|1⟩")
    assert_dirac_notation([0.5, 0.5, -0.5, -0.5],
                        "(0.5)|00⟩ + (0.5)|01⟩ + (-0.5)|10⟩ + (-0.5)|11⟩")


def test_dirac_notation_partial_state():
    sqrt = np.sqrt(0.5)
    exp_pi_2 = 0.5 + 0.5j
    assert_dirac_notation([1, 0], "|0⟩")
    assert_dirac_notation([1j, 0], "(1j)|0⟩")
    assert_dirac_notation([0, 1], "|1⟩")
    assert_dirac_notation([0, 1j], "(1j)|1⟩")
    assert_dirac_notation([sqrt, 0 , 0, sqrt], "(0.71)|00⟩ + (0.71)|11⟩")
    assert_dirac_notation([sqrt, sqrt , 0, 0], "(0.71)|00⟩ + (0.71)|01⟩")
    assert_dirac_notation([exp_pi_2, 0, 0, exp_pi_2],
                        "(0.5+0.5j)|00⟩ + (0.5+0.5j)|11⟩")
    assert_dirac_notation([0, 0, 0, 1], "|11⟩")


def test_dirac_notation_precision():
    sqrt = np.sqrt(0.5)
    assert_dirac_notation([sqrt, sqrt], "(0.7)|0⟩ + (0.7)|1⟩", decimals=1)
    assert_dirac_notation([sqrt, sqrt], "(0.707)|0⟩ + (0.707)|1⟩", decimals=3)



def test_to_valid_state_vector():
    np.testing.assert_almost_equal(cirq.to_valid_state_vector(
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex64), 2),
        np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_almost_equal(cirq.to_valid_state_vector(
        np.array([0.0, 1.0, 0.0, 0.0], dtype=np.complex64), 2),
        np.array([0.0, 1.0, 0.0, 0.0]))
    np.testing.assert_almost_equal(cirq.to_valid_state_vector(0, 2),
                                   np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_almost_equal(cirq.to_valid_state_vector(1, 2),
                                   np.array([0.0, 1.0, 0.0, 0.0]))


def test_invalid_to_valid_state_vector():
    with pytest.raises(ValueError):
        _ = cirq.to_valid_state_vector(
            np.array([1.0, 0.0], dtype=np.complex64), 2)
    with pytest.raises(ValueError):
        _ = cirq.to_valid_state_vector(-1, 2)
    with pytest.raises(ValueError):
        _ = cirq.to_valid_state_vector(5, 2)
    with pytest.raises(TypeError):
        _ = cirq.to_valid_state_vector('not an int', 2)


def test_check_state():
    cirq.validate_normalized_state(
        np.array([0.5, 0.5, 0.5, 0.5], dtype=np.complex64),
                                   2)
    with pytest.raises(ValueError):
        cirq.validate_normalized_state(np.array([1, 1], dtype=np.complex64), 2)
    with pytest.raises(ValueError):
        cirq.validate_normalized_state(
            np.array([1.0, 0.2, 0.0, 0.0], dtype=np.complex64), 2)
    with pytest.raises(ValueError):
        cirq.validate_normalized_state(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), 2)


def test_sample_state_little_endian():
    results = []
    for x in range(8):
        state = cirq.to_valid_state_vector(x, 3)
        sample = cirq.sample_state(state, [2, 1, 0])
        results.append(sample)
    expected = [[list(x)] for x in
                list(itertools.product([False, True], repeat=3))]
    assert results == expected


def test_sample_state_partial_indices():
    for index in range(3):
        for x in range(8):
            state = cirq.to_valid_state_vector(x, 3)
            print(index, x)
            assert cirq.sample_state(state, [index]) == [
                [bool(1 & (x >> index))]]
