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

from __future__ import annotations

import numpy as np
import pytest
import sympy

import cirq

_NUMPY_FLOAT_TYPES = (np.float32, np.float64, np.double)
_NUMPY_INT_TYPES = (np.int32, np.int64, np.short)
_NUMPY_UINT_TYPES = (np.uint8, np.uint16, np.uint32, np.uint64)
_NUMPY_NUMBER_SAMPLES = (
    np.float32(0.5),
    np.float64(0.5),
    np.double(1.5),
    np.int32(1),
    np.int64(3),
    np.short(-1),
)


def test_canonicalize_half_turns() -> None:
    assert cirq.canonicalize_half_turns(0) == 0
    assert cirq.canonicalize_half_turns(1) == +1
    assert cirq.canonicalize_half_turns(-1) == +1
    assert cirq.canonicalize_half_turns(0.5) == 0.5
    assert cirq.canonicalize_half_turns(1.5) == -0.5
    assert cirq.canonicalize_half_turns(-0.5) == -0.5
    assert cirq.canonicalize_half_turns(101.5) == -0.5
    # Variable sympy expression
    assert cirq.canonicalize_half_turns(sympy.Symbol('a')) == sympy.Symbol('a')
    assert cirq.canonicalize_half_turns(sympy.Symbol('a') + 1) == sympy.Symbol('a') + 1
    # Constant sympy expression
    assert cirq.canonicalize_half_turns(sympy.Symbol('a') * 0 + 3) == 1


def _assert_canonical_numpy(original, result, expected) -> None:
    assert result == expected
    assert type(result) is type(original)
    assert isinstance(result, np.number)
    assert -1 < result <= 1
    assert (float(result) - float(original)) % 2 == pytest.approx(0)


@pytest.mark.parametrize('dtype', _NUMPY_FLOAT_TYPES)
@pytest.mark.parametrize(
    'value,expected',
    [
        (0.0, 0.0),
        (1.0, 1.0),
        (-1.0, 1.0),
        (0.5, 0.5),
        (1.5, -0.5),
        (-0.5, -0.5),
        (101.5, -0.5),
        (2.0, 0.0),
        (2.5, 0.5),
        (-1.5, 0.5),
        (-2.0, 0.0),
        (3.0, 1.0),
        (-3.0, 1.0),
    ],
)
def test_canonicalize_half_turns_numpy_floats(dtype, value, expected) -> None:
    original = dtype(value)
    _assert_canonical_numpy(original, cirq.canonicalize_half_turns(original), expected)


@pytest.mark.parametrize('dtype', _NUMPY_INT_TYPES)
@pytest.mark.parametrize(
    'value,expected', [(0, 0), (1, 1), (-1, 1), (2, 0), (3, 1), (4, 0), (5, 1), (-2, 0), (-3, 1)]
)
def test_canonicalize_half_turns_numpy_ints(dtype, value, expected) -> None:
    original = dtype(value)
    _assert_canonical_numpy(original, cirq.canonicalize_half_turns(original), expected)


@pytest.mark.parametrize('dtype', _NUMPY_UINT_TYPES)
@pytest.mark.parametrize('value,expected', [(0, 0), (1, 1), (2, 0), (3, 1), (255, 1)])
def test_canonicalize_half_turns_numpy_uints(dtype, value, expected) -> None:
    original = dtype(value)
    _assert_canonical_numpy(original, cirq.canonicalize_half_turns(original), expected)


def test_canonicalize_half_turns_zero_dimensional_array_boundary() -> None:
    value = np.array(0.5)
    assert value.shape == ()
    assert not isinstance(value, np.number)
    assert np.ndarray not in cirq.TParamVal.__args__

    result = cirq.canonicalize_half_turns(value)
    assert isinstance(result, np.ndarray)
    assert result.shape == ()
    assert result.item() == 0.5


def test_canonicalize_half_turns_numpy_signed_zero() -> None:
    result = cirq.canonicalize_half_turns(np.float64(-0.0))
    assert result == 0.0
    assert not np.signbit(result)
    assert cirq.XPowGate(exponent=result) == cirq.XPowGate(exponent=0.0)


@pytest.mark.parametrize('value', [np.float64(np.nan), np.float64(np.inf), np.float64(-np.inf)])
def test_canonicalize_half_turns_numpy_nonfinite(value) -> None:
    with np.errstate(invalid='ignore'):
        result = cirq.canonicalize_half_turns(value)
    assert np.isnan(result)
    assert not cirq.is_parameterized(cirq.XPowGate(exponent=value))


@pytest.mark.parametrize('val', _NUMPY_NUMBER_SAMPLES)
def test_canonicalize_half_turns_numpy_number_instances(val) -> None:
    assert isinstance(val, np.number)
    result = cirq.canonicalize_half_turns(val)
    assert isinstance(result, np.number)
    assert type(result) is type(val)
    assert -1 < result <= 1
    assert (float(result) - float(val)) % 2 == pytest.approx(0)


def test_tparamval_includes_real_numpy_numbers_only() -> None:
    assert np.integer in cirq.TParamVal.__args__
    assert np.floating in cirq.TParamVal.__args__
    assert np.complexfloating not in cirq.TParamVal.__args__


@pytest.mark.parametrize('dtype', _NUMPY_FLOAT_TYPES + _NUMPY_INT_TYPES)
def test_tparamval_numpy_scalar_as_gate_exponent(dtype) -> None:
    val = dtype(1)
    assert isinstance(val, np.number)
    gate = cirq.XPowGate(exponent=val)
    stored: cirq.TParamVal = gate.exponent
    assert stored == val
    assert type(stored) is dtype
    assert not cirq.is_parameterized(gate)


@pytest.mark.parametrize('dtype', _NUMPY_FLOAT_TYPES)
def test_chosen_angle_to_canonical_half_turns_numpy_floats(dtype) -> None:
    original = dtype(1.5)
    result = cirq.chosen_angle_to_canonical_half_turns(half_turns=original)
    _assert_canonical_numpy(original, result, -0.5)


@pytest.mark.parametrize('dtype', _NUMPY_INT_TYPES)
def test_chosen_angle_to_canonical_half_turns_numpy_ints(dtype) -> None:
    original = dtype(3)
    result = cirq.chosen_angle_to_canonical_half_turns(half_turns=original)
    _assert_canonical_numpy(original, result, 1)


def test_chosen_angle_to_half_turns() -> None:
    assert cirq.chosen_angle_to_half_turns() == 1
    assert cirq.chosen_angle_to_half_turns(default=0.5) == 0.5
    assert cirq.chosen_angle_to_half_turns(half_turns=0.25, default=0.75) == 0.25
    np.testing.assert_allclose(cirq.chosen_angle_to_half_turns(rads=np.pi / 2), 0.5, atol=1e-8)
    np.testing.assert_allclose(cirq.chosen_angle_to_half_turns(rads=-np.pi / 4), -0.25, atol=1e-8)
    assert cirq.chosen_angle_to_half_turns(degs=90) == 0.5
    assert cirq.chosen_angle_to_half_turns(degs=1080) == 6.0
    assert cirq.chosen_angle_to_half_turns(degs=990) == 5.5

    with pytest.raises(ValueError):
        _ = cirq.chosen_angle_to_half_turns(half_turns=0, rads=0)
    with pytest.raises(ValueError):
        _ = cirq.chosen_angle_to_half_turns(half_turns=0, degs=0)
    with pytest.raises(ValueError):
        _ = cirq.chosen_angle_to_half_turns(degs=0, rads=0)
    with pytest.raises(ValueError):
        _ = cirq.chosen_angle_to_half_turns(half_turns=0, rads=0, degs=0)


def test_chosen_angle_to_canonical_half_turns() -> None:
    assert cirq.chosen_angle_to_canonical_half_turns() == 1
    assert cirq.chosen_angle_to_canonical_half_turns(default=0.5) == 0.5
    assert cirq.chosen_angle_to_canonical_half_turns(half_turns=0.25, default=0.75) == 0.25
    np.testing.assert_allclose(
        cirq.chosen_angle_to_canonical_half_turns(rads=np.pi / 2), 0.5, atol=1e-8
    )
    np.testing.assert_allclose(
        cirq.chosen_angle_to_canonical_half_turns(rads=-np.pi / 4), -0.25, atol=1e-8
    )
    assert cirq.chosen_angle_to_canonical_half_turns(degs=90) == 0.5
    assert cirq.chosen_angle_to_canonical_half_turns(degs=1080) == 0
    assert cirq.chosen_angle_to_canonical_half_turns(degs=990) == -0.5

    with pytest.raises(ValueError):
        _ = cirq.chosen_angle_to_canonical_half_turns(half_turns=0, rads=0)
    with pytest.raises(ValueError):
        _ = cirq.chosen_angle_to_canonical_half_turns(half_turns=0, degs=0)
    with pytest.raises(ValueError):
        _ = cirq.chosen_angle_to_canonical_half_turns(degs=0, rads=0)
    with pytest.raises(ValueError):
        _ = cirq.chosen_angle_to_canonical_half_turns(half_turns=0, rads=0, degs=0)
