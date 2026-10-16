# Copyright 2026 The Cirq Developers
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

_NUMPY_REAL_SCALAR_TYPES = (np.float32, np.float64, np.double, np.int32, np.int64, np.short)


def test_tparamval_includes_real_numpy_numbers_only() -> None:
    assert float in cirq.TParamVal.__args__
    assert np.integer in cirq.TParamVal.__args__
    assert np.floating in cirq.TParamVal.__args__
    assert sympy.Expr in cirq.TParamVal.__args__
    assert np.number not in cirq.TParamVal.__args__
    assert np.complexfloating not in cirq.TParamVal.__args__
    assert complex not in cirq.TParamVal.__args__


def test_tparamvalcomplex_keeps_numpy_number() -> None:
    assert complex in cirq.TParamValComplex.__args__
    assert np.number in cirq.TParamValComplex.__args__
    assert sympy.Expr in cirq.TParamValComplex.__args__


@pytest.mark.parametrize('dtype', _NUMPY_REAL_SCALAR_TYPES)
def test_tparamval_numpy_real_scalars_match_alias_bounds(dtype) -> None:
    val = dtype(1)
    assert isinstance(val, (np.integer, np.floating, float))
    assert not isinstance(val, np.complexfloating)


def test_tparamval_excludes_numpy_complex_scalars() -> None:
    assert isinstance(np.complex128(1j), np.number)
    assert isinstance(np.complex128(1j), np.complexfloating)
    assert not isinstance(np.complex128(1j), (np.integer, np.floating, float))
