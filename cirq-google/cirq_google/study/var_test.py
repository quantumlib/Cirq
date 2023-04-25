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
import pytest

import sympy

import cirq_google.study.var as var


def test_parameters():
    param = var.Parameter(path=('test', 'subdir'))
    s = var.Var(param, range(10), label='repetitions')
    assert str(s.symbol) == 'repetitions'
    assert s.descriptor == param
    assert s.unit is None
    assert s.keys == ['repetitions']

    param = var.Parameter(path=('test', 'subdir'), idx=4)
    s = var.Var(param, range(10), label='repetitions')
    assert str(s.symbol) == 'repetitions'
    assert s.descriptor == param
    assert s.unit is None

    param = var.Parameter(path=('test', 'subdir'))
    s = var.Var(param, range(10), label='repetitions', unit='MHz')
    assert str(s.symbol) == 'repetitions'
    assert s.descriptor == param
    assert s.unit == 'MHz'
    assert s.keys == ['repetitions']


def test_symbol():
    s = var.Var(sympy.Symbol('r'), range(5), label='repetitions')
    assert s.symbol == sympy.Symbol('r')
    assert s.keys == ['r']
    assert s.unit is None
    assert repr(s) == ("cirq_google.Var('r', [0, 1, 2, 3, 4], label=\'repetitions\')")


def test_str():
    s = var.Var('r', range(10), label='repetitions')
    assert s.symbol == sympy.Symbol('r')
    assert s.keys == ['r']
    assert s.unit is None


def test_invalid():
    param = var.Parameter(path=('test', 'subdir'), idx=4)
    with pytest.raises(ValueError, match='Label must be provided'):
        _ = var.Var(param, range(10), unit='ns')


def test_repr():
    param = var.Parameter(path=('test', 'subdir'), idx=2)
    s = var.Var(param, range(4), label='repetitions', unit='ns')
    assert repr(s) == (
        "cirq_google.Var("
        "Parameter(path=('test', 'subdir'), idx=2, value=None), "
        "[0, 1, 2, 3], unit=\'ns\', label=\'repetitions\')"
    )


def test_values():
    param = var.Parameter(path=('test', 'subdir'), idx=2)
    name = 'repetitions'
    s = var.Var(param, range(4), label=name, unit='ns')
    assert tuple(s.param_tuples()) == (((name, 0),), ((name, 1),), ((name, 2),), ((name, 3),))
