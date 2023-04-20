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
from sympy.core.assumptions import assumptions

import cirq
import cirq_google.study.var as var


def test_parameter_to_symbol():
    param = var.Parameter(path=('test', 'subdir'))
    s = var.Var(param, range(10), label='repetitions')
    assert assumptions(s.symbol).get('registry') is True
    assert str(s.symbol) == 'test/subdir'
    assert s.to_parameter() == param
    assert s.unit == ''
    assert s.keys == ['test/subdir']

    param = var.Parameter(path=('test', 'subdir'), idx=4)
    s = var.Var(param, range(10), label='repetitions')
    assert assumptions(s.symbol).get('registry') is True
    assert str(s.symbol) == 'test/subdir#4'
    assert s.to_parameter() == param
    assert s.unit == ''
    assert s.keys == ['test/subdir#4']

    param = var.Parameter(path=('test', 'subdir'))
    s = var.Var(param, range(10), label='repetitions', unit='MHz')
    assert assumptions(s.symbol).get('registry') is True
    assert str(s.symbol) == 'test/subdir$MHz'
    assert s.to_parameter() == param
    assert s.unit == 'MHz'
    assert s.keys == ['test/subdir$MHz']

    param = var.Parameter(path=('test', 'subdir'), idx=2)
    s = var.Var(param, range(10), label='repetitions', unit='ns')
    assert assumptions(s.symbol).get('registry') is True
    assert str(s.symbol) == 'test/subdir#2$ns'
    assert s.to_parameter() == param
    assert s.unit == 'ns'
    assert s.keys == ['test/subdir#2$ns']


def test_symbol():
    s = var.Var(sympy.Symbol('r'), range(10), label='repetitions')
    assert assumptions(s.symbol).get('registry') is None
    assert s.symbol == sympy.Symbol('r')
    assert s.keys == ['r']
    assert s.unit == ''
    with pytest.raises(ValueError, match='not a Parameter'):
        _ = s.to_parameter()


def test_str():
    s = var.Var('r', range(10), label='repetitions')
    assert assumptions(s.symbol).get('registry') is None
    assert s.symbol == sympy.Symbol('r')
    assert s.keys == ['r']
    assert s.unit == ''
    with pytest.raises(ValueError, match='not a Parameter'):
        _ = s.to_parameter()


def test_invalid():
    with pytest.raises(ValueError, match='Unknown descriptor'):
        _ = var.Var(cirq.LineQubit(4), range(10), label='repetitions')


def test_repr():
    param = var.Parameter(path=('test', 'subdir'), idx=2)
    s = var.Var(param, range(4), label='repetitions', unit='ns')
    assert repr(s) == (
        "cirq_google.Var("
        "sympy.Symbol(\'test/subdir#2$ns\', "
        "registry=True, with_idx=True, with_unit=True)"
        ", [0, 1, 2, 3],unit=\'ns\', label=repetitions)"
    )


def test_values():
    param = var.Parameter(path=('test', 'subdir'), idx=2)
    s = var.Var(param, range(4), label='repetitions', unit='ns')
    name = 'test/subdir#2$ns'
    assert tuple(s.param_tuples()) == (((name, 0),), ((name, 1),), ((name, 2),), ((name, 3),))
