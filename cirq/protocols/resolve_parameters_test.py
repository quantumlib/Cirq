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

import pytest
import sympy

import cirq
from cirq.study import ParamResolver


def test_resolve_parameters():

    class NoMethod:
        pass

    class ReturnsNotImplemented:
        def _is_parameterized_(self):
            return NotImplemented

        def _resolve_parameters_(self, resolver):
            return NotImplemented

    class SimpleParameterSwitch:
        def __init__(self, var):
            self.parameter = var

        def _is_parameterized_(self) -> bool:
            return self.parameter == 0

        def _resolve_parameters_(self, resolver: ParamResolver):
            self.parameter = resolver.value_of(self.parameter)
            return self

    assert not cirq.is_parameterized(NoMethod())
    assert not cirq.is_parameterized(ReturnsNotImplemented())
    assert not cirq.is_parameterized(SimpleParameterSwitch('a'))
    assert cirq.is_parameterized(SimpleParameterSwitch(0))

    ni = ReturnsNotImplemented()
    d = {'a': 0}
    r = cirq.ParamResolver(d)
    no = NoMethod()
    assert cirq.resolve_parameters(no, r) == no
    assert cirq.resolve_parameters(no, d) == no
    assert cirq.resolve_parameters(ni, r) == ni
    assert cirq.resolve_parameters(SimpleParameterSwitch(0), r).parameter == 0
    assert cirq.resolve_parameters(SimpleParameterSwitch('a'), r).parameter == 0
    assert cirq.resolve_parameters(SimpleParameterSwitch('a'), d).parameter == 0
    assert cirq.resolve_parameters(sympy.Symbol('a'), r) == 0


def test_skips_empty_resolution():
    class Tester:
        def _resolve_parameters_(self, param_resolver):
            return 5

    t = Tester()
    assert cirq.resolve_parameters(t, {}) is t
    assert cirq.resolve_parameters(t, {'x': 2}) == 5


def test_check_parameters():
    c = cirq.Circuit.from_ops(cirq.X(cirq.LineQubit(0))**sympy.Symbol('x'))
    r = cirq.ParamResolver({'y': 0.5})
    c_p = cirq.check_parameters(c, r)
    assert str(c_p) == str('[\'x\']')
    r_c = cirq.resolve_parameters(c, r)
    with pytest.raises(ValueError):
        cirq.Simulator().run(r_c)

    sweep = cirq.Linspace(key='y', start=0.1, stop=0.9, length=5)
    c_p = cirq.check_parameters(c, sweep)
    assert str(c_p) == str('[\'x\']')
    with pytest.raises(ValueError):
        cirq.Simulator().run_sweep(c, params=sweep)
