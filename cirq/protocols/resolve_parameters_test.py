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

    a, b, c = tuple(sympy.Symbol(l) for l in 'abc')
    x, y, z = 0, 4, 7
    resolver = {a: x, b: y, c: z}

    assert cirq.resolve_parameters((a, b, c), resolver) == (x, y, z)
    assert cirq.resolve_parameters([a, b, c], resolver) == [x, y, z]
    assert cirq.resolve_parameters((x, y, z), resolver) == (x, y, z)
    assert cirq.resolve_parameters([x, y, z], resolver) == [x, y, z]
    assert cirq.resolve_parameters((), resolver) == ()
    assert cirq.resolve_parameters([], resolver) == []

    assert not cirq.is_parameterized((x, y))
    assert not cirq.is_parameterized([x, y])
    assert cirq.is_parameterized([a, b])
    assert cirq.is_parameterized([a, x])
    assert cirq.is_parameterized((a, b))
    assert cirq.is_parameterized((a, x))
    assert not cirq.is_parameterized(())
    assert not cirq.is_parameterized([])


def test_skips_empty_resolution():
    class Tester:
        def _resolve_parameters_(self, param_resolver):
            return 5

    t = Tester()
    assert cirq.resolve_parameters(t, {}) is t
    assert cirq.resolve_parameters(t, {'x': 2}) == 5
