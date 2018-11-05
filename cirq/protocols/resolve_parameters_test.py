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
    r = cirq.ParamResolver({'a': 0})
    no = NoMethod()
    assert cirq.resolve_parameters(no, r) == no
    assert cirq.resolve_parameters(ni, r) == ni
    assert cirq.resolve_parameters(SimpleParameterSwitch(0), r).parameter == 0
    assert cirq.resolve_parameters(SimpleParameterSwitch('a'), r).parameter == 0
