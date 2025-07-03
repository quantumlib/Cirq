# Copyright 2025 The Cirq Developers
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
import tunits as tu

import cirq
from cirq_google.study import symbol_util as su


def test_dict_param_name():
    d = {"a": 54, "b": sympy.Symbol("t"), "c": sympy.Symbol("t"), "d": "sd"}

    assert su.dict_param_name(None) == set()
    assert su.dict_param_name(d) == {"t"}


@pytest.mark.parametrize(
    "d,expected",
    [
        (None, False),
        ({}, False),
        ({"a": 50}, False),
        ({"a": 54, "b": sympy.Symbol("t"), "c": sympy.Symbol("t"), "d": "sd"}, True),
    ],
)
def test_is_parameterized_dict(d, expected):
    assert su.is_parameterized_dict(d) == expected


def test_direct_symbol_replacement():
    value_list = [sympy.Symbol("t"), sympy.Symbol("v"), sympy.Symbol("z"), 123, "fd"]
    resolver = cirq.ParamResolver({"t": 5 * tu.ns, sympy.Symbol("v"): 8 * tu.GHz})
    value_resolved = [su.direct_symbol_replacement(v, resolver) for v in value_list]

    assert value_resolved == [5 * tu.ns, 8 * tu.GHz, sympy.Symbol("z"), 123, "fd"]
