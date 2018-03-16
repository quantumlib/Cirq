# Copyright 2018 Google LLC
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

"""Tests for parameter resolvers."""

import itertools

from cirq.study import resolver
from cirq.value import ParameterizedValue


def test_value_of():
    r = resolver.ParamResolver({'a': 0.5, 'b': 0.1})
    assert r.value_of(ParameterizedValue('a', 0.0)) == 0.5
    assert r.value_of(ParameterizedValue('a', 0.1)) == 0.5 + 0.1
    assert r.value_of(ParameterizedValue('', 0.5)) == 0.5
    assert r.value_of(ParameterizedValue('b', 0.0)) == 0.1
    assert r.value_of(ParameterizedValue('', 0.3)) == 0.3
    assert r.value_of(0.3) == 0.3


def test_param_dict():
    r = resolver.ParamResolver({'a': 0.5, 'b': 0.1})
    assert r.param_dict == {'a': 0.5, 'b': 0.1}


def test_hash():
    a = resolver.ParamResolver({})
    b = resolver.ParamResolver({'a': 0.0})
    c = resolver.ParamResolver({'a': 0.1})
    d = resolver.ParamResolver({'a': 0.0, 'b': 0.1})
    e = resolver.ParamResolver({'a': 0.3, 'b': 0.1})
    f = resolver.ParamResolver({'b': 0.1})
    g = resolver.ParamResolver({'c': 0.1})
    resolvers = [a, b, c, d, e, f, g]
    for r in resolvers:
        assert isinstance(hash(r), int)
    for r1, r2 in itertools.combinations(resolvers, 2):
        assert hash(r1) != hash(r2)
