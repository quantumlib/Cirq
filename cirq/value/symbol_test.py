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

from cirq.value import Symbol
from cirq.testing import EqualsTester


def test_parameterized_value_init():
    assert Symbol('a').name == 'a'
    assert Symbol('b').name == 'b'

def test_string_representation():
    assert str(Symbol('a1')) == 'a1'
    assert str(Symbol('_b23_')) == '_b23_'
    assert str(Symbol('1a')) == 'Symbol("1a")'
    assert str(Symbol('&%#')) == 'Symbol("&%#")'
    assert str(Symbol('')) == 'Symbol("")'

def test_parameterized_value_eq():
    eq = EqualsTester()
    eq.add_equality_group(Symbol('a'))
    eq.make_equality_pair(lambda: Symbol('rr'))
