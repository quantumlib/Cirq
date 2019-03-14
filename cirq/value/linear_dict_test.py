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

import numpy as np
import pytest

import cirq


@pytest.mark.parametrize('terms, bool_value', (
    ({}, False),
    ({'X': 0}, False),
    ({'Z': 1e-12}, True),
    ({'Y': 1}, True),
))
def test_bool(terms, bool_value):
    linear_dict = cirq.LinearDict(terms)
    if linear_dict:
        assert bool_value
    else:
        assert not bool_value


@pytest.mark.parametrize('terms_1, terms_2', (
    ({}, {}),
    ({}, {'X': 0}),
    ({'X': 0.0}, {'Y': 0.0}),
    ({'a': 1}, {'a': 1, 'b': 0}),
))
def test_equal(terms_1, terms_2):
    linear_dict_1 = cirq.LinearDict(terms_1)
    linear_dict_2 = cirq.LinearDict(terms_2)
    assert linear_dict_1 == linear_dict_2
    assert linear_dict_2 == linear_dict_1
    assert not linear_dict_1 != linear_dict_2
    assert not linear_dict_2 != linear_dict_1


@pytest.mark.parametrize('terms_1, terms_2', (
    ({}, {'a': 1}),
    ({'X': 1e-12}, {'X': 0}),
    ({'X': 0.0}, {'Y': 0.1}),
    ({'X': 1}, {'X': 1, 'Z': 1e-12}),
))
def test_unequal(terms_1, terms_2):
    linear_dict_1 = cirq.LinearDict(terms_1)
    linear_dict_2 = cirq.LinearDict(terms_2)
    assert linear_dict_1 != linear_dict_2
    assert linear_dict_2 != linear_dict_1
    assert not linear_dict_1 == linear_dict_2
    assert not linear_dict_2 == linear_dict_1


@pytest.mark.parametrize('terms_1, terms_2', (
    ({}, {'X': 1e-9}),
    ({'X': 1e-12}, {'X': 0}),
    ({'X': 5e-10}, {'Y': 2e-11}),
    ({'X': 1.000000001}, {'X': 1, 'Z': 0}),
))
def test_approximately_equal(terms_1, terms_2):
    linear_dict_1 = cirq.LinearDict(terms_1)
    linear_dict_2 = cirq.LinearDict(terms_2)
    assert cirq.approx_eq(linear_dict_1, linear_dict_2)
    assert cirq.approx_eq(linear_dict_2, linear_dict_1)


@pytest.mark.parametrize('terms, tolerance, terms_expected', (
    ({'X': 1, 'Y': 2, 'Z': 3}, 2, {'Z': 3}),
    ({'X': 0.1, 'Y': 1, 'Z': 10}, 1e-3, {'X': 0.1, 'Y': 1, 'Z': 10}),
    ({'X': 1e-10, 'H': 1e-11}, 1e-9, {}),
    ({}, 1, {}),
))
def test_clean(terms, tolerance, terms_expected):
    linear_dict = cirq.LinearDict(terms)
    linear_dict.clean(tolerance=tolerance)
    expected = cirq.LinearDict(terms_expected)
    assert linear_dict == expected
    assert expected == linear_dict


@pytest.mark.parametrize('terms_1, terms_2, terms_expected', (
    ({}, {}, {}),
    ({}, {'X': 0.1}, {'X': 0.1}),
    ({'X': 1}, {'Y': 2}, {'X': 1, 'Y': 2}),
    ({'X': 1}, {'X': 1}, {'X': 2}),
    ({'X': 1, 'Y': 2}, {'Y': -2}, {'X': 1}),
))
def test_vector_addition(terms_1, terms_2, terms_expected):
    linear_dict_1 = cirq.LinearDict(terms_1)
    linear_dict_2 = cirq.LinearDict(terms_2)
    actual_1 = linear_dict_1 + linear_dict_2
    actual_2 = linear_dict_1
    actual_2 += linear_dict_2
    expected = cirq.LinearDict(terms_expected)
    assert actual_1 == expected
    assert actual_2 == expected
    assert actual_1 == actual_2


@pytest.mark.parametrize('terms_1, terms_2, terms_expected', (
    ({}, {}, {}),
    ({'a': 2}, {'a': 2}, {}),
    ({'a': 3}, {'a': 2}, {'a': 1}),
    ({'X': 1}, {'Y': 2}, {'X': 1, 'Y': -2}),
    ({'X': 1}, {'X': 1}, {}),
    ({'X': 1, 'Y': 2}, {'Y': 2}, {'X': 1}),
    ({'X': 1, 'Y': 2}, {'Y': 3}, {'X': 1, 'Y': -1}),
))
def test_vector_subtraction(terms_1, terms_2, terms_expected):
    linear_dict_1 = cirq.LinearDict(terms_1)
    linear_dict_2 = cirq.LinearDict(terms_2)
    actual_1 = linear_dict_1 - linear_dict_2
    actual_2 = linear_dict_1
    actual_2 -= linear_dict_2
    expected = cirq.LinearDict(terms_expected)
    assert actual_1 == expected
    assert actual_2 == expected
    assert actual_1 == actual_2


@pytest.mark.parametrize('terms, terms_expected', (
    ({}, {}),
    ({'key': 1}, {'key': -1}),
    ({'1': 10, '2': -20}, {'1': -10, '2': 20}),
))
def test_vector_negation(terms, terms_expected):
    linear_dict = cirq.LinearDict(terms)
    actual = -linear_dict
    expected = cirq.LinearDict(terms_expected)
    assert actual == expected
    assert expected == actual


@pytest.mark.parametrize('scalar, terms, terms_expected', (
    (2, {}, {}),
    (2, {'X': 1, 'Y': -2}, {'X': 2, 'Y': -4}),
    (0, {'abc': 10}, {}),
    (1j, {'X': 4j}, {'X': -4}),
    (-1, {'a': 10, 'b': -20}, {'a': -10, 'b': 20}),
))
def test_scalar_multiplication(scalar, terms, terms_expected):
    linear_dict = cirq.LinearDict(terms)
    actual_1 = scalar * linear_dict
    actual_2 = linear_dict * scalar
    expected = cirq.LinearDict(terms_expected)
    assert actual_1 == expected
    assert actual_2 == expected
    assert actual_1 == actual_2


@pytest.mark.parametrize('scalar, terms, terms_expected', (
    (2, {}, {}),
    (2, {'X': 6, 'Y': -2}, {'X': 3, 'Y': -1}),
    (1j, {'X': 1, 'Y': 1j}, {'X': -1j, 'Y': 1}),
    (-1, {'a': 10, 'b': -20}, {'a': -10, 'b': 20}),
))
def test_scalar_division(scalar, terms, terms_expected):
    linear_dict = cirq.LinearDict(terms)
    actual = linear_dict / scalar
    expected = cirq.LinearDict(terms_expected)
    assert actual == expected
    assert expected == actual


@pytest.mark.parametrize('expression, expected', (
    ((cirq.LinearDict({'X': 10}) + cirq.LinearDict({'X': 10, 'Y': -40})) / 20,
     cirq.LinearDict({'X': 1, 'Y': -2})),
    (cirq.LinearDict({'a': -2}) + 'a', cirq.LinearDict({'a': -1})),
    (cirq.LinearDict({'b': 2}) - 'b', 'b'),
))
def test_expressions(expression, expected):
    assert expression == expected
    assert not expression != expected
    assert cirq.approx_eq(expression, expected)


@pytest.mark.parametrize('terms, string', (
    ({}, '0'),
    ({'X': 1.5, 'Y': 1e-5}, '1.500*X'),
    ({'Y': 2}, '2.000*Y'),
    ({'X': 1, 'Y': -1j}, '1.000*X-1.000i*Y'),
    ({'X': np.sqrt(3)/3, 'Y': np.sqrt(3)/3, 'Z': np.sqrt(3)/3},
     '0.577*X+0.577*Y+0.577*Z'),
    ({'I': np.sqrt(1j)}, '(0.707+0.707i)*I'),
    ({'X': np.sqrt(-1j)}, '(0.707-0.707i)*X'),
))
def test_str(terms, string):
    linear_dict = cirq.LinearDict(terms)
    assert str(linear_dict).replace(' ', '') == string.replace(' ', '')


@pytest.mark.parametrize('terms', (
    ({}, {'X': 1}, {'X': 2, 'Y': 3}, {'X': 1.23456789e-12})
))
def test_repr(terms):
    original = cirq.LinearDict(terms)
    print(repr(original))
    recovered = eval(repr(original))
    assert original == recovered
    assert recovered == original


class FakePrinter:
    def __init__(self):
        self.buffer = ''

    def text(self, s: str) -> None:
        self.buffer += s

    def reset(self) -> None:
        self.buffer = ''


@pytest.mark.parametrize('terms', (
    {}, {'Y': 2}, {'X': 1, 'Y': -1j},
    {'X': np.sqrt(3)/3, 'Y': np.sqrt(3)/3, 'Z': np.sqrt(3)/3},
    {'I': np.sqrt(1j)}, {'X': np.sqrt(-1j)},
))
def test_repr_pretty(terms):
    printer = FakePrinter()
    linear_dict = cirq.LinearDict(terms)

    linear_dict._repr_pretty_(printer, False)
    assert printer.buffer.replace(' ', '') == str(linear_dict).replace(' ', '')

    printer.reset()
    linear_dict._repr_pretty_(printer, True)
    assert printer.buffer == 'LinearDict(...)'
