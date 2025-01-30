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
import sympy

import cirq


def test_empty_init():
    v = cirq.LinearDict()
    assert v == cirq.LinearDict({})
    assert not v


sym = sympy.Symbol('sym')
expr = sym * -(2 + 3j)
symval = expr.subs({'sym': 5})
symvalresolved = -10 - 15j


@pytest.mark.parametrize(
    'keys, coefficient, terms_expected',
    (
        ((), 10, {}),
        (('X',), 2, {'X': 2}),
        (('a', 'b', 'c', 'd'), 0.5, {'a': 0.5, 'b': 0.5, 'c': 0.5, 'd': 0.5}),
        (('b', 'c', 'd', 'e'), -2j, {'b': -2j, 'c': -2j, 'd': -2j, 'e': -2j}),
        (('b', 'c'), sym, {'b': sym, 'c': sym}),
        (('b', 'c'), expr, {'b': expr, 'c': expr}),
        (('b', 'c'), symval, {'b': symvalresolved, 'c': symvalresolved}),
    ),
)
def test_fromkeys(keys, coefficient, terms_expected):
    actual = cirq.LinearDict.fromkeys(keys, coefficient)
    expected = cirq.LinearDict(terms_expected)
    assert actual == expected
    assert expected == actual


@pytest.mark.parametrize(
    'terms, valid_vectors, invalid_vectors',
    (({'X': 2}, ('X'), ('A', 'B')), ({'X': 2, 'Y': -2}, ('X', 'Y', 'Z'), ('A', 'B'))),
)
def test_invalid_vectors_are_rejected(terms, valid_vectors, invalid_vectors):
    linear_dict = cirq.LinearDict(terms, validator=lambda v: v in valid_vectors)

    with pytest.raises(ValueError):
        linear_dict += cirq.LinearDict.fromkeys(invalid_vectors, 1)
    assert linear_dict == cirq.LinearDict(terms)

    for vector in invalid_vectors:
        with pytest.raises(ValueError):
            linear_dict[vector] += 1
    assert linear_dict == cirq.LinearDict(terms)

    with pytest.raises(ValueError):
        linear_dict.update(cirq.LinearDict.fromkeys(invalid_vectors, 1))
    assert linear_dict == cirq.LinearDict(terms)


@pytest.mark.parametrize(
    'terms, valid_vectors', (({'X': 2}, ('X')), ({'X': 2, 'Y': -2}, ('X', 'Y', 'Z')))
)
def test_valid_vectors_are_accepted(terms, valid_vectors):
    linear_dict = cirq.LinearDict(terms, validator=lambda v: v in valid_vectors)

    original_dict = linear_dict.copy()
    delta_dict = cirq.LinearDict.fromkeys(valid_vectors, 1)

    linear_dict += cirq.LinearDict.fromkeys(valid_vectors, 1)
    assert linear_dict == original_dict + delta_dict

    for vector in valid_vectors:
        linear_dict[vector] += 1
    assert linear_dict == original_dict + 2 * delta_dict

    linear_dict.update(cirq.LinearDict.fromkeys(valid_vectors, 1))
    assert linear_dict == delta_dict


@pytest.mark.parametrize(
    'terms, atol, terms_expected',
    (
        ({'X': 1, 'Y': 2, 'Z': 3}, 2, {'Z': 3}),
        ({'X': 0.1, 'Y': 1, 'Z': 10}, 1e-3, {'X': 0.1, 'Y': 1, 'Z': 10}),
        ({'X': 1e-10, 'H': 1e-11}, 1e-9, {}),
        ({}, 1, {}),
    ),
)
def test_clean(terms, atol, terms_expected):
    linear_dict = cirq.LinearDict(terms)
    linear_dict.clean(atol=atol)
    expected = cirq.LinearDict(terms_expected)
    assert linear_dict == expected
    assert expected == linear_dict


@pytest.mark.parametrize('terms', ({'X': 1j / 2}, {'X': 1, 'Y': 2, 'Z': 3}, {}))
def test_copy(terms):
    original = cirq.LinearDict(terms)
    copy = original.copy()
    assert type(copy) == cirq.LinearDict
    assert copy == original
    assert original == copy
    original['a'] = 1
    assert copy != original
    assert original != copy
    assert 'a' in original
    assert 'a' not in copy


@pytest.mark.parametrize(
    'terms, expected_keys',
    (({}, ()), ({'X': 0}, ()), ({'X': 0.1}, ('X',)), ({'X': -1, 'Y': 0, 'Z': 1}, ('X', 'Z'))),
)
def test_keys(terms, expected_keys):
    linear_dict = cirq.LinearDict(terms)
    assert tuple(sorted(linear_dict.keys())) == expected_keys


@pytest.mark.parametrize(
    'terms, expected_values',
    (({}, ()), ({'X': 0}, ()), ({'X': 0.1}, (0.1,)), ({'X': -1, 'Y': 0, 'Z': 1}, (-1, 1))),
)
def test_values(terms, expected_values):
    linear_dict = cirq.LinearDict(terms)
    assert tuple(sorted(linear_dict.values())) == expected_values


@pytest.mark.parametrize(
    'terms, expected_items',
    (
        ({}, ()),
        ({'X': 0}, ()),
        ({'X': 0.1}, (('X', 0.1),)),
        ({'X': -1, 'Y': 0, 'Z': 1}, (('X', -1), ('Z', 1))),
    ),
)
def test_items(terms, expected_items):
    linear_dict = cirq.LinearDict(terms)
    assert tuple(sorted(linear_dict.items())) == expected_items


@pytest.mark.parametrize(
    'terms_1, terms_2, terms_expected',
    (
        ({}, {}, {}),
        ({}, {'X': 0.1}, {'X': 0.1}),
        ({'X': 1}, {'Y': 2}, {'X': 1, 'Y': 2}),
        ({'X': 1}, {'X': 4}, {'X': 4}),
        ({'X': 1, 'Y': 2}, {'Y': -2}, {'X': 1, 'Y': -2}),
    ),
)
def test_update(terms_1, terms_2, terms_expected):
    linear_dict_1 = cirq.LinearDict(terms_1)
    linear_dict_2 = cirq.LinearDict(terms_2)
    linear_dict_1.update(linear_dict_2)
    expected = cirq.LinearDict(terms_expected)
    assert linear_dict_1 == expected
    assert expected == linear_dict_1


@pytest.mark.parametrize(
    'terms, vector, expected_coefficient',
    (({}, '', 0), ({}, 'X', 0), ({'X': 0}, 'X', 0), ({'X': -1j}, 'X', -1j), ({'X': 1j}, 'Y', 0)),
)
def test_get(terms, vector, expected_coefficient):
    linear_dict = cirq.LinearDict(terms)
    actual_coefficient = linear_dict.get(vector)
    assert actual_coefficient == expected_coefficient


@pytest.mark.parametrize(
    'terms, vector, expected',
    (
        ({}, 'X', False),
        ({'X': 0}, 'X', False),
        ({'X': 0.1}, 'X', True),
        ({'X': 1, 'Y': -1}, 'Y', True),
    ),
)
def test_contains(terms, vector, expected):
    linear_dict = cirq.LinearDict(terms)
    actual = vector in linear_dict
    assert actual == expected


@pytest.mark.parametrize(
    'terms, vector, expected_coefficient',
    (
        ({}, 'X', 0),
        ({'X': 1}, 'X', 1),
        ({'Y': 1}, 'X', 0),
        ({'X': 2, 'Y': 3}, 'X', 2),
        ({'X': 1, 'Y': 2}, 'Z', 0),
    ),
)
def test_getitem(terms, vector, expected_coefficient):
    linear_dict = cirq.LinearDict(terms)
    actual_coefficient = linear_dict[vector]
    assert actual_coefficient == expected_coefficient


@pytest.mark.parametrize(
    'terms, vector, coefficient, terms_expected',
    (
        ({}, 'X', 0, {}),
        ({}, 'X', 1, {'X': 1}),
        ({'X': 1}, 'X', 2, {'X': 2}),
        ({'X': 1, 'Y': 3}, 'X', 2, {'X': 2, 'Y': 3}),
        ({'X': 1, 'Y': 2}, 'X', 0, {'Y': 2}),
    ),
)
def test_setitem(terms, vector, coefficient, terms_expected):
    linear_dict = cirq.LinearDict(terms)
    linear_dict[vector] = coefficient
    expected = cirq.LinearDict(terms_expected)
    assert linear_dict == expected
    assert expected == linear_dict


@pytest.mark.parametrize(
    'terms, vector, terms_expected',
    (
        ({}, 'X', {}),
        ({'X': 1}, 'X', {}),
        ({'X': 1}, 'Y', {'X': 1}),
        ({'X': 1, 'Y': 3}, 'X', {'Y': 3}),
    ),
)
def test_delitem(terms, vector, terms_expected):
    linear_dict = cirq.LinearDict(terms)
    del linear_dict[vector]
    expected = cirq.LinearDict(terms_expected)
    assert linear_dict == expected
    assert expected == linear_dict


def test_addition_in_iteration():
    linear_dict = cirq.LinearDict({'a': 2, 'b': 1, 'c': 0, 'd': -1, 'e': -2})
    for v in linear_dict:
        linear_dict[v] += 1
    assert linear_dict == cirq.LinearDict({'a': 3, 'b': 2, 'c': 0, 'd': 0, 'e': -1})
    assert linear_dict == cirq.LinearDict({'a': 3, 'b': 2, 'e': -1})


def test_multiplication_in_iteration():
    linear_dict = cirq.LinearDict({'u': 2, 'v': 1, 'w': -1})
    for v, c in linear_dict.items():
        if c > 0:
            linear_dict[v] *= 0
    assert linear_dict == cirq.LinearDict({'u': 0, 'v': 0, 'w': -1})
    assert linear_dict == cirq.LinearDict({'w': -1})


@pytest.mark.parametrize(
    'terms, expected_length',
    (({}, 0), ({'X': 0}, 0), ({'X': 0.1}, 1), ({'X': 1, 'Y': -2j}, 2), ({'X': 0, 'Y': 1}, 1)),
)
def test_len(terms, expected_length):
    linear_dict = cirq.LinearDict(terms)
    assert len(linear_dict) == expected_length


@pytest.mark.parametrize(
    'terms_1, terms_2, terms_expected',
    (
        ({}, {}, {}),
        ({}, {'X': 0.1}, {'X': 0.1}),
        ({'X': 1}, {'Y': 2}, {'X': 1, 'Y': 2}),
        ({'X': 1}, {'X': 1}, {'X': 2}),
        ({'X': 1, 'Y': 2}, {'Y': -2}, {'X': 1}),
        ({'X': 1}, {'X': sym}, {'X': sym + 1}),
        ({'X': 1}, {'X': expr}, {'X': expr + 1}),
        ({'X': 1}, {'X': symval}, {'X': symvalresolved + 1}),
    ),
)
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


@pytest.mark.parametrize(
    'terms_1, terms_2, terms_expected',
    (
        ({}, {}, {}),
        ({'a': 2}, {'a': 2}, {}),
        ({'a': 3}, {'a': 2}, {'a': 1}),
        ({'X': 1}, {'Y': 2}, {'X': 1, 'Y': -2}),
        ({'X': 1}, {'X': 1}, {}),
        ({'X': 1, 'Y': 2}, {'Y': 2}, {'X': 1}),
        ({'X': 1, 'Y': 2}, {'Y': 3}, {'X': 1, 'Y': -1}),
        ({'X': 1}, {'X': sym}, {'X': 1 - sym}),
        ({'X': 1}, {'X': expr}, {'X': 1 - expr}),
        ({'X': 1}, {'X': symval}, {'X': 1 - symvalresolved}),
    ),
)
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


@pytest.mark.parametrize(
    'terms, terms_expected',
    (
        ({}, {}),
        ({'key': 1}, {'key': -1}),
        ({'1': 10, '2': -20}, {'1': -10, '2': 20}),
        ({'key': sym}, {'key': -sym}),
        ({'key': expr}, {'key': -expr}),
        ({'key': symval}, {'key': -symvalresolved}),
    ),
)
def test_vector_negation(terms, terms_expected):
    linear_dict = cirq.LinearDict(terms)
    actual = -linear_dict
    expected = cirq.LinearDict(terms_expected)
    assert actual == expected
    assert expected == actual


@pytest.mark.parametrize(
    'scalar, terms, terms_expected',
    (
        (2, {}, {}),
        (2, {'X': 1, 'Y': -2}, {'X': 2, 'Y': -4}),
        (0, {'abc': 10, 'def': 20}, {}),
        (1j, {'X': 4j}, {'X': -4}),
        (-1, {'a': 10, 'b': -20}, {'a': -10, 'b': 20}),
        (2, {'X': sym}, {'X': 2 * sym}),
        (2, {'X': expr}, {'X': 2 * expr}),
        (2, {'X': symval}, {'X': 2 * symvalresolved}),
        (sym, {'X': 2}, {'X': 2 * sym}),
        (expr, {'X': 2}, {'X': 2 * expr}),
        (symval, {'X': 2}, {'X': 2 * symvalresolved}),
    ),
)
def test_scalar_multiplication(scalar, terms, terms_expected):
    linear_dict = cirq.LinearDict(terms)
    actual_1 = scalar * linear_dict
    actual_2 = linear_dict * scalar
    expected = cirq.LinearDict(terms_expected)
    assert actual_1 == expected
    assert actual_2 == expected
    assert actual_1 == actual_2


@pytest.mark.parametrize(
    'scalar, terms, terms_expected',
    (
        (2, {}, {}),
        (2, {'X': 6, 'Y': -2}, {'X': 3, 'Y': -1}),
        (1j, {'X': 1, 'Y': 1j}, {'X': -1j, 'Y': 1}),
        (-1, {'a': 10, 'b': -20}, {'a': -10, 'b': 20}),
        (2, {'X': sym}, {'X': 0.5 * sym}),
        (2, {'X': expr}, {'X': 0.5 * expr}),
        (2, {'X': symval}, {'X': 0.5 * symvalresolved}),
        (sym, {'X': 2}, {'X': 2 / sym}),
        (expr, {'X': 2}, {'X': 2 / expr}),
        (symval, {'X': 2}, {'X': 2 / symvalresolved}),
    ),
)
def test_scalar_division(scalar, terms, terms_expected):
    linear_dict = cirq.LinearDict(terms)
    actual = linear_dict / scalar
    expected = cirq.LinearDict(terms_expected)
    assert actual == expected
    assert expected == actual


@pytest.mark.parametrize(
    'expression, expected',
    (
        (
            (cirq.LinearDict({'X': 10}) + cirq.LinearDict({'X': 10, 'Y': -40})) / 20,
            cirq.LinearDict({'X': 1, 'Y': -2}),
        ),
        (cirq.LinearDict({'a': -2}) + 2 * cirq.LinearDict({'a': 1}), cirq.LinearDict({})),
        (cirq.LinearDict({'b': 2}) - 2 * cirq.LinearDict({'b': 1}), cirq.LinearDict({})),
    ),
)
def test_expressions(expression, expected):
    assert expression == expected
    assert not expression != expected
    assert cirq.approx_eq(expression, expected)


@pytest.mark.parametrize(
    'terms, bool_value', (({}, False), ({'X': 0}, False), ({'Z': 1e-12}, True), ({'Y': 1}, True))
)
def test_bool(terms, bool_value):
    linear_dict = cirq.LinearDict(terms)
    assert bool(linear_dict) == bool_value


@pytest.mark.parametrize(
    'terms_1, terms_2',
    (
        ({}, {}),
        ({}, {'X': 0}),
        ({'X': 0.0}, {'Y': 0.0}),
        ({'a': 1}, {'a': 1, 'b': 0}),
        ({'X': sym}, {'X': sym}),
    ),
)
def test_equal(terms_1, terms_2):
    linear_dict_1 = cirq.LinearDict(terms_1)
    linear_dict_2 = cirq.LinearDict(terms_2)
    assert linear_dict_1 == linear_dict_2
    assert linear_dict_2 == linear_dict_1
    assert not linear_dict_1 != linear_dict_2
    assert not linear_dict_2 != linear_dict_1


@pytest.mark.parametrize(
    'terms_1, terms_2',
    (
        ({}, {'a': 1}),
        ({'X': 1e-12}, {'X': 0}),
        ({'X': 0.0}, {'Y': 0.1}),
        ({'X': 1}, {'X': 1, 'Z': 1e-12}),
        ({'X': sym + 0.1}, {'X': sym}),
    ),
)
def test_unequal(terms_1, terms_2):
    linear_dict_1 = cirq.LinearDict(terms_1)
    linear_dict_2 = cirq.LinearDict(terms_2)
    assert linear_dict_1 != linear_dict_2
    assert linear_dict_2 != linear_dict_1
    assert not linear_dict_1 == linear_dict_2
    assert not linear_dict_2 == linear_dict_1


@pytest.mark.parametrize(
    'terms_1, terms_2',
    (
        ({}, {'X': 1e-9}),
        ({'X': 1e-12}, {'X': 0}),
        ({'X': 5e-10}, {'Y': 2e-11}),
        ({'X': 1.000000001}, {'X': 1, 'Z': 0}),
        ({'X': sym + 0.000000001}, {'X': sym}),
    ),
)
def test_approximately_equal(terms_1, terms_2):
    linear_dict_1 = cirq.LinearDict(terms_1)
    linear_dict_2 = cirq.LinearDict(terms_2)
    assert cirq.approx_eq(linear_dict_1, linear_dict_2)
    assert cirq.approx_eq(linear_dict_2, linear_dict_1)


@pytest.mark.parametrize(
    'a, b',
    (
        (cirq.LinearDict({}), None),
        (cirq.LinearDict({'X': 0}), 0),
        (cirq.LinearDict({'Y': 1}), 1),
        (cirq.LinearDict({'Z': 1}), 1j),
        (cirq.LinearDict({'I': 1}), 'I'),
    ),
)
def test_incomparable(a, b):
    assert a.__eq__(b) is NotImplemented
    assert a.__ne__(b) is NotImplemented
    assert a._approx_eq_(b, atol=1e-9) is NotImplemented


@pytest.mark.parametrize(
    'terms, fmt, expected_string',
    (
        ({}, '{}', '0'),
        ({}, '{:.2f}', '0.00'),
        ({}, '{:.2e}', '0.00e+00'),
        ({'X': 2**-10}, '{:.2f}', '0.00'),
        ({'X': 1 / 100}, '{:.2e}', '1.00e-02*X'),
        ({'X': 1j * 2**-10}, '{:.2f}', '0.00'),
        ({'X': 1j * 2**-10}, '{:.3f}', '0.001j*X'),
        ({'X': 2j, 'Y': -3}, '{:.2f}', '2.00j*X-3.00*Y'),
        ({'X': -2j, 'Y': 3}, '{:.2f}', '-2.00j*X+3.00*Y'),
        ({'X': np.sqrt(1j)}, '{:.3f}', '(0.707+0.707j)*X'),
        ({'X': np.sqrt(-1j)}, '{:.3f}', '(0.707-0.707j)*X'),
        ({'X': -np.sqrt(-1j)}, '{:.3f}', '(-0.707+0.707j)*X'),
        ({'X': -np.sqrt(1j)}, '{:.3f}', '-(0.707+0.707j)*X'),
        ({'X': 1, 'Y': -1, 'Z': 1j}, '{:.5f}', '1.00000*X-1.00000*Y+1.00000j*Z'),
        ({'X': 2, 'Y': -0.0001}, '{:.4f}', '2.0000*X-0.0001*Y'),
        ({'X': 2, 'Y': -0.0001}, '{:.3f}', '2.000*X'),
        ({'X': 2, 'Y': -0.0001}, '{:.1e}', '2.0e+00*X-1.0e-04*Y'),
    ),
)
def test_format(terms, fmt, expected_string):
    linear_dict = cirq.LinearDict(terms)
    actual_string = fmt.format(linear_dict)
    assert actual_string.replace(' ', '') == expected_string.replace(' ', '')


@pytest.mark.parametrize(
    'terms',
    (
        (
            {},
            {'X': 1},
            {'X': 2, 'Y': 3},
            {'X': 1.23456789e-12},
            {'X': sym},
            ({'X': sym * 2}),
            {'X': symval},
        )
    ),
)
def test_repr(terms):
    original = cirq.LinearDict(terms)
    recovered = eval(repr(original))
    assert original == recovered
    assert recovered == original


@pytest.mark.parametrize(
    'terms, string',
    (
        ({}, '0.000'),
        ({'X': 1.5, 'Y': 1e-5}, '1.500*X'),
        ({'Y': 2}, '2.000*Y'),
        ({'X': 1, 'Y': -1j}, '1.000*X-1.000j*Y'),
        (
            {'X': np.sqrt(3) / 3, 'Y': np.sqrt(3) / 3, 'Z': np.sqrt(3) / 3},
            '0.577*X+0.577*Y+0.577*Z',
        ),
        ({'I': np.sqrt(1j)}, '(0.707+0.707j)*I'),
        ({'X': np.sqrt(-1j)}, '(0.707-0.707j)*X'),
        ({'X': -np.sqrt(-1j)}, '(-0.707+0.707j)*X'),
        ({'X': -np.sqrt(1j)}, '-(0.707+0.707j)*X'),
        ({'X': -2, 'Y': -3}, '-2.000*X-3.000*Y'),
        ({'X': -2j, 'Y': -3}, '-2.000j*X-3.000*Y'),
        ({'X': -2j, 'Y': -3j}, '-2.000j*X-3.000j*Y'),
        ({'X': sym}, 'sym*X'),
        ({'X': sym * 2}, '2.000*sym*X'),
        ({'X': expr}, '-sym*(2.000+3.000j)*X'),
        ({'X': symval}, '-(10.000+15.000j)*X'),
    ),
)
def test_str(terms, string):
    linear_dict = cirq.LinearDict(terms)
    assert str(linear_dict).replace(' ', '') == string.replace(' ', '')


class FakePrinter:
    def __init__(self):
        self.buffer = ''

    def text(self, s: str) -> None:
        self.buffer += s

    def reset(self) -> None:
        self.buffer = ''


@pytest.mark.parametrize(
    'terms',
    (
        {},
        {'Y': 2},
        {'X': 1, 'Y': -1j},
        {'X': np.sqrt(3) / 3, 'Y': np.sqrt(3) / 3, 'Z': np.sqrt(3) / 3},
        {'I': np.sqrt(1j)},
        {'X': np.sqrt(-1j)},
        {cirq.X: 1, cirq.H: -1},
    ),
)
def test_repr_pretty(terms):
    printer = FakePrinter()
    linear_dict = cirq.LinearDict(terms)

    linear_dict._repr_pretty_(printer, False)
    assert printer.buffer.replace(' ', '') == str(linear_dict).replace(' ', '')

    printer.reset()
    linear_dict._repr_pretty_(printer, True)
    assert printer.buffer == 'LinearDict(...)'


def test_json_fails_with_validator():
    with pytest.raises(ValueError, match='not json serializable'):
        _ = cirq.to_json(cirq.LinearDict({}, validator=lambda: True))


@pytest.mark.parametrize(
    'terms, names',
    (
        ({'X': sym}, {'sym'}),
        ({'X': sym * sympy.Symbol('a')}, {'sym', 'a'}),
        ({'X': expr}, {'sym'}),
        ({'X': sym, 'Y': sympy.Symbol('a')}, {'sym', 'a'}),
        ({'X': symval}, set()),
    ),
)
def test_parameter_names(terms, names):
    linear_dict = cirq.LinearDict(terms)
    assert cirq.parameter_names(linear_dict) == names


@pytest.mark.parametrize(
    'terms, expected',
    (
        ({'X': sym}, {'X': 2}),
        ({'X': sym * sympy.Symbol('a')}, {'X': 6}),
        ({'X': expr}, {'X': -4 - 6j}),
        ({'X': sym, 'Y': sympy.Symbol('a')}, {'X': 2, 'Y': 3}),
        ({'X': symval}, {'X': symvalresolved}),
    ),
)
def test_resolve_parameters(terms, expected):
    linear_dict = cirq.LinearDict(terms)
    expected_dict = cirq.LinearDict(expected)
    assert cirq.resolve_parameters(linear_dict, {'sym': 2, 'a': 3}) == expected_dict
