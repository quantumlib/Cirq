# Copyright 2019 The Cirq Developers
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

from cirq.interop.quirk.cells.parse import parse_complex, parse_formula, parse_matrix


def test_parse_matrix():
    s = np.sqrt(0.5)
    np.testing.assert_allclose(
        parse_matrix('{{√½,√½},{-√½,√½}}'), np.array([[s, s], [-s, s]]), atol=1e-8
    )
    np.testing.assert_allclose(
        parse_matrix('{{√½,√½i},{√½i,√½}}'), np.array([[s, s * 1j], [s * 1j, s]]), atol=1e-8
    )
    np.testing.assert_allclose(
        parse_matrix('{{1,-i},{i,1+i}}'), np.array([[1, -1j], [1j, 1 + 1j]]), atol=1e-8
    )


def test_parse_matrix_failures():
    with pytest.raises(ValueError, match='Not surrounded by {{}}'):
        _ = parse_matrix('1')
    with pytest.raises(ValueError, match='Not surrounded by {{}}'):
        _ = parse_matrix('{{1}')
    with pytest.raises(ValueError, match='Not surrounded by {{}}'):
        _ = parse_matrix('{1}}')
    with pytest.raises(ValueError, match='Not surrounded by {{}}'):
        _ = parse_matrix('1}}')
    with pytest.raises(ValueError, match='Failed to parse complex'):
        _ = parse_matrix('{{x}}')


def test_parse_real_formula():
    t = sympy.Symbol('t')
    assert parse_formula('1/2') == 0.5
    assert parse_formula('t*t + ln(t)') == t * t + sympy.ln(t)
    assert parse_formula('cos(pi*t)') == sympy.cos(sympy.pi * t)
    assert parse_formula('5t') == 5.0 * t
    np.testing.assert_allclose(parse_formula('cos(pi)'), -1, atol=1e-8)
    assert type(parse_formula('cos(pi)')) is float

    with pytest.raises(ValueError, match='real result'):
        _ = parse_formula('i')


def test_parse_formula_failures():
    with pytest.raises(TypeError, match='formula must be a string'):
        _ = parse_formula(2)

    with pytest.raises(TypeError, match='formula must be a string'):
        _ = parse_formula([])

    with pytest.raises(ValueError, match='Unrecognized token'):
        _ = parse_formula('5*__**DSA **)SADD')

    with pytest.raises(ValueError, match='Unrecognized token'):
        _ = parse_formula('5*x')


def test_parse_complex():
    assert parse_complex('0') == 0
    assert parse_complex('1') == 1
    assert parse_complex('i') == 1j
    assert parse_complex('2i') == 2j
    assert parse_complex('-i') == -1j
    assert parse_complex('+i') == 1j
    assert parse_complex('1 + i - i') == 1
    assert parse_complex('1 + 2i - 3 i') == 1 - 1j
    np.testing.assert_allclose(parse_complex('exp 2'), np.e**2, atol=1e-8)


def test_parse_complex_raw_cases_from_quirk():
    assert parse_complex("0") == 0
    assert parse_complex("1") == 1
    assert parse_complex("-1") == -1
    assert parse_complex("i") == 1j
    assert parse_complex("-i") == -1j
    assert parse_complex("2") == 2
    assert parse_complex("2i") == 2j
    assert parse_complex("-2i") == -2j

    assert parse_complex("3-2i") == 3 - 2j
    assert parse_complex("1-i") == 1 - 1j
    assert parse_complex("1+i") == 1 + 1j
    assert parse_complex("-5+2i") == -5 + 2j
    assert parse_complex("-5-2i") == -5 - 2j

    assert parse_complex("3/2i") == 1.5j

    assert parse_complex("\u221a2-\u2153i") == np.sqrt(2) - 1j / 3

    assert parse_complex("1e-10") == 0.0000000001
    assert parse_complex("1e+10") == 10000000000
    assert parse_complex("2.5e-10") == 0.00000000025
    assert parse_complex("2.5E-10") == 0.00000000025
    assert parse_complex("2.5e+10") == 25000000000
    assert parse_complex("2.e+10") == 20000000000
    np.testing.assert_allclose(parse_complex("e"), np.e)
    np.testing.assert_allclose(parse_complex("pi e"), np.pi * np.e)
    np.testing.assert_allclose(parse_complex("pi e 2"), np.pi * np.e * 2)
    np.testing.assert_allclose(parse_complex("2       pi"), 2 * np.pi)


def test_parse_complex_expression_cases_from_quirk():
    np.testing.assert_allclose(parse_complex("1/3"), 1 / 3)
    np.testing.assert_allclose(parse_complex("2/3/5"), (2 / 3) / 5)
    np.testing.assert_allclose(parse_complex("2/3/5*7/13"), ((((2 / 3) / 5)) * 7) / 13)
    np.testing.assert_allclose(parse_complex("2-3-5"), -6)
    np.testing.assert_allclose(parse_complex("1/3+2i"), 1 / 3 + 2j)
    np.testing.assert_allclose(parse_complex("(1/3)+2i"), 1 / 3 + 2j)
    np.testing.assert_allclose(parse_complex("1/(3+2i)"), 1 / (3 + 2j), atol=1e-8)
    np.testing.assert_allclose(parse_complex("1/sqrt(3+2i)"), 1 / ((3 + 2j) ** 0.5), atol=1e-8)

    np.testing.assert_allclose(parse_complex("i^i"), 0.20787957635076193, atol=1e-8)
    np.testing.assert_allclose(parse_complex("√i"), np.sqrt(1j))
    np.testing.assert_allclose(parse_complex("√4i"), 2j)
    np.testing.assert_allclose(parse_complex("sqrt4i"), 2j)
    np.testing.assert_allclose(parse_complex("sqrt√4i"), np.sqrt(2) * 1j)
    np.testing.assert_allclose(parse_complex("sqrt√4-i"), np.sqrt(2) - 1j)
    np.testing.assert_allclose(parse_complex("----------1"), 1)
    np.testing.assert_allclose(parse_complex("---------1"), -1)
    np.testing.assert_allclose(parse_complex("---+--+--1"), -1)
    np.testing.assert_allclose(parse_complex("0---+--+--1"), -1)
    np.testing.assert_allclose(parse_complex("0---+--+--1*"), -1)
    np.testing.assert_allclose(parse_complex("2+3^"), 5)
    np.testing.assert_allclose(
        parse_complex("cos(45) + i sin(45)"), np.sqrt(0.5) + 1j * np.sqrt(0.5), atol=1e-8
    )
    np.testing.assert_allclose(
        parse_complex("cos(45) + i (sin 45)"), np.sqrt(0.5) + 1j * np.sqrt(0.5), atol=1e-8
    )
    np.testing.assert_allclose(parse_complex("e^(pi i)"), -1, atol=1e-8)
    np.testing.assert_allclose(parse_complex("exp(ln(2))"), 2, atol=1e-8)
    np.testing.assert_allclose(parse_complex("sin(arcsin(0.5))"), 0.5, atol=1e-8)
    np.testing.assert_allclose(parse_complex("cos(arccos(0.5))"), 0.5, atol=1e-8)
    np.testing.assert_allclose(parse_complex("sin(asin(0.5))"), 0.5, atol=1e-8)
    np.testing.assert_allclose(parse_complex("cos(acos(0.5))"), 0.5, atol=1e-8)


def test_parse_complex_expression_failures():
    with pytest.raises(ValueError, match='Incomplete expression'):
        _ = parse_formula('(')
    with pytest.raises(ValueError, match=r"unmatched '\)'"):
        _ = parse_formula(')')
    with pytest.raises(ValueError, match='binary op in bad spot'):
        _ = parse_formula('5+(/)')
    with pytest.raises(ValueError, match='operated on nothing'):
        _ = parse_formula('(5+)')
    with pytest.raises(ValueError, match='operated on nothing'):
        _ = parse_formula('(5/)')
    with pytest.raises(ValueError, match='binary op in bad spot'):
        _ = parse_formula('5-/2')
    with pytest.raises(ValueError, match='binary op in bad spot'):
        _ = parse_formula('/2')

    # Quirk silently ignores partial operations that show up while typing.
    assert parse_formula('2/ ') == 2
    assert parse_formula('2* ') == 2
    assert parse_formula('2+ ') == 2
    assert parse_formula('2- ') == 2
    assert parse_formula('2^ ') == 2
