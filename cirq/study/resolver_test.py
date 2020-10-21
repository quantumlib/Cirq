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

"""Tests for parameter resolvers."""
import fractions

import numpy as np
import pytest
import sympy

import cirq


@pytest.mark.parametrize('val', [
    3.2,
    np.float32(3.2),
    int(1),
    np.int(3),
    np.int32(45),
    np.float64(6.3),
    np.int32(2),
    np.complex64(1j),
    np.complex128(2j),
    np.complex(1j),
    fractions.Fraction(3, 2),
])
def test_value_of_pass_through_types(val):
    _assert_consistent_resolution(val, val)


@pytest.mark.parametrize('val,resolved', [(sympy.pi, np.pi),
                                          (sympy.S.NegativeOne, -1),
                                          (sympy.S.Half, 0.5),
                                          (sympy.S.One, 1)])
def test_value_of_transformed_types(val, resolved):
    _assert_consistent_resolution(val, resolved)


@pytest.mark.parametrize('val,resolved', [(sympy.I, 1j)])
def test_value_of_substituted_types(val, resolved):
    _assert_consistent_resolution(val, resolved, True)


def _assert_consistent_resolution(v, resolved, subs_called=False):
    """Asserts that parameter resolution works consistently.

    The ParamResolver.value_of method can resolve any Sympy expression -
    subclasses of sympy.Basic. In the generic case, it calls `sympy.Basic.subs`
    to substitute symbols with values specified in a dict, which is known to be
    very slow. Instead value_of defines a pass-through shortcut for known
    numeric types. For a given value `v` it is asserted that value_of resolves
    it to `resolved`, with the exact type of `resolved`.`subs_called` indicates
    whether it is expected to have `subs` called or not during the resolution.

    Args:
        v: the value to resolve
        resolved: the expected resolution result
        subs_called: if True, it is expected that the slow subs method is called
    Raises:
        AssertionError in case resolution assertion fail.
    """

    class SubsAwareSymbol(sympy.Symbol):
        """A Symbol that registers a call to its `subs` method."""

        def __init__(self, sym: str):
            self.called = False
            self.symbol = sympy.Symbol(sym)

        # note: super().subs() doesn't resolve based on the param_dict properly
        # for some reason, that's why a delegate (self.symbol) is used instead
        def subs(self, *args, **kwargs):
            self.called = True
            return self.symbol.subs(*args, **kwargs)

    r = cirq.ParamResolver({'a': v})

    # symbol based resolution
    s = SubsAwareSymbol('a')
    assert r.value_of(s) == resolved, (f"expected {resolved}, "
                                       f"got {r.value_of(s)}")
    assert subs_called == s.called, (
        f"For pass-through type "
        f"{type(v)} sympy.subs shouldn't have been called.")
    assert isinstance(r.value_of(s),
                      type(resolved)), (f"expected {type(resolved)} "
                                        f"got {type(r.value_of(s))}")

    # string based resolution (which in turn uses symbol based resolution)
    assert r.value_of('a') == resolved, (f"expected {resolved}, "
                                         f"got {r.value_of('a')}")
    assert isinstance(r.value_of('a'),
                      type(resolved)), (f"expected {type(resolved)} "
                                        f"got {type(r.value_of('a'))}")

    # value based resolution
    assert r.value_of(v) == resolved, (f"expected {resolved}, "
                                       f"got {r.value_of(v)}")
    assert isinstance(r.value_of(v),
                      type(resolved)), (f"expected {type(resolved)} "
                                        f"got {type(r.value_of(v))}")


def test_value_of_strings():
    assert cirq.ParamResolver().value_of('x') == sympy.Symbol('x')


def test_value_of_calculations():
    assert not bool(cirq.ParamResolver())

    r = cirq.ParamResolver({'a': 0.5, 'b': 0.1, 'c': 1 + 1j})
    assert bool(r)

    assert r.value_of(2 * sympy.pi) == 2 * np.pi
    assert r.value_of(4**sympy.Symbol('a') + sympy.Symbol('b') * 10) == 3
    assert r.value_of(sympy.I * sympy.pi) == np.pi * 1j
    assert r.value_of(sympy.Symbol('a') * 3) == 1.5
    assert r.value_of(sympy.Symbol('b') / 0.1 - sympy.Symbol('a')) == 0.5


def test_param_dict():
    r = cirq.ParamResolver({'a': 0.5, 'b': 0.1})
    r2 = cirq.ParamResolver(r)
    assert r2 is r
    assert r.param_dict == {'a': 0.5, 'b': 0.1}


def test_param_dict_iter():
    r = cirq.ParamResolver({'a': 0.5, 'b': 0.1})
    assert [key for key in r] == ['a', 'b']
    assert [r.value_of(key) for key in r] == [0.5, 0.1]
    assert list(r) == ['a', 'b']


def test_formulas_in_param_dict():
    """
    Param dicts are allowed to have str or sympy.Symbol as keys and
    floats or sympy.Symbol as values.  This should not be a common use case,
    but this tests makes sure something reasonable is returned when
    mixing these types and using formulas in ParamResolvers.

    Note that sympy orders expressions for deterministic resolution, so
    depending on the operands sent to sub(), the expression may not fully
    resolve if it needs to take several iterations of resolution.
    """
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    c = sympy.Symbol('c')
    e = sympy.Symbol('e')
    r = cirq.ParamResolver({a: b + 1, b: 2, b + c: 101, 'd': 2 * e})
    assert r.value_of('a') == 3
    assert r.value_of('b') == 2
    assert r.value_of(b + c) == 101
    assert r.value_of('c') == c
    assert r.value_of('d') == 2 * e


def test_equals():
    et = cirq.testing.EqualsTester()
    et.add_equality_group(cirq.ParamResolver(),
                          cirq.ParamResolver(None),
                          cirq.ParamResolver({}),
                          cirq.ParamResolver(cirq.ParamResolver({})))
    et.make_equality_group(lambda: cirq.ParamResolver({'a': 0.0}))
    et.add_equality_group(cirq.ParamResolver({'a': 0.1}))
    et.add_equality_group(cirq.ParamResolver({'a': 0.0, 'b': 0.1}))
    et.add_equality_group(cirq.ParamResolver({'a': 0.3, 'b': 0.1}))
    et.add_equality_group(cirq.ParamResolver({'b': 0.1}))
    et.add_equality_group(cirq.ParamResolver({'c': 0.1}))


def test_repr():
    cirq.testing.assert_equivalent_repr(cirq.ParamResolver())
    cirq.testing.assert_equivalent_repr(cirq.ParamResolver({'a': 2.0}))
    cirq.testing.assert_equivalent_repr(
        cirq.ParamResolver({'a': sympy.Symbol('a')}))
    cirq.testing.assert_equivalent_repr(
        cirq.ParamResolver({sympy.Symbol('a'): sympy.Symbol('b') + 1}))
