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


@pytest.mark.parametrize(
    'val',
    [
        None,
        3.2,
        np.float32(3.2),
        int(1),
        np.int32(45),
        np.float64(6.3),
        np.int32(2),
        np.complex64(1j),
        np.complex128(2j),
        complex(1j),
        fractions.Fraction(3, 2),
    ],
)
def test_value_of_pass_through_types(val):
    _assert_consistent_resolution(val, val)


@pytest.mark.parametrize(
    'val,resolved',
    [(sympy.pi, np.pi), (sympy.S.NegativeOne, -1), (sympy.S.Half, 0.5), (sympy.S.One, 1)],
)
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
    assert r.value_of(s) == resolved, f"expected {resolved}, got {r.value_of(s)}"
    assert (
        subs_called == s.called
    ), f"For pass-through type {type(v)} sympy.subs shouldn't have been called."
    assert isinstance(
        r.value_of(s), type(resolved)
    ), f"expected {type(resolved)} got {type(r.value_of(s))}"

    # string based resolution (which in turn uses symbol based resolution)
    assert r.value_of('a') == resolved, f"expected {resolved}, got {r.value_of('a')}"
    assert isinstance(
        r.value_of('a'), type(resolved)
    ), f"expected {type(resolved)} got {type(r.value_of('a'))}"

    # value based resolution
    assert r.value_of(v) == resolved, f"expected {resolved}, got {r.value_of(v)}"
    assert isinstance(
        r.value_of(v), type(resolved)
    ), f"expected {type(resolved)} got {type(r.value_of(v))}"


def test_value_of_strings():
    assert cirq.ParamResolver().value_of('x') == sympy.Symbol('x')


def test_value_of_calculations():
    assert not bool(cirq.ParamResolver())

    r = cirq.ParamResolver({'a': 0.5, 'b': 0.1, 'c': 1 + 1j})
    assert bool(r)

    assert r.value_of(2 * sympy.pi) == 2 * np.pi
    assert r.value_of(4 ** sympy.Symbol('a') + sympy.Symbol('b') * 10) == 3
    assert r.value_of(sympy.I * sympy.pi) == np.pi * 1j
    assert r.value_of(sympy.Symbol('a') * 3) == 1.5
    assert r.value_of(sympy.Symbol('b') / 0.1 - sympy.Symbol('a')) == 0.5


def test_resolve_integer_division():
    r = cirq.ParamResolver({'a': 1, 'b': 2})
    resolved = r.value_of(sympy.Symbol('a') / sympy.Symbol('b'))
    assert resolved == 0.5


def test_resolve_symbol_division():
    B = sympy.Symbol('B')
    r = cirq.ParamResolver({'a': 1, 'b': B})
    resolved = r.value_of(sympy.Symbol('a') / sympy.Symbol('b'))
    assert resolved == sympy.core.power.Pow(B, -1)


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
    """Tests that formula keys are rejected in a `param_dict`."""
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    c = sympy.Symbol('c')
    e = sympy.Symbol('e')
    with pytest.raises(TypeError, match='formula'):
        _ = cirq.ParamResolver({a: b + 1, b: 2, b + c: 101, 'd': 2 * e})


def test_recursive_evaluation():
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    c = sympy.Symbol('c')
    d = sympy.Symbol('d')
    e = sympy.Symbol('e')
    r = cirq.ParamResolver({a: a, b: e + 2, c: b + d, d: a + 3, e: 0})

    # sympy.Basic.subs evaluates in alphabetical order.
    assert c.subs(r.param_dict) == b + a + 3

    assert r.value_of(a) == a
    assert sympy.Eq(r.value_of(b), 2)
    assert sympy.Eq(r.value_of(c), a + 5)
    assert sympy.Eq(r.value_of(d), a + 3)
    assert sympy.Eq(r.value_of(e), 0)


def test_unbound_recursion_halted():
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    c = sympy.Symbol('c')

    # Non-recursive resolution ignores loops
    r = cirq.ParamResolver({a: b, b: a})
    assert r.value_of(a, recursive=False) == b
    assert r.value_of(r.value_of(a, recursive=False), recursive=False) == a

    # Self-definition is OK (this is a terminal symbol)
    r = cirq.ParamResolver({a: a})
    assert r.value_of(a) == a

    r = cirq.ParamResolver({a: a + 1})
    with pytest.raises(RecursionError):
        _ = r.value_of(a)

    r = cirq.ParamResolver({a: b, b: a})
    with pytest.raises(RecursionError):
        _ = r.value_of(a)

    r = cirq.ParamResolver({a: b, b: c, c: b})
    with pytest.raises(RecursionError):
        _ = r.value_of(a)

    r = cirq.ParamResolver({a: b + c, b: 1, c: a})
    with pytest.raises(RecursionError):
        _ = r.value_of(a)


def test_resolve_unknown_type():
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    r = cirq.ParamResolver({a: b})
    assert r.value_of(cirq.X) == cirq.X


def test_custom_resolved_value():
    class Foo:
        def _resolved_value_(self):
            return self

    class Bar:
        def _resolved_value_(self):
            return NotImplemented

    class Baz:
        def _resolved_value_(self):
            return 'Baz'

    foo = Foo()
    bar = Bar()
    baz = Baz()

    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    c = sympy.Symbol('c')
    r = cirq.ParamResolver({a: foo, b: bar, c: baz})
    assert r.value_of(a) is foo
    assert r.value_of(b) is b
    assert r.value_of(c) == 'Baz'


def test_compose():
    """Tests that cirq.resolve_parameters on a ParamResolver composes."""
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    c = sympy.Symbol('c')
    d = sympy.Symbol('d')
    r1 = cirq.ParamResolver({a: b})
    r2 = cirq.ParamResolver({b: c + d})
    r3 = cirq.ParamResolver({c: 12})

    r12 = cirq.resolve_parameters(r1, r2)
    assert r12.value_of('a') == c + d

    r23 = cirq.resolve_parameters(r2, r3)
    assert sympy.Eq(r23.value_of('b'), 12 + d)

    r123 = cirq.resolve_parameters(r12, r3)
    assert sympy.Eq(r123.value_of('a'), 12 + d)

    r13 = cirq.resolve_parameters(r1, r3)
    assert r13.value_of('a') == b


@pytest.mark.parametrize(
    'p1, p2, p3',
    [
        ({'a': 1}, {}, {}),
        ({}, {'a': 1}, {}),
        ({}, {}, {'a': 1}),
        ({'a': 'b'}, {}, {'b': 'c'}),
        ({'a': 'b'}, {'c': 'd'}, {'b': 'c'}),
        ({'a': 'b'}, {'c': 'a'}, {'b': 'd'}),
        ({'a': 'b'}, {'c': 'd', 'd': 1}, {'d': 2}),
        ({'a': 'b'}, {'c': 'd', 'd': 'a'}, {'b': 2}),
    ],
)
@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_compose_associative(p1, p2, p3, resolve_fn):
    r1, r2, r3 = [
        cirq.ParamResolver(
            {sympy.Symbol(k): (sympy.Symbol(v) if isinstance(v, str) else v) for k, v in pd.items()}
        )
        for pd in [p1, p2, p3]
    ]
    assert sympy.Eq(
        resolve_fn(r1, resolve_fn(r2, r3)).param_dict, resolve_fn(resolve_fn(r1, r2), r3).param_dict
    )


def test_equals():
    et = cirq.testing.EqualsTester()
    et.add_equality_group(
        cirq.ParamResolver(),
        cirq.ParamResolver(None),
        cirq.ParamResolver({}),
        cirq.ParamResolver(cirq.ParamResolver({})),
    )
    et.make_equality_group(lambda: cirq.ParamResolver({'a': 0.0}))
    et.add_equality_group(cirq.ParamResolver({'a': 0.1}))
    et.add_equality_group(cirq.ParamResolver({'a': 0.0, 'b': 0.1}))
    et.add_equality_group(cirq.ParamResolver({'a': 0.3, 'b': 0.1}))
    et.add_equality_group(cirq.ParamResolver({'b': 0.1}))
    et.add_equality_group(cirq.ParamResolver({'c': 0.1}))


def test_repr():
    cirq.testing.assert_equivalent_repr(cirq.ParamResolver())
    cirq.testing.assert_equivalent_repr(cirq.ParamResolver({'a': 2.0}))
    cirq.testing.assert_equivalent_repr(cirq.ParamResolver({'a': sympy.Symbol('a')}))
    cirq.testing.assert_equivalent_repr(
        cirq.ParamResolver({sympy.Symbol('a'): sympy.Symbol('b') + 1})
    )
