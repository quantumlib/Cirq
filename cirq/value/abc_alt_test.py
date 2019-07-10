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

import abc
import pytest

from cirq import ABCMetaImplementAnyOneOf, alternative


def test_regular_abstract():

    class RegularAbc(metaclass=ABCMetaImplementAnyOneOf):

        @abc.abstractmethod
        def my_method(self):
            """Docstring."""

    with pytest.raises(TypeError, match='abstract'):
        RegularAbc()


def test_single_alternative():

    class SingleAlternative(metaclass=ABCMetaImplementAnyOneOf):

        def _default_impl(self, arg, kw=99):
            return f'default({arg}, {kw}) ' + self.alt()

        @alternative(requires='alt', implementation=_default_impl)
        def my_method(self, arg, kw=99):
            """my_method doc."""

        @abc.abstractmethod
        def alt(self):
            pass

    class SingleAlternativeChild(SingleAlternative):

        def alt(self):
            return 'alt'

    class SingleAlternativeOverride(SingleAlternative):

        def my_method(self, arg, kw=99):
            """my_method override."""
            return 'override'

        def alt(self):
            """Unneeded alternative method."""

    class SingleAlternativeGrandchild(SingleAlternativeChild):

        def alt(self):
            return 'alt2'

    class SingleAlternativeGrandchildOverride(SingleAlternativeChild):

        def my_method(self, arg, kw=99):
            """my_method override."""
            return 'override2'

        def alt(self):
            """Unneeded alternative method."""

    with pytest.raises(TypeError, match='abstract'):
        SingleAlternative()
    assert SingleAlternativeChild().my_method(1) == 'default(1, 99) alt'
    assert SingleAlternativeChild().my_method(2, kw=3) == 'default(2, 3) alt'
    assert SingleAlternativeOverride().my_method(4, kw=5) == 'override'
    assert (SingleAlternativeGrandchild().my_method(
        6, kw=7) == 'default(6, 7) alt2')
    assert (SingleAlternativeGrandchildOverride().my_method(
        8, kw=9) == 'override2')


def test_doc_string():

    class SingleAlternative(metaclass=ABCMetaImplementAnyOneOf):

        def _default_impl(self, arg, kw=99):
            """Default implementation."""

        @alternative(requires='alt', implementation=_default_impl)
        def my_method(self, arg, kw=99):
            """my_method doc."""

        @abc.abstractmethod
        def alt(self):
            pass

    class SingleAlternativeChild(SingleAlternative):

        def alt(self):
            """Alternative method."""

    class SingleAlternativeOverride(SingleAlternative):

        def my_method(self, arg, kw=99):
            """my_method override."""

        def alt(self):
            """Unneeded alternative method."""

    assert SingleAlternative.my_method.__doc__ == 'my_method doc.'
    assert SingleAlternativeChild.my_method.__doc__ == 'my_method doc.'
    assert SingleAlternativeChild().my_method.__doc__ == 'my_method doc.'
    assert SingleAlternativeOverride.my_method.__doc__ == 'my_method override.'
    assert (
        SingleAlternativeOverride().my_method.__doc__ == 'my_method override.')


def test_bad_alternative():
    with pytest.raises(TypeError, match='not exist'):

        class _(metaclass=ABCMetaImplementAnyOneOf):

            @alternative(requires='missing_alt',
                         implementation=lambda self: None)
            def my_method(self, arg, kw=99):
                """my_method doc."""


def test_unrelated_attribute():
    # Test that the class is created without wrongly raising a TypeError
    class _(metaclass=ABCMetaImplementAnyOneOf):
        _none_attribute = None
        _false_attribute = False
        _true_attribute = True

        @alternative(requires='alt', implementation=lambda self: None)
        def my_method(self):
            """my_method doc."""

        def alt(self):
            """alt doc."""


def test_classcell_in_namespace():
    """Tests a historical issue where super() triggers python to add
    `__classcell__` to the namespace passed to the metaclass __new__.
    """

    # Test that the class is created without wrongly raising a TypeError
    class _(metaclass=ABCMetaImplementAnyOneOf):

        def other_method(self):
            # Triggers __classcell__ to be added to the class namespace
            super()  # coverage: ignore


def test_two_alternatives():

    class TwoAlternatives(metaclass=ABCMetaImplementAnyOneOf):

        def _default_impl1(self, arg, kw=99):
            return f'default1({arg}, {kw}) ' + self.alt1()

        def _default_impl2(self, arg, kw=99):
            return f'default2({arg}, {kw}) ' + self.alt2()

        @alternative(requires='alt1', implementation=_default_impl1)
        @alternative(requires='alt2', implementation=_default_impl2)
        def my_method(self, arg, kw=99):
            """Docstring."""

        @abc.abstractmethod
        def alt1(self):
            pass

        @abc.abstractmethod
        def alt2(self):
            pass

    class TwoAlternativesChild(TwoAlternatives):

        def alt1(self):
            return 'alt1'

        def alt2(self):
            raise RuntimeError  # coverage: ignore

    class TwoAlternativesOverride(TwoAlternatives):

        def my_method(self, arg, kw=99):
            return 'override'

        def alt1(self):
            raise RuntimeError  # coverage: ignore

        def alt2(self):
            raise RuntimeError  # coverage: ignore

    class TwoAlternativesForceSecond(TwoAlternatives):

        def _do_alt1_with_my_method(self):
            return 'reverse ' + self.my_method(0, kw=0)

        @alternative(requires='my_method',
                     implementation=_do_alt1_with_my_method)
        def alt1(self):
            """alt1 doc."""

        def alt2(self):
            return 'alt2'

    with pytest.raises(TypeError, match='abstract'):
        TwoAlternatives()
    assert TwoAlternativesChild().my_method(1) == 'default1(1, 99) alt1'
    assert TwoAlternativesChild().my_method(2, kw=3) == 'default1(2, 3) alt1'
    assert TwoAlternativesOverride().my_method(4, kw=5) == 'override'
    assert (TwoAlternativesForceSecond().my_method(
        6, kw=7) == 'default2(6, 7) alt2')
    assert TwoAlternativesForceSecond().alt1() == 'reverse default2(0, 0) alt2'


def test_implement_any_one():
    # Creates circular alternative dependencies
    class AnyOneAbc(metaclass=ABCMetaImplementAnyOneOf):

        def _method1_with_2(self):
            return '1-2 ' + self.method2()

        def _method1_with_3(self):
            return '1-3 ' + self.method3()

        def _method2_with_1(self):
            return '2-1 ' + self.method1()

        def _method2_with_3(self):
            return '2-3 ' + self.method3()

        def _method3_with_1(self):
            return '3-1 ' + self.method1()

        def _method3_with_2(self):
            return '3-2 ' + self.method2()

        @alternative(requires='method2', implementation=_method1_with_2)
        @alternative(requires='method3', implementation=_method1_with_3)
        def method1(self):
            """Method1."""

        @alternative(requires='method1', implementation=_method2_with_1)
        @alternative(requires='method3', implementation=_method2_with_3)
        def method2(self):
            """Method2."""

        @alternative(requires='method1', implementation=_method3_with_1)
        @alternative(requires='method2', implementation=_method3_with_2)
        def method3(self):
            """Method3."""

    class Implement1(AnyOneAbc):

        def method1(self):
            """Method1 child."""
            return 'child1'

    class Implement2(AnyOneAbc):

        def method2(self):
            """Method2 child."""
            return 'child2'

    class Implement3(AnyOneAbc):

        def method3(self):
            """Method3 child."""
            return 'child3'

    with pytest.raises(TypeError, match='abstract'):
        AnyOneAbc()
    assert Implement1().method1() == 'child1'
    assert Implement1().method2() == '2-1 child1'
    assert Implement1().method3() == '3-1 child1'
    assert Implement2().method1() == '1-2 child2'
    assert Implement2().method2() == 'child2'
    assert Implement2().method3() == '3-2 child2'
    assert Implement3().method1() == '1-3 child3'
    assert Implement3().method2() == '2-3 child3'
    assert Implement3().method3() == 'child3'
