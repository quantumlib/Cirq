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

import fractions
import pytest

from cirq.testing.equals_tester import EqualsTester


def test_add_equality_group_correct():
    eq = EqualsTester()

    eq.add_equality_group(fractions.Fraction(1, 1))

    eq.add_equality_group(fractions.Fraction(1, 2), fractions.Fraction(2, 4))

    eq.add_equality_group(
        fractions.Fraction(2, 3), fractions.Fraction(12, 18), fractions.Fraction(14, 21)
    )

    eq.add_equality_group(2, 2.0, fractions.Fraction(2, 1))

    eq.add_equality_group([1, 2, 3], [1, 2, 3])

    eq.add_equality_group({'b': 3, 'a': 2}, {'a': 2, 'b': 3})

    eq.add_equality_group('unrelated')


def test_assert_make_equality_group():
    eq = EqualsTester()

    with pytest.raises(AssertionError, match="can't be in the same"):
        eq.make_equality_group(object)

    eq.make_equality_group(lambda: 1)
    eq.make_equality_group(lambda: 2, lambda: 2.0)
    eq.add_equality_group(3)

    with pytest.raises(AssertionError, match="can't be in different"):
        eq.add_equality_group(1)
    with pytest.raises(AssertionError, match="can't be in different"):
        eq.make_equality_group(lambda: 1)
    with pytest.raises(AssertionError, match="can't be in different"):
        eq.make_equality_group(lambda: 3)


def test_add_equality_group_not_equivalent():
    eq = EqualsTester()
    with pytest.raises(AssertionError, match="can't be in the same"):
        eq.add_equality_group(1, 2)


def test_add_equality_group_not_disjoint():
    eq = EqualsTester()
    eq.add_equality_group(1)
    with pytest.raises(AssertionError, match="can't be in different"):
        eq.add_equality_group(1)


def test_add_equality_group_bad_hash():
    class KeyHash:
        def __init__(self, k, h):
            self._k = k
            self._h = h

        def __eq__(self, other):
            if not isinstance(other, KeyHash):
                return NotImplemented
            return self._k == other._k

        def __ne__(self, other):
            return not self == other

        def __hash__(self):
            return self._h

    eq = EqualsTester()
    eq.add_equality_group(KeyHash('a', 5), KeyHash('a', 5))
    eq.add_equality_group(KeyHash('b', 5))
    with pytest.raises(AssertionError, match='produced different hashes'):
        eq.add_equality_group(KeyHash('c', 2), KeyHash('c', 3))


def test_add_equality_group_exception_hash():
    class FailHash:
        def __hash__(self):
            raise ValueError('injected failure')

    eq = EqualsTester()
    with pytest.raises(ValueError, match='injected failure'):
        eq.add_equality_group(FailHash())


def test_fails_when_forgot_type_check():
    eq = EqualsTester()

    class NoTypeCheckEqualImplementation:
        def __init__(self):
            self.x = 1

        def __eq__(self, other):
            return self.x == other.x

        def __ne__(self, other):
            return not self == other

        def __hash__(self):
            return hash(self.x)

    with pytest.raises(AttributeError, match="has no attribute 'x'"):
        eq.add_equality_group(NoTypeCheckEqualImplementation())


def test_fails_when_equal_to_everything():
    eq = EqualsTester()

    class AllEqual:
        __hash__ = None

        def __eq__(self, other):
            return True

        def __ne__(self, other):
            return False

    with pytest.raises(AssertionError, match="can't be in different"):
        eq.add_equality_group(AllEqual())


def test_fails_hash_is_default_and_inconsistent():
    eq = EqualsTester()

    class DefaultHashImplementation:
        __hash__ = object.__hash__

        def __init__(self):
            self.x = 1

        def __eq__(self, other):
            if not isinstance(other, type(self)):
                return NotImplemented
            return self.x == other.x

        def __ne__(self, other):
            return not self == other

    with pytest.raises(AssertionError, match='different hashes'):
        eq.make_equality_group(DefaultHashImplementation)


def test_fails_when_ne_is_inconsistent():
    eq = EqualsTester()

    class InconsistentNeImplementation:
        def __init__(self):
            self.x = 1

        def __eq__(self, other):
            if not isinstance(other, type(self)):
                return NotImplemented  # coverage: ignore
            return self.x == other.x

        def __ne__(self, other):
            if not isinstance(other, type(self)):
                return NotImplemented  # coverage: ignore
            return self.x == other.x

        def __hash__(self):
            return hash(self.x)  # coverage: ignore

    with pytest.raises(AssertionError, match='inconsistent'):
        eq.make_equality_group(InconsistentNeImplementation)


def test_fails_when_ne_is_inconsistent_due_to_not_implemented():
    eq = EqualsTester()

    class InconsistentNeImplementation:
        def __init__(self):
            self.x = 1

        def __eq__(self, other):
            if not isinstance(other, type(self)):
                return NotImplemented
            return self.x == other.x

        def __ne__(self, other):
            return NotImplemented

        def __hash__(self):
            return hash(self.x)

    with pytest.raises(AssertionError, match='inconsistent'):
        eq.make_equality_group(InconsistentNeImplementation)


def test_fails_when_not_reflexive():
    eq = EqualsTester()

    class NotReflexiveImplementation:
        def __init__(self):
            self.x = 1

        def __eq__(self, other):
            if other is not self:
                return NotImplemented
            return False

        def __ne__(self, other):
            return not self == other

    with pytest.raises(AssertionError, match='equal to itself'):
        eq.add_equality_group(NotReflexiveImplementation())


def test_fails_when_not_commutative():
    eq = EqualsTester()

    class NotCommutativeImplementation:
        def __init__(self, x):
            self.x = x

        def __eq__(self, other):
            if not isinstance(other, type(self)):
                return NotImplemented
            return self.x <= other.x

        def __ne__(self, other):
            return not self == other

    with pytest.raises(AssertionError, match="can't be in the same"):
        eq.add_equality_group(NotCommutativeImplementation(0), NotCommutativeImplementation(1))

    with pytest.raises(AssertionError, match="can't be in the same"):
        eq.add_equality_group(NotCommutativeImplementation(1), NotCommutativeImplementation(0))


def test_works_on_types():
    eq = EqualsTester()
    eq.add_equality_group(object)
    eq.add_equality_group(int)
    eq.add_equality_group(object())


def test_returns_not_implemented_for_other_types():
    # First we demonstrate an example of the problem.

    # FirstClass is the class that is broken.
    # It returns False when it should return NotImplemented when its __eq__ is called
    # on a class it does not recognize.
    class FirstClass:
        def __init__(self, val):
            self.val = val

        def __eq__(self, other):
            if not isinstance(other, FirstClass):
                return False
            return self.val == other.val

    # So, for example, here is a class that we want to be equal to FirstClass.
    class SecondClass:
        def __init__(self, val):
            self.val = val

        def __eq__(self, other):
            if isinstance(other, (FirstClass, SecondClass)):
                return self.val == other.val
            # Ignore coverage, this is just for illustrative purposes.
            return NotImplemented  # coverage: ignore

    # But we see that this does not work because it fails commutativity of ==
    assert SecondClass("a") == FirstClass("a")
    assert FirstClass("a") != SecondClass("a")

    # The problem is that in the second case FirstClass should return NotImplemented, which
    # will then cause the == call to check whether SecondClass is equal to FirstClass.

    # So if we had done this correctly we would have instead of FirstClass and SecondClass,
    # ThirdClass and FourthClass, respectively.
    class ThirdClass:
        def __init__(self, val):
            self.val = val

        def __eq__(self, other):
            if not isinstance(other, ThirdClass):
                return NotImplemented
            return self.val == other.val

    class FourthClass:
        def __init__(self, val):
            self.val = val

        def __eq__(self, other):
            if isinstance(other, (ThirdClass, FourthClass)):
                return self.val == other.val
            # Ignore coverage, this is just for illustrative purposes.
            return NotImplemented  # coverage: ignore

    # We see this is fixed:
    assert ThirdClass("a") == FourthClass("a")
    assert FourthClass("a") == ThirdClass("a")

    # Now test that EqualsTester catches this.
    eq = EqualsTester()

    with pytest.raises(AssertionError, match="NotImplemented"):
        eq.add_equality_group(FirstClass("a"), FirstClass("a"))

    eq = EqualsTester()
    eq.add_equality_group(ThirdClass("a"), ThirdClass("a"))


def test_not_implemented_error():
    # Common bug is to return NotImplementedError instead of NotImplemented.
    class NotImplementedErrorCase:
        def __init__(self, val):
            self.val = val

        def __eq__(self, other):
            if not isinstance(other, NotImplementedErrorCase):
                return NotImplementedError
            return self.val == other.val

    eq = EqualsTester()
    with pytest.raises(AssertionError, match="NotImplemented"):
        eq.add_equality_group(NotImplementedErrorCase("a"), NotImplementedErrorCase("a"))
