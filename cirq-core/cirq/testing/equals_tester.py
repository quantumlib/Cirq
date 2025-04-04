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

"""A utility class for testing equality methods.

To test an equality method, create an EqualityTester and add several groups
of items to it. The equality tester will check that the items within each
group are all equal to each other, but that items between each group are never
equal to each other. It will also check that a==b implies hash(a)==hash(b).
"""

import collections
import itertools
from typing import Any, Callable, List, Tuple, Union


class EqualsTester:
    """Tests equality against user-provided disjoint equivalence groups."""

    def __init__(self) -> None:
        self._groups: List[Tuple[Union[Any, _ClassUnknownToSubjects], ...]] = [
            (_ClassUnknownToSubjects(),)
        ]

    def _verify_equality_group(self, *group_items: Any):
        """Verifies that a group is an equivalence group.

        This methods asserts that items within the group must all be equal to
        each other, but not equal to any items in other groups that have been
        or will be added.

        Args:
          *group_items: The items making up the equivalence group.

        Raises:
            AssertionError: Items within the group are not equal to each other,
                or items in another group are equal to items within the new
                group, or the items violate the equals-implies-same-hash rule.
        """

        assert group_items

        # Within-group items must be equal.
        for v1, v2 in itertools.product(group_items, group_items):
            same = _eq_check(v1, v2)
            assert same or v1 is not v2, f"{v1!r} isn't equal to itself!"
            assert (
                same
            ), f"{v1!r} and {v2!r} can't be in the same equality group. They're not equal."

        # Between-group items must be unequal.
        for other_group in self._groups:
            for v1, v2 in itertools.product(group_items, other_group):
                assert not _eq_check(
                    v1, v2
                ), f"{v1!r} and {v2!r} can't be in different equality groups. They're equal."

        # Check that group items hash to the same thing, or are all unhashable.
        hashes = [hash(v) if isinstance(v, collections.abc.Hashable) else None for v in group_items]
        if len(set(hashes)) > 1:
            examples = (
                (v1, h1, v2, h2)
                for v1, h1 in zip(group_items, hashes)
                for v2, h2 in zip(group_items, hashes)
                if h1 != h2
            )
            example = next(examples)
            raise AssertionError(
                "Items in the same group produced different hashes. "
                f"Example: hash({example[0]!r}) is {example[1]!r} but "
                f"hash({example[2]!r}) is {example[3]!r}."
            )

        # Test that the objects correctly returns NotImplemented when tested against classes
        # that the object does not know the type of.
        for v in group_items:
            assert _TestsForNotImplemented(v) == v and v == _TestsForNotImplemented(v), (
                "An item did not return NotImplemented when checking equality of this "
                f"item against a different type than the item. Relevant item: {v!r}. "
                "Common problem: returning NotImplementedError instead of NotImplemented. "
            )

    def add_equality_group(self, *group_items: Any):
        """Tries to add a disjoint equivalence group to the equality tester.

        This methods asserts that items within the group must all be equal to
        each other, but not equal to any items in other groups that have been
        or will be added.

        Args:
          *group_items: The items making up the equivalence group.

        Raises:
            AssertionError: Items within the group are not equal to each other,
                or items in another group are equal to items within the new
                group, or the items violate the equals-implies-same-hash rule.
        """

        self._verify_equality_group(*group_items)

        # Remember this group, to enable disjoint checks vs later groups.
        self._groups.append(group_items)

    def make_equality_group(self, *factories: Callable[[], Any]):
        """Tries to add a disjoint equivalence group to the equality tester.

        Uses the factory methods to produce two different objects with the same
        initialization for each factory. Asserts that the objects are equal, but
        not equal to any items in other groups that have been or will be added.
        Adds the objects as a group.

        Args:
            *factories: Methods for producing independent copies of an item.

        Raises:
            AssertionError: The factories produce items not equal to the others,
                or items in another group are equal to items from the factory,
                or the items violate the equal-implies-same-hash rule.
        """
        self.add_equality_group(*(f() for f in factories for _ in range(2)))


class _ClassUnknownToSubjects:
    """Equality methods should be able to deal with the unexpected."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ClassUnknownToSubjects)

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __hash__(self):
        return hash(_ClassUnknownToSubjects)  # pragma: no cover


class _TestsForNotImplemented:
    """Used to test that objects return NotImplemented for equality with other types.

    This class is equal to a specific instance or delegates by returning NotImplemented.
    """

    def __init__(self, other: object) -> None:
        self.other = other

    def __eq__(self, other: object) -> bool:
        if other is not self.other:
            return NotImplemented  # pragma: no cover
        return True


def _eq_check(v1: Any, v2: Any) -> bool:
    eq = v1 == v2
    ne = v1 != v2

    assert eq != ne, f"__eq__ is inconsistent with __ne__ between {v1!r} and {v2!r}"
    return eq
