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

"""A utility class for testing ordering methods.

To test an ordering method, create an OrderTester and add several
equivalence groups or items to it. The order tester will check that
the items within each group are all equal to each other, and every new
added item or group is strictly ascending with regard to the previously
added items or groups.

It will also check that a==b implies hash(a)==hash(b).
"""

from typing import Any

from cirq.testing.equals_tester import EqualsTester


class OrderTester(EqualsTester):
    """Tests ordering against user-provided disjoint ordered groups or items."""

    def __init__(self):
        self.groups = [(ClassSmallerThanEverythingElse(),)]

    def _verify_ascending(self, v1, v2):
        """Verifies that (v1, v2) is a strictly ascending sequence."""
        lt_1 = NotImplemented
        not_lt_2 = NotImplemented
        gt_2 = NotImplemented
        not_gt_1 = NotImplemented
        le_1 = NotImplemented
        not_le_2 = NotImplemented
        ge_2 = NotImplemented
        not_ge_1 = NotImplemented

        try:
            lt_1 = v1 < v2
        except TypeError:
            pass

        try:
            not_lt_2 = not v2 < v1
        except TypeError:
            pass

        assert(lt_1, not_lt_2) in [
            (True, True),
            (True, NotImplemented),
            (NotImplemented, True),
            (NotImplemented, NotImplemented)
                ]

        try:
            gt_2 = not hasattr(v2, '__gt__') or v2 > v1
        except TypeError:
            pass

        try:
            not_gt_1 = not hasattr(v1, '__gt__') or not v1 > v2
        except TypeError:
            pass

        assert(gt_2, not_gt_1) in [
            (True, True),
            (True, NotImplemented),
            (NotImplemented, True),
            (NotImplemented, NotImplemented),
                ]

        # at least one strict ordering operator should work
        assert(lt_1, gt_2) in [
            (True, True),
            (True, NotImplemented),
            (NotImplemented, True),
        ]

        try:
            le_1 = not hasattr(v1, '__le__') or v1 <= v2
        except TypeError:
            pass

        try:
            not_le_2 = not hasattr(v2, '__le__') or not v2 <= v1
        except TypeError:
            pass

        assert(le_1, not_le_2) in [
            (True, True),
            (True, NotImplemented),
            (NotImplemented, True),
            (NotImplemented, NotImplemented),
                ]

        try:
            ge_2 = not hasattr(v2, '__ge__') or v2 >= v1
        except TypeError:
            pass

        try:
            not_ge_1 = not hasattr(v1, '__ge__') or not v1 >= v2
        except TypeError:
            pass

        assert(ge_2, not_ge_1) in [
            (True, True),
            (True, NotImplemented),
            (NotImplemented, True),
            (NotImplemented, NotImplemented),
                ]

    def verify_ascending_group(self, *group_items: Any):
        """Verifies that the given items are strictly ascending
        with regard to the groups which have been added before.
        """
        assert group_items

        for lesser_group in self.groups:
            for lesser_item in lesser_group:
                for larger_item in group_items:
                    self._verify_ascending(lesser_item, larger_item)


    def add_ascending(self, *items: Any):
        """Tries to add a sequence of ascending items to the order tester.

        This methods asserts that items must all be ascending
        with regard to both each other and the elements which have been already
        added during previous calls.
        Some of the previously added elements might be equivalence groups,
        which are supposed to be equal to each other within that group.

        Args:
          *items: The sequence of strictly ascending items.

        Raises:
            AssertionError: Items are not ascending either
                with regard to each other, or with regard to the elements
                which have been added before.
        """

        assert items

        for item in items:
            self.add_ascending_equivalence_group(item)

    def add_ascending_equivalence_group(self, *group_items: Any):
        """Tries to add an ascending equivalence group to the order tester.

        Asserts that the group items are equal to each other, but strictly
        ascending with regard to the already added groups.

        Adds the objects as a group.

        Args:
            group_items: items making the equivalence group

        Raises:
            AssertionError: The group elements aren't equal to each other,
                or items in another group overlap with the new group.
        """
        # Check that elements are equal with regard to each other.
        super(OrderTester, self).verify_equality_group(*group_items)

        # Check that the new group is strictly ascending with regard to
        # the groups which have been added before.
        self.verify_ascending_group(*group_items)

        # Remember this group.
        self.groups.append(tuple(group_items))


class ClassSmallerThanEverythingElse:
    """Assume that the element of this class is less than anything else."""

    def __eq__(self, other):
        return isinstance(other, ClassSmallerThanEverythingElse)

    def __ne__(self, other):
        return not isinstance(other, ClassSmallerThanEverythingElse)

    def __lt__(self, other):
        return not isinstance(other, ClassSmallerThanEverythingElse)

    def __hash__(self):
        return hash(ClassSmallerThanEverythingElse)


class ClassLargerThanEverythingElse:
    """Assume that the element of this class is less than anything else."""

    def __eq__(self, other):
        return isinstance(other, ClassLargerThanEverythingElse)

    def __ne__(self, other):
        return not isinstance(other, ClassLargerThanEverythingElse)

    def __gt__(self, other):
        return not isinstance(other, ClassLargerThanEverythingElse)

    def __hash__(self):
        return hash(ClassLargerThanEverythingElse)


class UnorderableClass:
    """Assume that the element of this class is less than anything else."""

    def __eq__(self, other):
        return isinstance(other, UnorderableClass)

    def __ne__(self, other):
        return not isinstance(other, UnorderableClass)

    def __lt__(self, other):
        return NotImplemented

    def __hash__(self):
        return hash(UnorderableClass)
