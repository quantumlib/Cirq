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


class OrderTester:
    """Tests ordering against user-provided disjoint ordered groups or items."""

    def __init__(self):
        self.eq_tester = EqualsTester()
        self.eq_tester.groups = []  # otherwise ClassUnknownToSubjects
                                    # throws TypeError for comparisons

    def _groups(self):
        return self.eq_tester.groups

    def _try_comparison(self, comp, a, b) -> bool:
        try:
            return comp(a, b)
        except TypeError as inst:
            assert False, inst

    def _verify_ascending(self, v1, v2):
        """Verifies that (v1, v2) is a strictly ascending sequence."""

        comparisons = [
            ('<', lambda a, b: a < b),
            ('>', lambda a, b: a > b),
            ('<=', lambda a, b: a <= b),
            ('>=', lambda a, b: a >= b)
        ]

        for s, f in comparisons:
            first_int = f(1, 2)
            second_int = f(2, 1)
            first_elem = self._try_comparison(f, v1, v2)
            second_elem = self._try_comparison(f, v2, v1)
            assert first_elem == first_int, (
                '{} {} {} returned {} instead of {}'.
            format(v1, s, v2, first_elem, first_int))
            assert second_elem == second_int, (
                '{} {} {} returned {} instead of {}'.
            format(v2, s, v1, second_elem, second_int))

    def _assert_not_implemented_vs_unknown(self, item: Any):
        self._verify_ascending(ClassSmallerThanEverythingElse(), item)
        self._verify_ascending(item, ClassLargerThanEverythingElse())

    def verify_ascending_group(self, *group_items: Any):
        """Verifies that the given items are strictly ascending
        with regard to the groups which have been added before.
        """
        assert group_items

        for lesser_group in self._groups():
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
            self._assert_not_implemented_vs_unknown(item)
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
        # Check that elements are equal with regard to each other
        # and not equal to any other group.
        self.eq_tester.verify_equality_group(*group_items)

        # Check that the new group is strictly ascending with regard to
        # the groups which have been added before.
        self.verify_ascending_group(*group_items)

        # Remember this group.
        self.eq_tester.groups.append(tuple(group_items))


class ClassSmallerThanEverythingElse:
    """Assume that the element of this class is less than anything else."""

    def __eq__(self, other):
        # coverage: ignore
        return isinstance(other, ClassSmallerThanEverythingElse)

    def __ne__(self, other):
        # coverage: ignore
        return not isinstance(other, ClassSmallerThanEverythingElse)

    def __lt__(self, other):
        return not isinstance(other, ClassSmallerThanEverythingElse)

    def __le__(self, other):
        return True

    def __gt__(self, other):
        return False

    def __ge__(self, other):
        return isinstance(other, ClassSmallerThanEverythingElse)

    def __hash__(self):
        # coverage: ignore
        return hash(ClassSmallerThanEverythingElse)


class ClassLargerThanEverythingElse:
    """Assume that the element of this class is larger than anything else."""

    def __eq__(self, other):
        # coverage: ignore
        return isinstance(other, ClassLargerThanEverythingElse)

    def __ne__(self, other):
        # coverage: ignore
        return not isinstance(other, ClassLargerThanEverythingElse)

    def __lt__(self, other):
        return False

    def __le__(self, other):
        return isinstance(other, ClassLargerThanEverythingElse)

    def __gt__(self, other):
        return not isinstance(other, ClassLargerThanEverythingElse)

    def __ge__(self, other):
        return True

    def __hash__(self):
        # coverage: ignore
        return hash(ClassLargerThanEverythingElse)
