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


_NAMED_COMPARISON_OPERATORS = [
    ('<', lambda a, b: a < b),
    ('>', lambda a, b: a > b),
    ('==', lambda a, b: a == b),
    ('!=', lambda a, b: a != b),
    ('<=', lambda a, b: a <= b),
    ('>=', lambda a, b: a >= b)
]


class OrderTester:
    """Tests ordering against user-provided disjoint ordered groups or items."""

    def __init__(self):
        self._groups = []
        self._eq_tester = EqualsTester()

    def _verify_ordering_one_sided(self, a: Any, b: Any, sign: int):
        """Checks that (a vs b) == (0 vs sign)."""
        for cmp_name, cmp_func in _NAMED_COMPARISON_OPERATORS:
            expected = cmp_func(0, sign)
            actual = cmp_func(a, b)
            assert expected == actual, (
                "Ordering constraint violated. Expected X={} to {} Y={}, "
                "but X {} Y returned {}".format(
                    a,
                    ['be more than', 'equal', 'be less than'][sign + 1],
                    b,
                    cmp_name,
                    actual))

    def _verify_ordering(self, a: Any, b: Any, sign: int):
        """Checks that (a vs b) == (0 vs sign) and (b vs a) == (sign vs 0)."""
        self._verify_ordering_one_sided(a, b, sign)
        self._verify_ordering_one_sided(b, a, -sign)

    def _verify_not_implemented_vs_unknown(self, item: Any):
        try:
            self._verify_ordering(_SmallerThanEverythingElse(), item, +1)
            self._verify_ordering(_EqualToEverything(), item, 0)
            self._verify_ordering(_LargerThanEverythingElse(), item, -1)
        except AssertionError as ex:
            raise AssertionError(
                "Objects should return NotImplemented when compared to an "
                "unknown value, i.e. comparison methods should start with\n"
                "\n"
                "    if not isinstance(other, type(self)):\n"
                "        return NotImplemented\n"
                "\n"
                "That rule is being violated by this value: {!r}".format(
                    item)) from ex

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

        for item in group_items:
            self._verify_not_implemented_vs_unknown(item)

        for item1 in group_items:
            for item2 in group_items:
                self._verify_ordering(item1, item2, 0)

        for lesser_group in self._groups:
            for lesser_item in lesser_group:
                for larger_item in group_items:
                    self._verify_ordering(lesser_item, larger_item, +1)

        # Use equals tester to check hash function consistency.
        self._eq_tester.add_equality_group(*group_items)

        self._groups.append(group_items)


class _EqualToEverything:
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False

    def __lt__(self, other):
        return False

    def __le__(self, other):
        return True

    def __gt__(self, other):
        return False

    def __ge__(self, other):
        return True

    def __repr__(self):
        return '_EqualToEverything'


class _SmallerThanEverythingElse:
    def __eq__(self, other):
        return isinstance(other, _SmallerThanEverythingElse)

    def __ne__(self, other):
        return not isinstance(other, _SmallerThanEverythingElse)

    def __lt__(self, other):
        return not isinstance(other, _SmallerThanEverythingElse)

    def __le__(self, other):
        return True

    def __gt__(self, other):
        return False

    def __ge__(self, other):
        return isinstance(other, _SmallerThanEverythingElse)

    def __repr__(self):
        return 'SmallerThanEverythingElse'


class _LargerThanEverythingElse:
    def __eq__(self, other):
        return isinstance(other, _LargerThanEverythingElse)

    def __ne__(self, other):
        return not isinstance(other, _LargerThanEverythingElse)

    def __lt__(self, other):
        return False

    def __le__(self, other):
        return isinstance(other, _LargerThanEverythingElse)

    def __gt__(self, other):
        return not isinstance(other, _LargerThanEverythingElse)

    def __ge__(self, other):
        return True

    def __repr__(self):
        return 'LargerThanEverythingElse()'
