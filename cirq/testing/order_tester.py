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

import collections

from typing import Any

import itertools


class OrderTester:
    """Tests ordering against user-provided disjoint ordered groups or items."""

    def __init__(self):
        self.groups = [(_ClassUnknownToSubjects(),)]

    def _verify_ascending(self, v1, v2):
        # __lt__ and __gt__ might not necessarily be implemented both.
        lt, gt = None, None
        assert hasattr(v1, '__lt__') or hasattr(v2, '__gt__')
        lt = hasattr(v1, '__lt__') and v1.__lt__(v2)
        gt = hasattr(v2, '__gt__') and v2.__gt__(v1)
        assert lt or gt

    def _verify_equality(self, v1, v2):
        # Binary operators should always work.
        assert v1 == v2
        assert not v1 != v2

        # __eq__ and __neq__ should both be correct or not implemented.
        assert hasattr(v1, '__eq__') == hasattr(v1, '__ne__')
        # Careful: python2 int doesn't have __eq__ or __ne__.
        if hasattr(v1, '__eq__'):
            eq = v1.__eq__(v2)
        if hasattr(v1, '__ne__'):
            ne = v1.__ne__(v2)
        assert (eq, ne) in [(True, False),
                            (NotImplemented, False),
                            (NotImplemented, NotImplemented)]

    def _verify_hash_property(self, *items):
        # TODO: this reuses method from EqualityTester.
        # Either introduce a common library or make one class inherit the other.
        hashes = [hash(v) if isinstance(v, collections.Hashable) else None
                  for v in items]
        if len(set(hashes)) > 1:
            examples = ((v1, h1, v2, h2)
                        for v1, h1 in zip(items, hashes)
                        for v2, h2 in zip(items, hashes)
                        if h1 != h2)
            example = next(examples)
            raise AssertionError(
                'Items in the same group produced different hashes. '
                'Example: hash({}) is {} but hash({}) is {}.'.format(*example))

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

        # Check that items are ascending with regard to each other.
        if len(items) > 1:
            for v1, v2 in zip(items[:-1], items[1:]):
                self._verify_ascending(v1, v2)

        # Check that already added groups are not overlapping with the new
        # group.
        # For this, it should be enough to check that the last item of the last
        # already existing group is coming before first of the new items.
        if self.groups:
            self._verify_ascending(self.groups[-1][-1], items[0])

        # Remember this group, to enable checks vs later groups.
        # Every item becomes its own equivalence group.
        self.groups.extend((item,) for item in items)

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
        for v1, v2 in itertools.product(group_items, group_items):
            self._verify_equality(v1, v2)

        # Check the hash property.
        self._verify_hash_property(*group_items)

        # Check that already added groups are not overlapping with the new
        # group.
        if self.groups:
            self._verify_ascending(self.groups[-1][-1], group_items[0])

        # Remember this group.
        self.groups.append(tuple(group_items))


class _ClassUnknownToSubjects:
    """Assume that the element of this class is less than anything else."""

    def __eq__(self, other):
        return isinstance(other, _ClassUnknownToSubjects)

    def __lt__(self, other):
        return not isinstance(other, _ClassUnknownToSubjects)

    def __hash__(self):
        return hash(_ClassUnknownToSubjects)
