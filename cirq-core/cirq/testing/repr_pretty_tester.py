# Copyright 2021 The Cirq Developers
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


from typing import Any


class FakePrinter:
    """A fake of iPython's PrettyPrinter which captures text added to this printer.

    Can be used in tests to test a classes `_repr_pretty_` method:

    >>> p = FakePrinter()
    >>> s = object_under_test._repr_pretty(p, cycle=False)
    >>> p.text_pretty
    'my pretty_text'

    Prefer to use `assert_repr_pretty` below.
    """

    def __init__(self):
        self.text_pretty = ""

    def text(self, to_print):
        self.text_pretty += to_print


def assert_repr_pretty(val: Any, text: str, cycle: bool = False):
    """Assert that the given object has a `_repr_pretty_` method that produces the given text.

    Args:
            val: The object to test.
            text: The string that `_repr_pretty_` is expected to return.
            cycle: The value of `cycle` passed to `_repr_pretty_`.  `cycle` represents whether
                the call is made with a potential cycle. Typically one should handle the
                `cycle` equals `True` case by returning text that does not recursively call
                the `_repr_pretty_` to break this cycle.

    Raises:
        AssertionError: If `_repr_pretty_` does not pretty print the given text.
    """
    p = FakePrinter()
    val._repr_pretty_(p, cycle=cycle)
    assert p.text_pretty == text, f"{p.text_pretty} != {text}"
