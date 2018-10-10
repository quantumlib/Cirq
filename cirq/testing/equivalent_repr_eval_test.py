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

import numpy as np
import pytest

import cirq


def test_external():
    cirq.testing.assert_equivalent_repr(
        1, 2, 3, 'a', 0.5, 2**0.5, 1.1, 1j)

    cirq.testing.assert_equivalent_repr(
        1, 2, 3, 'a', 0.5, 2**0.5, 1.1, 1j,
        setup_code='')

    cirq.testing.assert_equivalent_repr(
        np.array([5]),
        setup_code='from numpy import array')

    with pytest.raises(AssertionError, match='not defined'):
        cirq.testing.assert_equivalent_repr(
            np.array([5]))


def test_custom_class_repr():
    class CustomRepr:
        setup_code = """class CustomRepr:
            def __init__(self, val):
                self.val = val
        """

        def __init__(self, val, repr_str=None):
            self.val = val
            self.repr_str = repr_str

        def __eq__(self, other):
            return type(other).__name__.endswith(
                'CustomRepr') and self.val == other.val

        def __repr__(self):
            return self.repr_str

    cirq.testing.assert_equivalent_repr(
        CustomRepr('a', "CustomRepr('a')"),
        CustomRepr('b', "CustomRepr('b')"),
        setup_code=CustomRepr.setup_code)

    # Non-equal values.
    with pytest.raises(AssertionError, match=r'eval\(repr\(value\)\): a'):
        cirq.testing.assert_equivalent_repr(
            CustomRepr('a', "'a'"))
    with pytest.raises(AssertionError, match=r'eval\(repr\(value\)\): 1'):
        cirq.testing.assert_equivalent_repr(
            CustomRepr('a', "1"))

    # Single failure out of many.
    with pytest.raises(AssertionError, match=r'eval\(repr\(value\)\): a'):
        cirq.testing.assert_equivalent_repr(
            1,
            5,
            CustomRepr('a', "'a'"))

    # Syntax errors.
    with pytest.raises(AssertionError, match='SyntaxError'):
        cirq.testing.assert_equivalent_repr(
            CustomRepr('a', "("))
    with pytest.raises(AssertionError, match='SyntaxError'):
        cirq.testing.assert_equivalent_repr(
            CustomRepr('a', "return 1"))


def test_imports_cirq_by_default():
    cirq.testing.assert_equivalent_repr(cirq.NamedQubit('a'))
