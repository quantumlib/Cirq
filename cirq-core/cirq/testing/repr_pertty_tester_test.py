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

import cirq.testing


def test_fake_printer():
    p = cirq.testing.FakePrinter()
    assert p.text_pretty == ""
    p.text("stuff")
    assert p.text_pretty == "stuff"
    p.text(" more")
    assert p.text_pretty == "stuff more"


def test_assert_repr_pretty():
    class TestClass:
        def _repr_pretty_(self, p, cycle):
            p.text("TestClass" if cycle else "I'm so pretty")

    cirq.testing.assert_repr_pretty(TestClass(), "I'm so pretty")
    cirq.testing.assert_repr_pretty(TestClass(), "TestClass", cycle=True)

    class TestClassMultipleTexts:
        def _repr_pretty_(self, p, cycle):
            if cycle:
                p.text("TestClass")
            else:
                p.text("I'm so pretty")
                p.text(" I am")

    cirq.testing.assert_repr_pretty(TestClassMultipleTexts(), "I'm so pretty I am")
    cirq.testing.assert_repr_pretty(TestClassMultipleTexts(), "TestClass", cycle=True)
