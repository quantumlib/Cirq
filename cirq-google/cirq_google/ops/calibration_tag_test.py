# Copyright 2020 The Cirq Developers
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

from __future__ import annotations

import cirq
import cirq_google


def test_equality():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq_google.CalibrationTag('blah'), cirq_google.CalibrationTag('blah'))
    eq.add_equality_group(cirq_google.CalibrationTag('blah2'))


def test_hash():
    s = set()
    s.add(cirq_google.CalibrationTag('foo'))
    assert len(s) == 1
    s.add(cirq_google.CalibrationTag('foo'))
    assert len(s) == 1
    s.add(cirq_google.CalibrationTag('bar'))
    assert len(s) == 2


def test_str_repr():
    example_tag = cirq_google.CalibrationTag('foo')
    assert str(example_tag) == 'CalibrationTag(\'foo\')'
    assert repr(example_tag) == 'cirq_google.CalibrationTag(\'foo\')'
    cirq.testing.assert_equivalent_repr(example_tag, setup_code=('import cirq\nimport cirq_google'))
