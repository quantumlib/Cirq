# Copyright 2018 Google LLC
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

"""Tests for studies."""

import pytest

from cirq import study
from cirq.devices import UnconstrainedDevice
from cirq.schedules import Schedule
from cirq.study import Points, Study
from cirq.testing import EqualsTester


def test_equality():
    eq = EqualsTester()
    s = Schedule(UnconstrainedDevice)

    eq.add_equality_group(Study(s, [Points('a', [1])]),
                          Study(s, [Points('a', [1])]))

    eq.add_equality_group(Study(s, [Points('a', [2])]),
                          Study(s, [Points('a', [2])]))


class BadResult(study.TrialResult):
    pass


def test_bad_result():
    with pytest.raises(NotImplementedError):
        BadResult()
