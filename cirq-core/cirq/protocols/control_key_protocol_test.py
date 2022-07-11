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

import cirq


def test_control_key():
    class Named:
        def _control_keys_(self):
            return frozenset([cirq.MeasurementKey('key')])

    class NoImpl:
        def _control_keys_(self):
            return NotImplemented

    assert cirq.control_keys(Named()) == {cirq.MeasurementKey('key')}
    assert not cirq.control_keys(NoImpl())
    assert not cirq.control_keys(5)
