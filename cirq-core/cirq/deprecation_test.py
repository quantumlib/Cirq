# Copyright 2019 The Cirq Developers
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


def test_deprecated_modules():
    with cirq.testing.assert_deprecated(
        "cirq.ion", "Use cirq.devices.ion instead", deadline="v0.13", count=None
    ):
        import cirq.ion as ion  # type: ignore

        assert ion is not None

    with cirq.testing.assert_deprecated(
        "cirq.neutral_atoms", "Use cirq.devices.neutral_atoms instead", deadline="v0.13", count=None
    ):
        import cirq.neutral_atoms as na  # type: ignore

        assert na is not None
