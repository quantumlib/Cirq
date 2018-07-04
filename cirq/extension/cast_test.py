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

import pytest

import cirq


class DesiredType:
    pass


class ChildType(DesiredType):
    pass


class UnrelatedType:
    pass


class PotentialType(cirq.PotentialImplementation):
    def try_cast_to(self, desired_type, ext):
        if desired_type is DesiredType:
            return DesiredType()
        return None


def test_cast():
    desired = DesiredType()
    child = ChildType()
    unrelated = UnrelatedType()
    potential = PotentialType()

    assert cirq.try_cast(DesiredType, desired) is desired
    assert cirq.try_cast(DesiredType, child) is child
    assert cirq.try_cast(DesiredType, potential) is not None
    assert cirq.try_cast(DesiredType, unrelated) is None
    assert cirq.try_cast(DesiredType, object()) is None
    assert cirq.try_cast(UnrelatedType, potential) is None

    assert cirq.can_cast(DesiredType, desired)
    assert cirq.can_cast(DesiredType, child)
    assert cirq.can_cast(DesiredType, potential)
    assert not cirq.can_cast(DesiredType, unrelated)
    assert not cirq.can_cast(DesiredType, object())
    assert not cirq.can_cast(UnrelatedType, potential)

    assert cirq.cast(DesiredType, desired) is desired
    assert cirq.cast(DesiredType, child) is child
    assert cirq.cast(DesiredType, potential) is not None
    with pytest.raises(TypeError):
        _ = cirq.cast(DesiredType, unrelated)
    with pytest.raises(TypeError):
        _ = cirq.cast(DesiredType, object())
    with pytest.raises(TypeError):
        _ = cirq.cast(UnrelatedType, potential)
