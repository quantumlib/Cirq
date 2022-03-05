# Copyright 2022 The Cirq Developers
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
from cirq_google.ops.physical_z_tag import PhysicalZTag
from cirq_google.ops.z_gate_family import ZGateFamily


q = cirq.LineQubit(0)


@pytest.mark.parametrize(
    'physical_z, gate, accepted',
    [
        (True, cirq.Z(q).with_tags(PhysicalZTag()), True),
        (True, cirq.Z(q).with_tags('some_tag'), False),
        (True, cirq.Z(q).with_tags(), False),
        (True, cirq.Z(q), False),
        (True, cirq.Z, False),
        (True, (cirq.Z ** 0.5)(q), False),
        (True, cirq.Z ** 0.5, False),
        (False, cirq.Z(q).with_tags(PhysicalZTag()), False),
        (False, cirq.Z(q).with_tags('some_tag'), True),
        (False, cirq.Z(q).with_tags(), True),
        (False, cirq.Z(q), True),
        (False, cirq.Z, True),
        (False, (cirq.Z ** 0.5)(q), False),
        (False, cirq.Z ** 0.5, False),
    ],
)
def test_z_gate_family_accept(gate, physical_z, accepted):
    assert (gate in ZGateFamily(physical_z)) == accepted


def test_virtual_z_gate_family_repr():
    gate_family = ZGateFamily(physical_z=False)
    cirq.testing.assert_equivalent_repr(gate_family, setup_code='import cirq\nimport cirq_google')
    assert 'ZGateFamily(Virtual)' in str(gate_family)


def test_physical_z_gate_family_repr():
    gate_family = ZGateFamily(physical_z=True)
    cirq.testing.assert_equivalent_repr(gate_family, setup_code='import cirq\nimport cirq_google')
    assert 'ZGateFamily(Physical)' in str(gate_family)
