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
import cirq_google

_qa, _qb = cirq.NamedQubit('a'), cirq.NamedQubit('b')


@pytest.mark.parametrize(
    'before, gate_family',
    [
        (
            cirq.Circuit(cirq.Z(_qa) ** 0.5, cirq.CZ(_qa, _qb)),
            cirq.GateFamily(cirq.ZPowGate, tags_to_ignore=[cirq_google.PhysicalZTag()]),
        ),
        (
            cirq.Circuit(
                (cirq.Z**0.5)(_qa).with_tags(cirq_google.PhysicalZTag()), cirq.CZ(_qa, _qb)
            ),
            cirq.GateFamily(cirq.ZPowGate, tags_to_accept=[cirq_google.PhysicalZTag()]),
        ),
        (
            cirq.Circuit(cirq.PhasedXPowGate(phase_exponent=0.125).on(_qa), cirq.CZ(_qa, _qb)),
            cirq.GateFamily(cirq.PhasedXPowGate),
        ),
    ],
)
def test_eject_paulis_disabled(before, gate_family):
    after = cirq.optimize_for_target_gateset(
        before,
        gateset=cirq_google.GoogleCZTargetGateset(additional_gates=[gate_family]),
        ignore_failures=False,
    )
    cirq.testing.assert_same_circuits(after, before)


@pytest.mark.parametrize(
    'before, expected, gate_family',
    [
        (
            cirq.Circuit(cirq.Z(_qa) ** 0.75, cirq.CZ(_qa, _qb)),
            cirq.Circuit(cirq.CZ(_qa, _qb), cirq.Z(_qa) ** 0.75),
            cirq.GateFamily(cirq.ZPowGate, tags_to_ignore=[cirq_google.PhysicalZTag()]),
        ),
        (
            # PhysicalZ tag is erased
            cirq.Circuit(
                (cirq.Z**0.75)(_qa).with_tags(cirq_google.PhysicalZTag()), cirq.CZ(_qa, _qb)
            ),
            cirq.Circuit(cirq.CZ(_qa, _qb), cirq.Z(_qa) ** 0.75),
            cirq.GateFamily(cirq.ZPowGate, tags_to_accept=[cirq_google.PhysicalZTag()]),
        ),
        (
            cirq.Circuit(cirq.PhasedXPowGate(phase_exponent=0.125).on(_qa), cirq.CZ(_qa, _qb)),
            cirq.Circuit(
                (cirq.CZ**-1)(_qa, _qb),
                cirq.PhasedXPowGate(phase_exponent=0.125).on(_qa),
                cirq.Z(_qb),
            ),
            cirq.PhasedXPowGate,
        ),
    ],
)
def test_eject_paulis_enabled(before, expected, gate_family):
    after = cirq.optimize_for_target_gateset(
        before,
        gateset=cirq_google.GoogleCZTargetGateset(
            eject_paulis=True, additional_gates=[gate_family]
        ),
        ignore_failures=False,
    )
    cirq.testing.assert_same_circuits(after, expected)


@pytest.mark.parametrize(
    'gateset',
    [
        cirq_google.GoogleCZTargetGateset(),
        cirq_google.GoogleCZTargetGateset(
            atol=1e-6, eject_paulis=True, additional_gates=[cirq.SQRT_ISWAP, cirq.XPowGate]
        ),
        cirq_google.GoogleCZTargetGateset(additional_gates=()),
    ],
)
def test_repr(gateset):
    cirq.testing.assert_equivalent_repr(gateset, setup_code='import cirq\nimport cirq_google')
