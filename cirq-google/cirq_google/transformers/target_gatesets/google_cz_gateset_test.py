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


def test_eject_phased_paulis():
    before = cirq.Circuit(cirq.PhasedXPowGate(phase_exponent=0.125).on(_qa), cirq.CZ(_qa, _qb))
    expected = cirq.Circuit(
        cirq.Z(_qb), (cirq.CZ**-1)(_qa, _qb), cirq.PhasedXPowGate(phase_exponent=0.125).on(_qa)
    )

    after = cirq.optimize_for_target_gateset(
        before, gateset=cirq_google.GoogleCZTargetGateset(), ignore_failures=False
    )
    cirq.testing.assert_same_circuits(after, expected)


@pytest.mark.parametrize(
    'before',
    [
        cirq.Circuit(cirq.Z(_qa) ** 0.5, cirq.CZ(_qa, _qb)),
        cirq.Circuit(cirq.Z(_qa).with_tags(cirq_google.PhysicalZTag()) ** 0.5, cirq.CZ(_qa, _qb)),
    ],
)
def test_eject_z_disabled(before):
    after = cirq.optimize_for_target_gateset(
        before,
        gateset=cirq_google.GoogleCZTargetGateset(
            additional_gates=[
                cirq.GateFamily(cirq.ZPowGate, tags_to_ignore=[cirq_google.PhysicalZTag()]),
                cirq.GateFamily(cirq.ZPowGate, tags_to_accept=[cirq_google.PhysicalZTag()]),
            ]
        ),
        ignore_failures=False,
    )
    cirq.testing.assert_same_circuits(after, before)


@pytest.mark.parametrize(
    'before, expected',
    [
        (
            cirq.Circuit(cirq.Z(_qa) ** 0.5, cirq.CZ(_qa, _qb)),
            cirq.Circuit(cirq.CZ(_qa, _qb), cirq.Z(_qa) ** 0.5),
        ),
        (
            # PhysicalZ tag is erased
            cirq.Circuit(
                cirq.Z(_qa).with_tags(cirq_google.PhysicalZTag()) ** 0.5, cirq.CZ(_qa, _qb)
            ),
            cirq.Circuit(cirq.CZ(_qa, _qb), cirq.Z(_qa) ** 0.5),
        ),
    ],
)
def test_eject_z_enabled(before, expected):
    after = cirq.optimize_for_target_gateset(
        before,
        gateset=cirq_google.GoogleCZTargetGateset(
            eject_z=True,
            additional_gates=[
                cirq.GateFamily(cirq.ZPowGate, tags_to_ignore=[cirq_google.PhysicalZTag()]),
                cirq.GateFamily(cirq.ZPowGate, tags_to_accept=[cirq_google.PhysicalZTag()]),
            ],
        ),
        ignore_failures=False,
    )
    cirq.testing.assert_same_circuits(after, expected)


@pytest.mark.parametrize(
    'gateset',
    [
        cirq_google.GoogleCZTargetGateset(),
        cirq_google.GoogleCZTargetGateset(
            atol=1e-6, eject_z=True, additional_gates=[cirq.SQRT_ISWAP, cirq.XPowGate]
        ),
        cirq_google.GoogleCZTargetGateset(additional_gates=()),
    ],
)
def test_repr(gateset):
    cirq.testing.assert_equivalent_repr(gateset, setup_code='import cirq\nimport cirq_google')
