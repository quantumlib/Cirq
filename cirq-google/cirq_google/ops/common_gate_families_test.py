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

"""Tests for Common Gate Families used in cirq-google"""

import pytest
import sympy
import numpy as np
import cirq
import cirq_google

ALL_POSSIBLE_FSIM_GATES = [
    cirq.CZPowGate,
    cirq.FSimGate,
    cirq.PhasedFSimGate,
    cirq.ISwapPowGate,
    cirq.PhasedISwapPowGate,
    cirq.IdentityGate,
]
theta, phi = sympy.Symbol("theta"), sympy.Symbol("phi")
FOR_IDENTITY = [
    (cirq.CZ ** theta, {theta: 2}),
    (cirq.IdentityGate(2), {}),
    (cirq.FSimGate(theta, phi), {theta: 0, phi: 0}),
    (cirq.PhasedFSimGate(theta, 0, 0, 0, phi), {theta: 0, phi: 0}),
    (cirq.ISWAP ** theta, {theta: 4}),
    (cirq.PhasedISwapPowGate(exponent=theta, phase_exponent=phi), {theta: 4, phi: 2}),
]

FOR_CZPOW_GATE = [
    (cirq.CZPowGate(exponent=theta, global_shift=2.5), {theta: 0.1}),
    (cirq.CZ ** theta, {theta: 0.5}),
    (cirq.FSimGate(theta=theta, phi=0.1), {theta: 0}),
    (cirq.FSimGate(theta=2 * np.pi, phi=phi), {phi: -0.4}),
    (
        cirq.PhasedFSimGate.from_fsim_rz(
            theta=theta,
            phi=10,
            rz_angles_before=(theta, theta),
            rz_angles_after=(phi, phi),
        ),
        {theta: 0, phi: 2 * np.pi},
    ),
    (
        cirq.PhasedFSimGate.from_fsim_rz(
            theta=theta,
            phi=phi,
            rz_angles_before=(2 * np.pi, 2 * np.pi),
            rz_angles_after=(0, 0),
        ),
        {theta: 2 * np.pi, phi: -0.01},
    ),
] + FOR_IDENTITY

FOR_ISWAP_GATE = [
    (cirq.ISwapPowGate(exponent=theta, global_shift=2.5), {theta: 0.1}),
    (cirq.PhasedISwapPowGate(exponent=0.1, phase_exponent=phi), {phi: 2}),
    (cirq.SQRT_ISWAP_INV ** theta, {theta: 0.1}),
    (cirq.FSimGate(theta=theta, phi=phi), {theta: 0.1, phi: 0}),
    (cirq.FSimGate(theta=-0.4, phi=phi), {phi: 2 * np.pi}),
    (
        cirq.PhasedFSimGate.from_fsim_rz(
            theta=10,
            phi=phi,
            rz_angles_before=(phi, phi),
            rz_angles_after=(theta, theta),
        ),
        {theta: 2 * np.pi, phi: 0},
    ),
    (
        cirq.PhasedFSimGate.from_fsim_rz(
            theta=theta,
            phi=phi,
            rz_angles_before=(phi, phi),
            rz_angles_after=(0, 0),
        ),
        {theta: -0.01, phi: 2 * np.pi},
    ),
] + FOR_IDENTITY

FOR_PHASED_ISWAP_GATE = [
    (cirq.PhasedISwapPowGate(exponent=0.1, phase_exponent=phi), {phi: 0.24}),
    *[
        (cirq.PhasedFSimGate.from_fsim_rz(theta, phi, (-p, p), (p, -p)), {theta: tv, phi: pv})
        for p in [np.pi / 4, 0.01, theta, phi]
        for tv, pv in [(0.4, 0), (-0.1, 2 * np.pi)]
    ],
] + FOR_ISWAP_GATE[
    1:
]  # type: ignore

FOR_FSIM_GATE = (
    [
        (cirq.CZPowGate(exponent=theta, global_shift=2.5), {theta: 0.8}),
        (cirq.ISwapPowGate(exponent=theta, global_shift=2.5), {theta: 0.8}),
        (cirq.FSimGate(theta, phi), {theta: 7 * np.pi / 4, phi: 0.0}),
        (cirq_google.SYC, {}),
        (cirq.PhasedFSimGate(theta=0.3, chi=theta, phi=phi), {theta: 0, phi: 0.3}),
    ]
    + FOR_CZPOW_GATE[1:]  # type: ignore
    + FOR_ISWAP_GATE[2 : -len(FOR_IDENTITY)]  # type: ignore
)

FOR_PHASED_FSIM_GATE = FOR_FSIM_GATE + [
    (cirq.PhasedISwapPowGate(exponent=theta, phase_exponent=phi), {theta: -0.5, phi: 0.75}),
    (cirq.PhasedISwapPowGate(exponent=theta, phase_exponent=phi), {theta: 0.5, phi: 10}),
]


@pytest.mark.parametrize(
    'gate, params, target_type',
    [
        *[(g, param, cirq.IdentityGate) for (g, param) in FOR_IDENTITY],
        *[(g, param, cirq.CZPowGate) for (g, param) in FOR_CZPOW_GATE],
        *[(g, param, cirq.ISwapPowGate) for (g, param) in FOR_ISWAP_GATE],
        *[(g, param, cirq.PhasedISwapPowGate) for (g, param) in FOR_PHASED_ISWAP_GATE],
        *[(g, param, cirq.FSimGate) for (g, param) in FOR_FSIM_GATE],
        *[(g, param, cirq.PhasedFSimGate) for (g, param) in FOR_PHASED_FSIM_GATE],
    ],
)
def test_fsim_gate_family_convert_accept(gate, params, target_type):
    # Test Parameterized gate conversion.
    gate_family_allow_symbols = cirq_google.FSimGateFamily(allow_symbols=True)
    assert isinstance(gate_family_allow_symbols.convert(gate, target_type), target_type)
    # Test Non-Parameterized gate conversion.
    resolved_gate = cirq.resolve_parameters(gate, params)
    target_gate = cirq_google.FSimGateFamily().convert(resolved_gate, target_type)
    assert isinstance(target_gate, target_type)
    np.testing.assert_array_almost_equal(cirq.unitary(resolved_gate), cirq.unitary(target_gate))
    # Test Parameterized gate accepted.
    assert gate in cirq_google.FSimGateFamily(gates_to_accept=[target_type], allow_symbols=True)
    assert gate in cirq_google.FSimGateFamily(gates_to_accept=[resolved_gate], allow_symbols=True)
    # Test Non-Parameterized gate accepted.
    assert resolved_gate in cirq_google.FSimGateFamily(gates_to_accept=[target_type])
    assert resolved_gate in cirq_google.FSimGateFamily(gates_to_accept=[resolved_gate])


def test_fsim_gate_family_raises():
    with pytest.raises(ValueError, match='must be one of'):
        _ = cirq_google.FSimGateFamily(gate_types_to_check=[cirq_google.SycamoreGate])
    with pytest.raises(ValueError, match='Parameterized gate'):
        _ = cirq_google.FSimGateFamily(gates_to_accept=[cirq.CZ ** sympy.Symbol('theta')])
    with pytest.raises(ValueError, match='must be either a type from or an instance of'):
        _ = cirq_google.FSimGateFamily(gates_to_accept=[cirq.CNOT])
    with pytest.raises(ValueError, match='must be either a type from or an instance of'):
        _ = cirq_google.FSimGateFamily(gates_to_accept=[cirq_google.SycamoreGate])
    with pytest.raises(ValueError, match='must be one of'):
        _ = cirq_google.FSimGateFamily().convert(cirq.ISWAP, cirq_google.SycamoreGate)


def test_fsim_gate_family_eq():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(
        cirq_google.FSimGateFamily(),
        cirq_google.FSimGateFamily(gate_types_to_check=ALL_POSSIBLE_FSIM_GATES),
        cirq_google.FSimGateFamily(gate_types_to_check=ALL_POSSIBLE_FSIM_GATES[::-1]),
    )
    eq.add_equality_group(
        cirq_google.FSimGateFamily(allow_symbols=True),
        cirq_google.FSimGateFamily(gate_types_to_check=ALL_POSSIBLE_FSIM_GATES, allow_symbols=True),
        cirq_google.FSimGateFamily(
            gate_types_to_check=ALL_POSSIBLE_FSIM_GATES[::-1], allow_symbols=True
        ),
    )
    eq.add_equality_group(
        cirq_google.FSimGateFamily(
            gates_to_accept=[
                cirq_google.SYC,
                cirq.SQRT_ISWAP,
                cirq.SQRT_ISWAP,
                cirq.CZPowGate,
                cirq.PhasedISwapPowGate,
            ],
            allow_symbols=True,
        ),
        cirq_google.FSimGateFamily(
            gates_to_accept=[
                cirq.FSimGate(theta=np.pi / 2, phi=np.pi / 6),
                cirq.SQRT_ISWAP,
                cirq.CZPowGate,
                cirq.CZPowGate,
                cirq.PhasedISwapPowGate,
                cirq.PhasedISwapPowGate,
            ],
            gate_types_to_check=ALL_POSSIBLE_FSIM_GATES + [cirq.FSimGate],
            allow_symbols=True,
        ),
        cirq_google.FSimGateFamily(
            gates_to_accept=[
                cirq.FSimGate(theta=np.pi / 2, phi=np.pi / 6),
                cirq.SQRT_ISWAP,
                cirq.CZPowGate,
                cirq.PhasedISwapPowGate,
            ],
            gate_types_to_check=ALL_POSSIBLE_FSIM_GATES[::-1] + [cirq.FSimGate],
            allow_symbols=True,
        ),
    )


@pytest.mark.parametrize(
    'gate_family',
    [
        cirq_google.FSimGateFamily(),
        cirq_google.FSimGateFamily(allow_symbols=True),
        cirq_google.FSimGateFamily(
            gates_to_accept=[
                cirq.FSimGate(theta=np.pi / 2, phi=np.pi / 6),
                cirq.SQRT_ISWAP,
                cirq.CZPowGate,
                cirq.PhasedISwapPowGate,
            ],
            gate_types_to_check=ALL_POSSIBLE_FSIM_GATES[::-1] + [cirq.FSimGate],  # type:ignore
            allow_symbols=True,
            atol=1e-8,
        ),
    ],
)
def test_fsim_gate_family_repr(gate_family):
    cirq.testing.assert_equivalent_repr(gate_family, setup_code='import cirq\nimport cirq_google')
    assert 'FSimGateFamily' in str(gate_family)


@cirq.value.value_equality(approximate=True, distinct_child_types=True)
class UnequalSycGate(cirq.FSimGate):
    def __init__(self, is_parameterized: bool = False):
        super().__init__(
            theta=theta if is_parameterized else np.pi / 2,
            phi=phi if is_parameterized else np.pi / 6,
        )


def test_fsim_gate_family_convert_rejects():
    # Non compatible, random 1/2/3 qubit gates.
    for gate in [cirq.rx(np.pi / 2), cirq.CNOT, cirq.CCNOT]:
        assert cirq_google.FSimGateFamily().convert(gate, cirq.PhasedFSimGate) is None
        assert gate not in cirq_google.FSimGateFamily(gates_to_accept=[cirq.PhasedFSimGate])
    # Custom gate with an overriden `_value_equality_values_cls_`.
    assert UnequalSycGate() not in cirq_google.FSimGateFamily(gates_to_accept=[cirq_google.SYC])
    assert UnequalSycGate(is_parameterized=True) not in cirq_google.FSimGateFamily(
        gates_to_accept=[cirq_google.SYC], allow_symbols=True
    )
    # Partially paramaterized incompatible gate.
    assert cirq.FSimGate(theta, np.pi / 2) not in cirq_google.FSimGateFamily(
        gates_to_accept=[cirq.PhasedISwapPowGate(exponent=0.5, phase_exponent=0.1), cirq.CZPowGate]
    )
