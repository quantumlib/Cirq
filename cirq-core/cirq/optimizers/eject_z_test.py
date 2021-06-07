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
import numpy as np
import sympy

import cirq
from cirq.optimizers.eject_z import _try_get_known_z_half_turns


def assert_optimizes(
    before: cirq.Circuit, expected: cirq.Circuit, eject_parameterized: bool = False
):
    opt = cirq.EjectZ(eject_parameterized=eject_parameterized)

    if cirq.has_unitary(before):
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
            before, expected, atol=1e-8
        )

    circuit = before.copy()
    opt.optimize_circuit(circuit)
    opt.optimize_circuit(expected)

    cirq.testing.assert_same_circuits(circuit, expected)

    # And it should be idempotent.
    opt.optimize_circuit(circuit)
    cirq.testing.assert_same_circuits(circuit, expected)


def assert_removes_all_z_gates(circuit: cirq.Circuit, eject_parameterized: bool = True):
    opt = cirq.EjectZ(eject_parameterized=eject_parameterized)
    optimized = circuit.copy()
    opt.optimize_circuit(optimized)
    for op in optimized.all_operations():
        assert _try_get_known_z_half_turns(op, eject_parameterized) is None
        if isinstance(op.gate, cirq.PhasedXZGate) and (
            eject_parameterized or not cirq.is_parameterized(op.gate.z_exponent)
        ):
            assert op.gate.z_exponent == 0

    if cirq.is_parameterized(circuit):
        for a in (0, 0.1, 0.5, 1.0, -1.0, 3.0):
            (
                cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
                    cirq.resolve_parameters(circuit, {'a': a}),
                    cirq.resolve_parameters(optimized, {'a': a}),
                    atol=1e-8,
                )
            )
    else:
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
            circuit, optimized, atol=1e-8
        )


def test_single_z_stays():
    q = cirq.NamedQubit('q')
    assert_optimizes(
        before=cirq.Circuit(
            [
                cirq.Moment([cirq.Z(q) ** 0.5]),
            ]
        ),
        expected=cirq.Circuit(
            [
                cirq.Moment([cirq.Z(q) ** 0.5]),
            ]
        ),
    )


def test_single_phased_xz_stays():
    gate = cirq.PhasedXZGate(axis_phase_exponent=0.2, x_exponent=0.3, z_exponent=0.4)
    q = cirq.NamedQubit('q')
    assert_optimizes(before=cirq.Circuit(gate(q)), expected=cirq.Circuit(gate(q)))


def test_ignores_xz_and_cz():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    assert_optimizes(
        before=cirq.Circuit(
            [
                cirq.Moment([cirq.X(a) ** 0.5]),
                cirq.Moment([cirq.Y(b) ** 0.5]),
                cirq.Moment([cirq.CZ(a, b) ** 0.25]),
                cirq.Moment([cirq.Y(a) ** 0.5]),
                cirq.Moment([cirq.X(b) ** 0.5]),
            ]
        ),
        expected=cirq.Circuit(
            [
                cirq.Moment([cirq.X(a) ** 0.5]),
                cirq.Moment([cirq.Y(b) ** 0.5]),
                cirq.Moment([cirq.CZ(a, b) ** 0.25]),
                cirq.Moment([cirq.Y(a) ** 0.5]),
                cirq.Moment([cirq.X(b) ** 0.5]),
            ]
        ),
    )


def test_early_z():
    q = cirq.NamedQubit('q')
    assert_optimizes(
        before=cirq.Circuit(
            [
                cirq.Moment([cirq.Z(q) ** 0.5]),
                cirq.Moment(),
                cirq.Moment(),
            ]
        ),
        expected=cirq.Circuit(
            [
                cirq.Moment([cirq.Z(q) ** 0.5]),
                cirq.Moment(),
                cirq.Moment(),
            ]
        ),
    )


def test_multi_z_merges():
    q = cirq.NamedQubit('q')
    assert_optimizes(
        before=cirq.Circuit(
            [
                cirq.Moment([cirq.Z(q) ** 0.5]),
                cirq.Moment([cirq.Z(q) ** 0.25]),
            ]
        ),
        expected=cirq.Circuit(
            [
                cirq.Moment(),
                cirq.Moment([cirq.Z(q) ** 0.75]),
            ]
        ),
    )


def test_z_pushes_past_xy_and_phases_it():
    q = cirq.NamedQubit('q')
    assert_optimizes(
        before=cirq.Circuit(
            [
                cirq.Moment([cirq.Z(q) ** 0.5]),
                cirq.Moment([cirq.Y(q) ** 0.25]),
            ]
        ),
        expected=cirq.Circuit(
            [
                cirq.Moment(),
                cirq.Moment([cirq.X(q) ** 0.25]),
                cirq.Moment([cirq.Z(q) ** 0.5]),
            ]
        ),
    )


def test_z_pushes_past_cz():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    assert_optimizes(
        before=cirq.Circuit(
            [
                cirq.Moment([cirq.Z(a) ** 0.5]),
                cirq.Moment([cirq.CZ(a, b) ** 0.25]),
            ]
        ),
        expected=cirq.Circuit(
            [
                cirq.Moment(),
                cirq.Moment([cirq.CZ(a, b) ** 0.25]),
                cirq.Moment([cirq.Z(a) ** 0.5]),
            ]
        ),
    )


def test_measurement_consumes_zs():
    q = cirq.NamedQubit('q')
    assert_optimizes(
        before=cirq.Circuit(
            [
                cirq.Moment([cirq.Z(q) ** 0.5]),
                cirq.Moment([cirq.Z(q) ** 0.25]),
                cirq.Moment([cirq.measure(q)]),
            ]
        ),
        expected=cirq.Circuit(
            [
                cirq.Moment(),
                cirq.Moment(),
                cirq.Moment([cirq.measure(q)]),
            ]
        ),
    )


def test_unphaseable_causes_earlier_merge_without_size_increase():
    class UnknownGate(cirq.SingleQubitGate):
        pass

    u = UnknownGate()

    # pylint: disable=not-callable
    q = cirq.NamedQubit('q')
    assert_optimizes(
        before=cirq.Circuit(
            [
                cirq.Moment([cirq.Z(q)]),
                cirq.Moment([u(q)]),
                cirq.Moment([cirq.Z(q) ** 0.5]),
                cirq.Moment([cirq.X(q)]),
                cirq.Moment([cirq.Z(q) ** 0.25]),
                cirq.Moment([cirq.X(q)]),
                cirq.Moment([u(q)]),
            ]
        ),
        expected=cirq.Circuit(
            [
                cirq.Moment([cirq.Z(q)]),
                cirq.Moment([u(q)]),
                cirq.Moment(),
                cirq.Moment([cirq.PhasedXPowGate(phase_exponent=-0.5)(q)]),
                cirq.Moment(),
                cirq.Moment([cirq.PhasedXPowGate(phase_exponent=-0.75).on(q)]),
                cirq.Moment([cirq.Z(q) ** 0.75]),
                cirq.Moment([u(q)]),
            ]
        ),
    )


@pytest.mark.parametrize(
    'sym',
    [
        sympy.Symbol('a'),
        sympy.Symbol('a') + 1,
    ],
)
def test_symbols_block(sym):
    q = cirq.NamedQubit('q')
    assert_optimizes(
        before=cirq.Circuit(
            [
                cirq.Moment([cirq.Z(q)]),
                cirq.Moment([cirq.Z(q) ** sym]),
                cirq.Moment([cirq.Z(q) ** 0.25]),
            ]
        ),
        expected=cirq.Circuit(
            [
                cirq.Moment(),
                cirq.Moment([cirq.Z(q) ** sym]),
                cirq.Moment([cirq.Z(q) ** 1.25]),
            ]
        ),
    )


@pytest.mark.parametrize(
    'sym',
    [
        sympy.Symbol('a'),
        sympy.Symbol('a') + 1,
    ],
)
def test_symbols_eject(sym):
    q = cirq.NamedQubit('q')
    assert_optimizes(
        before=cirq.Circuit(
            [
                cirq.Moment([cirq.Z(q)]),
                cirq.Moment([cirq.Z(q) ** sym]),
                cirq.Moment([cirq.Z(q) ** 0.25]),
            ]
        ),
        expected=cirq.Circuit(
            [
                cirq.Moment(),
                cirq.Moment(),
                cirq.Moment([cirq.Z(q) ** (sym + 1.25)]),
            ]
        ),
        eject_parameterized=True,
    )


def test_removes_zs():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    assert_removes_all_z_gates(cirq.Circuit(cirq.Z(a), cirq.measure(a)))

    assert_removes_all_z_gates(cirq.Circuit(cirq.Z(a), cirq.measure(a, b)))

    assert_removes_all_z_gates(cirq.Circuit(cirq.Z(a), cirq.Z(a), cirq.measure(a)))

    assert_removes_all_z_gates(cirq.Circuit(cirq.Z(a), cirq.measure(a, key='k')))

    assert_removes_all_z_gates(cirq.Circuit(cirq.Z(a), cirq.X(a), cirq.measure(a)))

    assert_removes_all_z_gates(cirq.Circuit(cirq.Z(a), cirq.X(a), cirq.X(a), cirq.measure(a)))

    assert_removes_all_z_gates(
        cirq.Circuit(cirq.Z(a), cirq.Z(b), cirq.CZ(a, b), cirq.CZ(a, b), cirq.measure(a, b))
    )

    assert_removes_all_z_gates(
        cirq.Circuit(
            cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=0, z_exponent=1).on(a),
            cirq.measure(a),
        )
    )

    assert_removes_all_z_gates(
        cirq.Circuit(
            cirq.Z(a) ** sympy.Symbol('a'),
            cirq.Z(b) ** (sympy.Symbol('a') + 1),
            cirq.CZ(a, b),
            cirq.CZ(a, b),
            cirq.measure(a, b),
        ),
        eject_parameterized=True,
    )


def test_unknown_operation_blocks():
    q = cirq.NamedQubit('q')

    class UnknownOp(cirq.Operation):
        @property
        def qubits(self):
            return [q]

        def with_qubits(self, *new_qubits):
            raise NotImplementedError()

    u = UnknownOp()

    assert_optimizes(
        before=cirq.Circuit(
            [
                cirq.Moment([cirq.Z(q)]),
                cirq.Moment([u]),
            ]
        ),
        expected=cirq.Circuit(
            [
                cirq.Moment([cirq.Z(q)]),
                cirq.Moment([u]),
            ]
        ),
    )


def test_swap():
    a, b = cirq.LineQubit.range(2)
    original = cirq.Circuit([cirq.rz(0.123).on(a), cirq.SWAP(a, b)])
    optimized = original.copy()

    cirq.EjectZ().optimize_circuit(optimized)
    cirq.DropEmptyMoments().optimize_circuit(optimized)

    assert optimized[0].operations == (cirq.SWAP(a, b),)
    # Note: EjectZ drops `global_phase` from Rz turning it into a Z
    assert optimized[1].operations == (cirq.Z(b) ** (0.123 / np.pi),)
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(original), cirq.unitary(optimized), atol=1e-8
    )


@pytest.mark.parametrize('exponent', (0, 2, 1.1, -2, -1.6))
def test_not_a_swap(exponent):
    a, b = cirq.LineQubit.range(2)
    assert not cirq.optimizers.eject_z._is_swaplike(cirq.SWAP(a, b) ** exponent)


@pytest.mark.parametrize('theta', (np.pi / 2, -np.pi / 2, np.pi / 2 + 5 * np.pi))
def test_swap_fsim(theta):
    a, b = cirq.LineQubit.range(2)
    original = cirq.Circuit([cirq.rz(0.123).on(a), cirq.FSimGate(theta=theta, phi=0.123).on(a, b)])
    optimized = original.copy()

    cirq.EjectZ().optimize_circuit(optimized)
    cirq.DropEmptyMoments().optimize_circuit(optimized)

    assert optimized[0].operations == (cirq.FSimGate(theta=theta, phi=0.123).on(a, b),)
    # Note: EjectZ drops `global_phase` from Rz turning it into a Z
    assert optimized[1].operations == (cirq.Z(b) ** (0.123 / np.pi),)
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(original), cirq.unitary(optimized), atol=1e-8
    )


@pytest.mark.parametrize('theta', (0, 5 * np.pi, -np.pi))
def test_not_a_swap_fsim(theta):
    a, b = cirq.LineQubit.range(2)
    assert not cirq.optimizers.eject_z._is_swaplike(cirq.FSimGate(theta=theta, phi=0.456).on(a, b))


@pytest.mark.parametrize('exponent', (1, -1))
def test_swap_iswap(exponent):
    a, b = cirq.LineQubit.range(2)
    original = cirq.Circuit([cirq.rz(0.123).on(a), cirq.ISWAP(a, b) ** exponent])
    optimized = original.copy()

    cirq.EjectZ().optimize_circuit(optimized)
    cirq.DropEmptyMoments().optimize_circuit(optimized)

    assert optimized[0].operations == (cirq.ISWAP(a, b) ** exponent,)
    # Note: EjectZ drops `global_phase` from Rz turning it into a Z
    assert optimized[1].operations == (cirq.Z(b) ** (0.123 / np.pi),)
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(original), cirq.unitary(optimized), atol=1e-8
    )
