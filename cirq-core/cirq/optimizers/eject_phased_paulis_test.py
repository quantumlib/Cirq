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
from typing import cast, Iterable

import numpy as np
import pytest
import sympy

import cirq


def assert_optimizes(
    before: cirq.Circuit,
    expected: cirq.Circuit,
    compare_unitaries: bool = True,
    eject_parameterized: bool = False,
):
    with cirq.testing.assert_deprecated("Use cirq.eject_phased_paulis", deadline='v1.0'):
        opt = cirq.EjectPhasedPaulis(eject_parameterized=eject_parameterized)

    circuit = before.copy()
    expected = cirq.drop_empty_moments(expected)
    opt.optimize_circuit(circuit)

    # They should have equivalent effects.
    if compare_unitaries:
        if cirq.is_parameterized(circuit):
            for a in (0, 0.1, 0.5, -1.0, np.pi, np.pi / 2):
                params: cirq.ParamDictType = {'x': a, 'y': a / 2, 'z': -2 * a}
                (
                    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
                        cirq.resolve_parameters(circuit, params),
                        cirq.resolve_parameters(expected, params),
                        1e-8,
                    )
                )
        else:
            (
                cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
                    circuit, expected, 1e-8
                )
            )

    # And match the expected circuit.
    assert circuit == expected, (
        "Circuit wasn't optimized as expected.\n"
        "INPUT:\n"
        "{}\n"
        "\n"
        "EXPECTED OUTPUT:\n"
        "{}\n"
        "\n"
        "ACTUAL OUTPUT:\n"
        "{}\n"
        "\n"
        "EXPECTED OUTPUT (detailed):\n"
        "{!r}\n"
        "\n"
        "ACTUAL OUTPUT (detailed):\n"
        "{!r}"
    ).format(before, expected, circuit, expected, circuit)

    # And it should be idempotent.
    opt.optimize_circuit(circuit)
    assert circuit == expected


def quick_circuit(*moments: Iterable[cirq.OP_TREE]) -> cirq.Circuit:
    return cirq.Circuit(
        [cirq.Moment(cast(Iterable[cirq.Operation], cirq.flatten_op_tree(m))) for m in moments]
    )


def test_absorbs_z():
    q = cirq.NamedQubit('q')
    x = sympy.Symbol('x')

    # Full Z.
    assert_optimizes(
        before=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.125).on(q)], [cirq.Z(q)]),
        expected=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.625).on(q)], []),
    )

    # Partial Z.
    assert_optimizes(
        before=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.125).on(q)], [cirq.S(q)]),
        expected=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.375).on(q)], []),
    )

    # parameterized Z.
    assert_optimizes(
        before=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.125).on(q)], [cirq.Z(q) ** x]),
        expected=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.125 + x / 2).on(q)], []),
        eject_parameterized=True,
    )
    assert_optimizes(
        before=quick_circuit(
            [cirq.PhasedXPowGate(phase_exponent=0.125).on(q)], [cirq.Z(q) ** (x + 1)]
        ),
        expected=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.625 + x / 2).on(q)], []),
        eject_parameterized=True,
    )

    # Multiple Zs.
    assert_optimizes(
        before=quick_circuit(
            [cirq.PhasedXPowGate(phase_exponent=0.125).on(q)], [cirq.S(q)], [cirq.T(q) ** -1]
        ),
        expected=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.25).on(q)], [], []),
    )

    # Multiple Parameterized Zs.
    assert_optimizes(
        before=quick_circuit(
            [cirq.PhasedXPowGate(phase_exponent=0.125).on(q)], [cirq.S(q) ** x], [cirq.T(q) ** -x]
        ),
        expected=quick_circuit(
            [cirq.PhasedXPowGate(phase_exponent=0.125 + x * 0.125).on(q)], [], []
        ),
        eject_parameterized=True,
    )

    # Parameterized Phase and Partial Z
    assert_optimizes(
        before=quick_circuit([cirq.PhasedXPowGate(phase_exponent=x).on(q)], [cirq.S(q)]),
        expected=quick_circuit([cirq.PhasedXPowGate(phase_exponent=x + 0.25).on(q)], []),
        eject_parameterized=True,
    )


def test_crosses_czs():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    z = sympy.Symbol('z')

    # Full CZ.
    assert_optimizes(
        before=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.25).on(a)], [cirq.CZ(a, b)]),
        expected=quick_circuit(
            [cirq.Z(b)], [cirq.CZ(a, b)], [cirq.PhasedXPowGate(phase_exponent=0.25).on(a)]
        ),
    )
    assert_optimizes(
        before=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.125).on(a)], [cirq.CZ(b, a)]),
        expected=quick_circuit(
            [cirq.Z(b)], [cirq.CZ(a, b)], [cirq.PhasedXPowGate(phase_exponent=0.125).on(a)]
        ),
    )
    assert_optimizes(
        before=quick_circuit([cirq.PhasedXPowGate(phase_exponent=x).on(a)], [cirq.CZ(b, a)]),
        expected=quick_circuit(
            [cirq.Z(b)], [cirq.CZ(a, b)], [cirq.PhasedXPowGate(phase_exponent=x).on(a)]
        ),
        eject_parameterized=True,
    )

    # Partial CZ.
    assert_optimizes(
        before=quick_circuit([cirq.X(a)], [cirq.CZ(a, b) ** 0.25]),
        expected=quick_circuit([cirq.Z(b) ** 0.25], [cirq.CZ(a, b) ** -0.25], [cirq.X(a)]),
    )
    assert_optimizes(
        before=quick_circuit([cirq.X(a)], [cirq.CZ(a, b) ** x]),
        expected=quick_circuit([cirq.Z(b) ** x], [cirq.CZ(a, b) ** -x], [cirq.X(a)]),
        eject_parameterized=True,
    )

    # Double cross.
    assert_optimizes(
        before=quick_circuit(
            [cirq.PhasedXPowGate(phase_exponent=0.125).on(a)],
            [cirq.PhasedXPowGate(phase_exponent=0.375).on(b)],
            [cirq.CZ(a, b) ** 0.25],
        ),
        expected=quick_circuit(
            [],
            [],
            [cirq.CZ(a, b) ** 0.25],
            [
                cirq.PhasedXPowGate(phase_exponent=0.5).on(b),
                cirq.PhasedXPowGate(phase_exponent=0.25).on(a),
            ],
        ),
    )
    assert_optimizes(
        before=quick_circuit(
            [cirq.PhasedXPowGate(phase_exponent=x).on(a)],
            [cirq.PhasedXPowGate(phase_exponent=y).on(b)],
            [cirq.CZ(a, b) ** z],
        ),
        expected=quick_circuit(
            [],
            [],
            [cirq.CZ(a, b) ** z],
            [
                cirq.PhasedXPowGate(phase_exponent=y + z / 2).on(b),
                cirq.PhasedXPowGate(phase_exponent=x + z / 2).on(a),
            ],
        ),
        eject_parameterized=True,
    )


def test_toggles_measurements():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    x = sympy.Symbol('x')

    # Single.
    assert_optimizes(
        before=quick_circuit(
            [cirq.PhasedXPowGate(phase_exponent=0.25).on(a)], [cirq.measure(a, b)]
        ),
        expected=quick_circuit([], [cirq.measure(a, b, invert_mask=(True,))]),
    )
    assert_optimizes(
        before=quick_circuit(
            [cirq.PhasedXPowGate(phase_exponent=0.25).on(b)], [cirq.measure(a, b)]
        ),
        expected=quick_circuit([], [cirq.measure(a, b, invert_mask=(False, True))]),
    )
    assert_optimizes(
        before=quick_circuit([cirq.PhasedXPowGate(phase_exponent=x).on(b)], [cirq.measure(a, b)]),
        expected=quick_circuit([], [cirq.measure(a, b, invert_mask=(False, True))]),
        eject_parameterized=True,
    )

    # Multiple.
    assert_optimizes(
        before=quick_circuit(
            [cirq.PhasedXPowGate(phase_exponent=0.25).on(a)],
            [cirq.PhasedXPowGate(phase_exponent=0.25).on(b)],
            [cirq.measure(a, b)],
        ),
        expected=quick_circuit([], [], [cirq.measure(a, b, invert_mask=(True, True))]),
    )

    # Xmon.
    assert_optimizes(
        before=quick_circuit(
            [cirq.PhasedXPowGate(phase_exponent=0.25).on(a)], [cirq.measure(a, b, key='t')]
        ),
        expected=quick_circuit([], [cirq.measure(a, b, invert_mask=(True,), key='t')]),
    )


def test_cancels_other_full_w():
    q = cirq.NamedQubit('q')
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')

    assert_optimizes(
        before=quick_circuit(
            [cirq.PhasedXPowGate(phase_exponent=0.25).on(q)],
            [cirq.PhasedXPowGate(phase_exponent=0.25).on(q)],
        ),
        expected=quick_circuit([], []),
    )

    assert_optimizes(
        before=quick_circuit(
            [cirq.PhasedXPowGate(phase_exponent=x).on(q)],
            [cirq.PhasedXPowGate(phase_exponent=x).on(q)],
        ),
        expected=quick_circuit([], []),
        eject_parameterized=True,
    )

    assert_optimizes(
        before=quick_circuit(
            [cirq.PhasedXPowGate(phase_exponent=0.25).on(q)],
            [cirq.PhasedXPowGate(phase_exponent=0.125).on(q)],
        ),
        expected=quick_circuit([], [cirq.Z(q) ** -0.25]),
    )

    assert_optimizes(
        before=quick_circuit([cirq.X(q)], [cirq.PhasedXPowGate(phase_exponent=0.25).on(q)]),
        expected=quick_circuit([], [cirq.Z(q) ** 0.5]),
    )

    assert_optimizes(
        before=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.25).on(q)], [cirq.X(q)]),
        expected=quick_circuit([], [cirq.Z(q) ** -0.5]),
    )

    assert_optimizes(
        before=quick_circuit(
            [cirq.PhasedXPowGate(phase_exponent=x).on(q)],
            [cirq.PhasedXPowGate(phase_exponent=y).on(q)],
        ),
        expected=quick_circuit([], [cirq.Z(q) ** (2 * (y - x))]),
        eject_parameterized=True,
    )


def test_phases_partial_ws():
    q = cirq.NamedQubit('q')
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    z = sympy.Symbol('z')

    assert_optimizes(
        before=quick_circuit(
            [cirq.X(q)], [cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(q)]
        ),
        expected=quick_circuit(
            [], [cirq.PhasedXPowGate(phase_exponent=-0.25, exponent=0.5).on(q)], [cirq.X(q)]
        ),
    )

    assert_optimizes(
        before=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.25).on(q)], [cirq.X(q) ** 0.5]),
        expected=quick_circuit(
            [],
            [cirq.PhasedXPowGate(phase_exponent=0.5, exponent=0.5).on(q)],
            [cirq.PhasedXPowGate(phase_exponent=0.25).on(q)],
        ),
    )

    assert_optimizes(
        before=quick_circuit(
            [cirq.PhasedXPowGate(phase_exponent=0.25).on(q)],
            [cirq.PhasedXPowGate(phase_exponent=0.5, exponent=0.75).on(q)],
        ),
        expected=quick_circuit(
            [], [cirq.X(q) ** 0.75], [cirq.PhasedXPowGate(phase_exponent=0.25).on(q)]
        ),
    )

    assert_optimizes(
        before=quick_circuit(
            [cirq.X(q)], [cirq.PhasedXPowGate(exponent=-0.25, phase_exponent=0.5).on(q)]
        ),
        expected=quick_circuit(
            [], [cirq.PhasedXPowGate(exponent=-0.25, phase_exponent=-0.5).on(q)], [cirq.X(q)]
        ),
    )

    assert_optimizes(
        before=quick_circuit(
            [cirq.PhasedXPowGate(phase_exponent=x).on(q)],
            [cirq.PhasedXPowGate(phase_exponent=y, exponent=z).on(q)],
        ),
        expected=quick_circuit(
            [],
            [cirq.PhasedXPowGate(phase_exponent=2 * x - y, exponent=z).on(q)],
            [cirq.PhasedXPowGate(phase_exponent=x).on(q)],
        ),
        eject_parameterized=True,
    )


@pytest.mark.parametrize('sym', [sympy.Symbol('x'), sympy.Symbol('x') + 1])
def test_blocked_by_unknown_and_symbols(sym):
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    assert_optimizes(
        before=quick_circuit([cirq.X(a)], [cirq.SWAP(a, b)], [cirq.X(a)]),
        expected=quick_circuit([cirq.X(a)], [cirq.SWAP(a, b)], [cirq.X(a)]),
    )

    assert_optimizes(
        before=quick_circuit([cirq.X(a)], [cirq.Z(a) ** sym], [cirq.X(a)]),
        expected=quick_circuit([cirq.X(a)], [cirq.Z(a) ** sym], [cirq.X(a)]),
        compare_unitaries=False,
    )

    assert_optimizes(
        before=quick_circuit([cirq.X(a)], [cirq.CZ(a, b) ** sym], [cirq.X(a)]),
        expected=quick_circuit([cirq.X(a)], [cirq.CZ(a, b) ** sym], [cirq.X(a)]),
        compare_unitaries=False,
    )


def test_zero_x_rotation():
    a = cirq.NamedQubit('a')

    assert_optimizes(before=quick_circuit([cirq.rx(0)(a)]), expected=quick_circuit([cirq.rx(0)(a)]))
