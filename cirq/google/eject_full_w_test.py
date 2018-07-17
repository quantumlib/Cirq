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
from typing import Iterable

import cirq
import cirq.google as cg


def assert_optimizes(before: cirq.Circuit,
                     expected: cirq.Circuit):
    opt = cg.EjectFullW()

    circuit = before.copy()
    opt.optimize_circuit(circuit)

    # They should have equivalent effects.
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        circuit, expected, 1e-8)

    # And match the expected circuit.
    if circuit != expected:
        # coverage: ignore
        print("BEFORE")
        print(before)
        print("EXPECTED")
        print(expected)
        print("AFTER")
        print(circuit)
        print(repr(circuit))
    assert circuit == expected

    # And it should be idempotent.
    opt.optimize_circuit(circuit)
    assert circuit == expected


def quick_circuit(*moments: Iterable[cirq.OP_TREE]) -> cirq.Circuit:
    return cirq.Circuit([cirq.Moment(cirq.flatten_op_tree(m)) for m in moments])


def test_absorbs_z():
    q = cirq.NamedQubit('q')

    # Full Z.
    assert_optimizes(
        before=quick_circuit(
            [cg.ExpWGate(axis_half_turns=0.125).on(q)],
            [cirq.Z(q)],
        ),
        expected=quick_circuit(
            [cg.ExpWGate(axis_half_turns=0.625).on(q)],
            [],
        ))

    # Partial Z.
    assert_optimizes(
        before=quick_circuit(
            [cg.ExpWGate(axis_half_turns=0.125).on(q)],
            [cirq.S(q)],
        ),
        expected=quick_circuit(
            [cg.ExpWGate(axis_half_turns=0.375).on(q)],
            [],
        ))

    # Multiple Zs.
    assert_optimizes(
        before=quick_circuit(
            [cg.ExpWGate(axis_half_turns=0.125).on(q)],
            [cirq.S(q)],
            [cirq.T(q)**-1],
        ),
        expected=quick_circuit(
            [cg.ExpWGate(axis_half_turns=0.25).on(q)],
            [],
            [],
        ))


def test_crosses_czs():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    # Full CZ.
    assert_optimizes(
        before=quick_circuit(
            [cg.ExpWGate(axis_half_turns=0.25).on(a)],
            [cirq.CZ(a, b)],
        ),
        expected=quick_circuit(
            [cg.ExpZGate().on(b)],
            [cg.Exp11Gate().on(a, b)],
            [cg.ExpWGate(axis_half_turns=0.25).on(a)],
        ))
    assert_optimizes(
        before=quick_circuit(
            [cg.ExpWGate(axis_half_turns=0.125).on(a)],
            [cirq.CZ(b, a)],
        ),
        expected=quick_circuit(
            [cg.ExpZGate().on(b)],
            [cg.Exp11Gate().on(a, b)],
            [cg.ExpWGate(axis_half_turns=0.125).on(a)],
        ))

    # Partial CZ.
    assert_optimizes(
        before=quick_circuit(
            [cg.ExpWGate().on(a)],
            [cirq.CZ(a, b)**0.25],
        ),
        expected=quick_circuit(
            [cg.ExpZGate(half_turns=0.25).on(b)],
            [cg.Exp11Gate(half_turns=-0.25).on(a, b)],
            [cg.ExpWGate().on(a)],
        ))

    # Double cross.
    assert_optimizes(
        before=quick_circuit(
            [cg.ExpWGate(axis_half_turns=0.125).on(a)],
            [cg.ExpWGate(axis_half_turns=0.375).on(b)],
            [cirq.CZ(a, b)**0.25],
        ),
        expected=quick_circuit(
            [],
            [],
            [cirq.CZ(a, b)**0.25],
            [cg.ExpWGate(axis_half_turns=0.25).on(a),
             cg.ExpWGate(axis_half_turns=0.5).on(b)],
        ))


def test_toggles_measurements():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    # Single.
    assert_optimizes(
        before=quick_circuit(
            [cg.ExpWGate(axis_half_turns=0.25).on(a)],
            [cirq.measure(a, b)],
        ),
        expected=quick_circuit(
            [],
            [cirq.measure(a, b, invert_mask=(True,))],
        ))
    assert_optimizes(
        before=quick_circuit(
            [cg.ExpWGate(axis_half_turns=0.25).on(b)],
            [cirq.measure(a, b)],
        ),
        expected=quick_circuit(
            [],
            [cirq.measure(a, b, invert_mask=(False, True))],
        ))

    # Multiple.
    assert_optimizes(
        before=quick_circuit(
            [cg.ExpWGate(axis_half_turns=0.25).on(a)],
            [cg.ExpWGate(axis_half_turns=0.25).on(b)],
            [cirq.measure(a, b)],
        ),
        expected=quick_circuit(
            [],
            [],
            [cirq.measure(a, b, invert_mask=(True, True))],
        ))

    # Xmon.
    assert_optimizes(
        before=quick_circuit(
            [cg.ExpWGate(axis_half_turns=0.25).on(a)],
            [cg.XmonMeasurementGate(key='t').on(a, b)],
        ),
        expected=quick_circuit(
            [],
            [cg.XmonMeasurementGate(key='t', invert_mask=(True,)).on(a, b)],
        ))


def test_cancels_other_full_w():
    q = cirq.NamedQubit('q')

    assert_optimizes(
        before=quick_circuit(
            [cg.ExpWGate(axis_half_turns=0.25).on(q)],
            [cg.ExpWGate(axis_half_turns=0.25).on(q)],
        ),
        expected=quick_circuit(
            [],
            [],
        ))

    assert_optimizes(
        before=quick_circuit(
            [cg.ExpWGate(axis_half_turns=0.25).on(q)],
            [cg.ExpWGate(axis_half_turns=0.125).on(q)],
        ),
        expected=quick_circuit(
            [],
            [cg.ExpZGate(half_turns=-0.25).on(q)],
        ))

    assert_optimizes(
        before=quick_circuit(
            [cg.ExpWGate().on(q)],
            [cg.ExpWGate(axis_half_turns=0.25).on(q)],
        ),
        expected=quick_circuit(
            [],
            [cg.ExpZGate(half_turns=0.5).on(q)],
        ))

    assert_optimizes(
        before=quick_circuit(
            [cg.ExpWGate(axis_half_turns=0.25).on(q)],
            [cg.ExpWGate().on(q)],
        ),
        expected=quick_circuit(
            [],
            [cg.ExpZGate(half_turns=-0.5).on(q)],
        ))


def test_phases_partial_ws():
    q = cirq.NamedQubit('q')

    assert_optimizes(
        before=quick_circuit(
            [cg.ExpWGate().on(q)],
            [cg.ExpWGate(axis_half_turns=0.25, half_turns=0.5).on(q)],
        ),
        expected=quick_circuit(
            [],
            [cg.ExpWGate(axis_half_turns=0.25, half_turns=0.5).on(q)],
            [cg.ExpWGate().on(q)],
        ))

    assert_optimizes(
        before=quick_circuit(
            [cg.ExpWGate(axis_half_turns=0.25).on(q)],
            [cg.ExpWGate(half_turns=0.5).on(q)],
        ),
        expected=quick_circuit(
            [],
            [cg.ExpWGate(axis_half_turns=-0.5, half_turns=0.5).on(q)],
            [cg.ExpWGate(axis_half_turns=0.25).on(q)],
        ))

    assert_optimizes(
        before=quick_circuit(
            [cg.ExpWGate(axis_half_turns=0.25).on(q)],
            [cg.ExpWGate(axis_half_turns=0.5, half_turns=0.75).on(q)],
        ),
        expected=quick_circuit(
            [],
            [cg.ExpWGate(half_turns=0.75).on(q)],
            [cg.ExpWGate(axis_half_turns=0.25).on(q)],
        ))


def test_blocked_by_unknown_and_symbols():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    assert_optimizes(
        before=quick_circuit(
            [cg.ExpWGate().on(a)],
            [cirq.SWAP(a, b)],
            [cg.ExpWGate().on(a)],
        ),
        expected=quick_circuit(
            [cg.ExpWGate().on(a)],
            [cirq.SWAP(a, b)],
            [cg.ExpWGate().on(a)],
        ))

    assert_optimizes(
        before=quick_circuit(
            [cg.ExpWGate().on(a)],
            [cg.ExpZGate(half_turns=cirq.Symbol('z')).on(a)],
            [cg.ExpWGate().on(a)],
        ),
        expected=quick_circuit(
            [cg.ExpWGate().on(a)],
            [cg.ExpZGate(half_turns=cirq.Symbol('z')).on(a)],
            [cg.ExpWGate().on(a)],
        ))

    assert_optimizes(
        before=quick_circuit(
            [cg.ExpWGate().on(a)],
            [cg.Exp11Gate(half_turns=cirq.Symbol('z')).on(a, b)],
            [cg.ExpWGate().on(a)],
        ),
        expected=quick_circuit(
            [cg.ExpWGate().on(a)],
            [cg.Exp11Gate(half_turns=cirq.Symbol('z')).on(a, b)],
            [cg.ExpWGate().on(a)],
        ))
