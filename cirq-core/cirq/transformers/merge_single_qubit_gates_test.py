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

from typing import List

import cirq


def assert_optimizes(optimized: cirq.AbstractCircuit, expected: cirq.AbstractCircuit):
    # Ignore differences that would be caught by follow-up optimizations.
    followup_transformers: List[cirq.TRANSFORMER] = [
        cirq.drop_negligible_operations,
        cirq.drop_empty_moments,
    ]
    for transform in followup_transformers:
        optimized = transform(optimized)
        expected = transform(expected)

    cirq.testing.assert_same_circuits(optimized, expected)


def test_merge_single_qubit_gates_to_phased_x_and_z():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(
        cirq.X(a),
        cirq.Y(b) ** 0.5,
        cirq.CZ(a, b),
        cirq.H(a),
        cirq.Z(a),
        cirq.measure(b, key="m"),
        cirq.H(a).with_classical_controls("m"),
    )
    assert_optimizes(
        optimized=cirq.merge_single_qubit_gates_to_phased_x_and_z(c),
        expected=cirq.Circuit(
            cirq.PhasedXPowGate(phase_exponent=1)(a),
            cirq.Y(b) ** 0.5,
            cirq.CZ(a, b),
            (cirq.PhasedXPowGate(phase_exponent=-0.5)(a)) ** 0.5,
            cirq.measure(b, key="m"),
            cirq.H(a).with_classical_controls("m"),
        ),
    )


def test_merge_single_qubit_gates_to_phased_x_and_z_deep():
    a = cirq.NamedQubit("a")
    c_nested = cirq.FrozenCircuit(cirq.H(a), cirq.Z(a), cirq.H(a).with_tags("ignore"))
    c_nested_merged = cirq.FrozenCircuit(
        cirq.PhasedXPowGate(phase_exponent=-0.5, exponent=0.5).on(a), cirq.H(a).with_tags("ignore")
    )
    c_orig = cirq.Circuit(
        c_nested,
        cirq.CircuitOperation(c_nested).repeat(4).with_tags("ignore"),
        c_nested,
        cirq.CircuitOperation(c_nested).repeat(5).with_tags("preserve_tags"),
        c_nested,
        cirq.CircuitOperation(c_nested).repeat(6),
    )
    c_expected = cirq.Circuit(
        c_nested_merged,
        cirq.CircuitOperation(c_nested).repeat(4).with_tags("ignore"),
        c_nested_merged,
        cirq.CircuitOperation(c_nested_merged).repeat(5).with_tags("preserve_tags"),
        c_nested_merged,
        cirq.CircuitOperation(c_nested_merged).repeat(6),
    )
    context = cirq.TransformerContext(tags_to_ignore=["ignore"], deep=True)
    c_new = cirq.merge_single_qubit_gates_to_phased_x_and_z(c_orig, context=context)
    cirq.testing.assert_same_circuits(c_new, c_expected)


def _phxz(a: float, x: float, z: float):
    return cirq.PhasedXZGate(axis_phase_exponent=a, x_exponent=x, z_exponent=z)


def test_merge_single_qubit_gates_to_phxz():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(
        cirq.X(a),
        cirq.Y(b) ** 0.5,
        cirq.CZ(a, b),
        cirq.H(a),
        cirq.Z(a),
        cirq.measure(b, key="m"),
        cirq.H(a).with_classical_controls("m"),
    )
    assert_optimizes(
        optimized=cirq.merge_single_qubit_gates_to_phxz(c),
        expected=cirq.Circuit(
            _phxz(-1, 1, 0).on(a),
            _phxz(0.5, 0.5, 0).on(b),
            cirq.CZ(a, b),
            _phxz(-0.5, 0.5, 0).on(a),
            cirq.measure(b, key="m"),
            cirq.H(a).with_classical_controls("m"),
        ),
    )


def test_merge_single_qubit_gates_to_phxz_deep():
    a = cirq.NamedQubit("a")
    c_nested = cirq.FrozenCircuit(cirq.H(a), cirq.Z(a), cirq.H(a).with_tags("ignore"))
    c_nested_merged = cirq.FrozenCircuit(_phxz(-0.5, 0.5, 0).on(a), cirq.H(a).with_tags("ignore"))
    c_orig = cirq.Circuit(
        c_nested,
        cirq.CircuitOperation(c_nested).repeat(4).with_tags("ignore"),
        c_nested,
        cirq.CircuitOperation(c_nested).repeat(5).with_tags("preserve_tags"),
        c_nested,
        cirq.CircuitOperation(c_nested).repeat(6),
    )
    c_expected = cirq.Circuit(
        c_nested_merged,
        cirq.CircuitOperation(c_nested).repeat(4).with_tags("ignore"),
        c_nested_merged,
        cirq.CircuitOperation(c_nested_merged).repeat(5).with_tags("preserve_tags"),
        c_nested_merged,
        cirq.CircuitOperation(c_nested_merged).repeat(6),
    )
    context = cirq.TransformerContext(tags_to_ignore=["ignore"], deep=True)
    c_new = cirq.merge_single_qubit_gates_to_phxz(c_orig, context=context)
    cirq.testing.assert_same_circuits(c_new, c_expected)


def test_merge_single_qubit_moments_to_phxz():
    q = cirq.LineQubit.range(3)
    c_orig = cirq.Circuit(
        cirq.Moment(cirq.X.on_each(*q[:2])),
        cirq.Moment(cirq.T.on_each(*q[1:])),
        cirq.Moment(cirq.Y.on_each(*q[:2])),
        cirq.Moment(cirq.CZ(*q[:2]), cirq.Y(q[2])),
        cirq.Moment(cirq.X.on_each(*q[:2])),
        cirq.Moment(cirq.T.on_each(*q[1:])),
        cirq.Moment(cirq.Y.on_each(*q[:2])),
        cirq.Moment(cirq.Y(q[0]).with_tags("nocompile"), cirq.Z.on_each(*q[1:])),
        cirq.Moment(cirq.X.on_each(q[0])),
        cirq.Moment(cirq.measure(q[0], key="a")),
        cirq.Moment(cirq.X(q[1]).with_classical_controls("a")),
        cirq.Moment(cirq.X.on_each(q[1])),
    )
    cirq.testing.assert_has_diagram(
        c_orig,
        '''
0: ───X───────Y───@───X───────Y───Y['nocompile']───X───M───────────
                  │                                    ║
1: ───X───T───Y───@───X───T───Y───Z────────────────────╫───X───X───
                                                       ║   ║
2: ───────T───────Y───────T───────Z────────────────────╫───╫───────
                                                       ║   ║
a: ════════════════════════════════════════════════════@═══^═══════
''',
    )
    context = cirq.TransformerContext(tags_to_ignore=("nocompile",))
    c_new = cirq.merge_single_qubit_moments_to_phxz(c_orig, context=context)
    cirq.testing.assert_has_diagram(
        c_new,
        '''
0: ───PhXZ(a=-0.5,x=0,z=-1)──────@───PhXZ(a=-0.5,x=0,z=-1)──────Y['nocompile']───X───M───────────
                                 │                                                   ║
1: ───PhXZ(a=-0.25,x=0,z=0.75)───@───PhXZ(a=-0.25,x=0,z=0.75)───Z────────────────────╫───X───X───
                                                                                     ║   ║
2: ───PhXZ(a=0.25,x=0,z=0.25)────Y───PhXZ(a=0.25,x=0,z=0.25)────Z────────────────────╫───╫───────
                                                                                     ║   ║
a: ══════════════════════════════════════════════════════════════════════════════════@═══^═══════
''',
    )


def test_merge_single_qubit_moments_to_phxz_deep():
    q = cirq.LineQubit.range(3)
    x_t_y = cirq.FrozenCircuit(
        cirq.Moment(cirq.X.on_each(*q[:2])),
        cirq.Moment(cirq.T.on_each(*q[1:])),
        cirq.Moment(cirq.Y.on_each(*q[:2])),
    )
    c_nested = cirq.FrozenCircuit(
        x_t_y,
        cirq.Moment(cirq.CZ(*q[:2]), cirq.Y(q[2])),
        x_t_y,
        cirq.Moment(cirq.Y(q[0]).with_tags("ignore"), cirq.Z.on_each(*q[1:])),
    )

    c_nested_merged = cirq.FrozenCircuit(
        [_phxz(-0.25, 0.0, 0.75)(q[1]), _phxz(0.25, 0.0, 0.25)(q[2]), _phxz(-0.5, 0.0, -1.0)(q[0])],
        [cirq.CZ(q[0], q[1]), cirq.Y(q[2])],
        [_phxz(-0.25, 0.0, 0.75)(q[1]), _phxz(0.25, 0.0, 0.25)(q[2]), _phxz(-0.5, 0.0, -1.0)(q[0])],
        cirq.Moment(cirq.Y(q[0]).with_tags("ignore"), cirq.Z.on_each(*q[1:])),
    )
    c_orig = cirq.Circuit(
        c_nested,
        cirq.CircuitOperation(c_nested).repeat(4).with_tags("ignore"),
        c_nested,
        cirq.CircuitOperation(c_nested).repeat(5).with_tags("preserve_tags"),
        c_nested,
        cirq.CircuitOperation(c_nested).repeat(6),
    )
    c_expected = cirq.Circuit(
        c_nested_merged,
        cirq.CircuitOperation(c_nested).repeat(4).with_tags("ignore"),
        c_nested_merged,
        cirq.CircuitOperation(c_nested_merged).repeat(5).with_tags("preserve_tags"),
        c_nested_merged,
        cirq.CircuitOperation(c_nested_merged).repeat(6),
    )
    context = cirq.TransformerContext(tags_to_ignore=["ignore"], deep=True)
    c_new = cirq.merge_single_qubit_moments_to_phxz(c_orig, context=context)
    cirq.testing.assert_allclose_up_to_global_phase(
        c_new.unitary(), c_expected.unitary(), atol=1e-7
    )


def test_merge_single_qubit_moments_to_phxz_global_phase():
    c = cirq.Circuit(cirq.GlobalPhaseGate(1j).on())
    c2 = cirq.merge_single_qubit_gates_to_phxz(c)
    assert c == c2


def test_merge_single_qubit_moments_to_phased_x_and_z_global_phase():
    c = cirq.Circuit(cirq.GlobalPhaseGate(1j).on())
    c2 = cirq.merge_single_qubit_gates_to_phased_x_and_z(c)
    assert c == c2
