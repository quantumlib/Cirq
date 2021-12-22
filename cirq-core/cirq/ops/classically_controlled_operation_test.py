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
import pytest
import sympy
from sympy.parsing import sympy_parser

import cirq

ALL_SIMULATORS = (
    cirq.Simulator(),
    cirq.DensityMatrixSimulator(),
    cirq.CliffordSimulator(),
)


def test_diagram():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.measure(q0, key='a'), cirq.X(q1).with_classical_controls('a'))

    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───M───────
      ║
1: ───╫───X───
      ║   ║
a: ═══@═══^═══
""",
        use_unicode_characters=True,
    )


def test_diagram_pauli():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure_single_paulistring(cirq.X(q0), key='a'),
        cirq.X(q1).with_classical_controls('a'),
    )

    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───M(X)───────
      ║
1: ───╫──────X───
      ║      ║
a: ═══@══════^═══
""",
        use_unicode_characters=True,
    )


def test_diagram_extra_measurements():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a'),
        cirq.measure(q0, key='b'),
        cirq.X(q1).with_classical_controls('a'),
    )

    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───M───M('b')───
      ║
1: ───╫───X────────
      ║   ║
a: ═══@═══^════════
""",
        use_unicode_characters=True,
    )


def test_diagram_extra_controlled_bits():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a'),
        cirq.CX(q0, q1).with_classical_controls('a'),
    )

    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───M───@───
      ║   ║
1: ───╫───X───
      ║   ║
a: ═══@═══^═══
""",
        use_unicode_characters=True,
    )


def test_diagram_extra_control_bits():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a'),
        cirq.measure(q0, key='b'),
        cirq.X(q1).with_classical_controls('a', 'b'),
    )

    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───M───M───────
      ║   ║
1: ───╫───╫───X───
      ║   ║   ║
a: ═══@═══╬═══^═══
          ║   ║
b: ═══════@═══^═══
""",
        use_unicode_characters=True,
    )


def test_diagram_multiple_ops_single_moment():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a'),
        cirq.measure(q1, key='b'),
        cirq.X(q0).with_classical_controls('a'),
        cirq.X(q1).with_classical_controls('b'),
    )

    cirq.testing.assert_has_diagram(
        circuit,
        """
      ┌──┐   ┌──┐
0: ────M──────X─────
       ║      ║
1: ────╫M─────╫X────
       ║║     ║║
a: ════@╬═════^╬════
        ║      ║
b: ═════@══════^════
      └──┘   └──┘
""",
        use_unicode_characters=True,
    )


def test_diagram_subcircuit():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.CircuitOperation(
            cirq.FrozenCircuit(
                cirq.measure(q0, key='a'),
                cirq.X(q1).with_classical_controls('a'),
            )
        )
    )

    cirq.testing.assert_has_diagram(
        circuit,
        """
      [ 0: ───M─────── ]
      [       ║        ]
0: ───[ 1: ───╫───X─── ]───
      [       ║   ║    ]
      [ a: ═══@═══^═══ ]
      │
1: ───#2───────────────────
""",
        use_unicode_characters=True,
    )


def test_diagram_subcircuit_layered():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a'),
        cirq.CircuitOperation(
            cirq.FrozenCircuit(
                cirq.measure(q0, key='a'),
                cirq.X(q1).with_classical_controls('a'),
            ),
        ),
        cirq.X(q1).with_classical_controls('a'),
    )

    cirq.testing.assert_has_diagram(
        circuit,
        """
          [ 0: ───M─────── ]
          [       ║        ]
0: ───M───[ 1: ───╫───X─── ]───────
      ║   [       ║   ║    ]
      ║   [ a: ═══@═══^═══ ]
      ║   ║
1: ───╫───#2───────────────────X───
      ║   ║                    ║
a: ═══@═══╩════════════════════^═══
""",
        use_unicode_characters=True,
    )


def test_qasm():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.measure(q0, key='a'), cirq.X(q1).with_classical_controls('a'))
    qasm = cirq.qasm(circuit)
    assert (
        qasm
        == """// Generated from Cirq v0.14.0.dev

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [0, 1]
qreg q[2];
creg m_a[1];


measure q[0] -> m_a[0];
if (m_a!=0) x q[1];
"""
    )


@pytest.mark.parametrize('sim', ALL_SIMULATORS)
def test_key_unset(sim):
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a'),
        cirq.X(q1).with_classical_controls('a'),
        cirq.measure(q1, key='b'),
    )
    result = sim.run(circuit)
    assert result.measurements['a'] == 0
    assert result.measurements['b'] == 0


@pytest.mark.parametrize('sim', ALL_SIMULATORS)
def test_key_set(sim):
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.X(q0),
        cirq.measure(q0, key='a'),
        cirq.X(q1).with_classical_controls('a'),
        cirq.measure(q1, key='b'),
    )
    result = sim.run(circuit)
    assert result.measurements['a'] == 1
    assert result.measurements['b'] == 1


@pytest.mark.parametrize('sim', ALL_SIMULATORS)
def test_subcircuit_key_unset(sim):
    q0, q1 = cirq.LineQubit.range(2)
    inner = cirq.Circuit(
        cirq.measure(q0, key='c'),
        cirq.X(q1).with_classical_controls('c'),
        cirq.measure(q1, key='b'),
    )
    circuit = cirq.Circuit(
        cirq.CircuitOperation(inner.freeze(), repetitions=2, measurement_key_map={'c': 'a'})
    )
    result = sim.run(circuit)
    assert result.measurements['0:a'] == 0
    assert result.measurements['0:b'] == 0
    assert result.measurements['1:a'] == 0
    assert result.measurements['1:b'] == 0


@pytest.mark.parametrize('sim', ALL_SIMULATORS)
def test_subcircuit_key_set(sim):
    q0, q1 = cirq.LineQubit.range(2)
    inner = cirq.Circuit(
        cirq.X(q0),
        cirq.measure(q0, key='c'),
        cirq.X(q1).with_classical_controls('c'),
        cirq.measure(q1, key='b'),
    )
    circuit = cirq.Circuit(
        cirq.CircuitOperation(inner.freeze(), repetitions=4, measurement_key_map={'c': 'a'})
    )
    result = sim.run(circuit)
    assert result.measurements['0:a'] == 1
    assert result.measurements['0:b'] == 1
    assert result.measurements['1:a'] == 0
    assert result.measurements['1:b'] == 1
    assert result.measurements['2:a'] == 1
    assert result.measurements['2:b'] == 0
    assert result.measurements['3:a'] == 0
    assert result.measurements['3:b'] == 0


def test_key_unset_in_subcircuit_outer_scope():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a'),
    )
    # TODO (daxfohl): This will not need an InsertStrategy after scope PR.
    circuit.append(
        cirq.CircuitOperation(cirq.FrozenCircuit(cirq.X(q1).with_classical_controls('a'))),
        strategy=cirq.InsertStrategy.NEW,
    )
    circuit.append(cirq.measure(q1, key='b'))
    result = cirq.Simulator().run(circuit)
    assert result.measurements['a'] == 0
    assert result.measurements['b'] == 0


def test_key_set_in_subcircuit_outer_scope():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.X(q0),
        cirq.measure(q0, key='a'),
    )
    # TODO (daxfohl): This will not need an InsertStrategy after scope PR.
    circuit.append(
        cirq.CircuitOperation(cirq.FrozenCircuit(cirq.X(q1).with_classical_controls('a'))),
        strategy=cirq.InsertStrategy.NEW,
    )
    circuit.append(cirq.measure(q1, key='b'))
    result = cirq.Simulator().run(circuit)
    assert result.measurements['a'] == 1
    assert result.measurements['b'] == 1


def test_condition_types():
    q0 = cirq.LineQubit(0)
    sympy_cond = sympy_parser.parse_expr('a >= 2')
    op = cirq.X(q0).with_classical_controls(cirq.MeasurementKey('a'), 'b', 'a > b', sympy_cond)
    assert set(map(str, op.classical_controls)) == {'a', 'b', 'a > b', 'a >= 2'}


def test_condition_flattening():
    q0 = cirq.LineQubit(0)
    op = cirq.X(q0).with_classical_controls('a').with_classical_controls('b')
    assert set(map(str, op.classical_controls)) == {'a', 'b'}
    assert isinstance(op._sub_operation, cirq.GateOperation)


def test_condition_stacking():
    q0 = cirq.LineQubit(0)
    op = cirq.X(q0).with_classical_controls('a').with_tags('t').with_classical_controls('b')
    assert set(map(str, cirq.control_keys(op))) == {'a', 'b'}
    assert set(map(str, op.classical_controls)) == {'a', 'b'}
    assert not op.tags


def test_condition_removal():
    q0 = cirq.LineQubit(0)
    op = (
        cirq.X(q0)
        .with_tags('t1')
        .with_classical_controls('a')
        .with_tags('t2')
        .with_classical_controls('b')
    )
    op = op.without_classical_controls()
    assert not cirq.control_keys(op)
    assert not op.classical_controls
    assert set(map(str, op.tags)) == {'t1'}


def test_qubit_mapping():
    q0, q1 = cirq.LineQubit.range(2)
    op = cirq.X(q0).with_classical_controls('a')
    assert op.with_qubits(q1).qubits == (q1,)


def test_parameterizable():
    s = sympy.Symbol('s')
    q0 = cirq.LineQubit(0)
    op = cirq.X(q0).with_classical_controls('a')
    opa = cirq.XPowGate(exponent=s).on(q0).with_classical_controls('a')
    assert cirq.is_parameterized(opa)
    assert not cirq.is_parameterized(op)
    assert cirq.resolve_parameters(opa, cirq.ParamResolver({'s': 1})) == op


def test_decompose():
    q0 = cirq.LineQubit(0)
    op = cirq.H(q0).with_classical_controls('a')
    assert cirq.decompose(op) == [
        (cirq.Y(q0) ** 0.5).with_classical_controls('a'),
        cirq.XPowGate(exponent=1.0, global_shift=-0.25).on(q0).with_classical_controls('a'),
    ]


def test_str():
    q0 = cirq.LineQubit(0)
    op = cirq.X(q0).with_classical_controls('a')
    assert str(op) == 'X(0).with_classical_controls(a)'


def test_scope_local():
    q = cirq.LineQubit(0)
    inner = cirq.Circuit(
        cirq.measure(q, key='a'),
        cirq.X(q).with_classical_controls('a'),
    )
    middle = cirq.Circuit(cirq.CircuitOperation(inner.freeze(), repetitions=2))
    outer_subcircuit = cirq.CircuitOperation(middle.freeze(), repetitions=2)
    circuit = outer_subcircuit.mapped_circuit(deep=True)
    internal_control_keys = [
        str(condition) for op in circuit.all_operations() for condition in cirq.control_keys(op)
    ]
    assert internal_control_keys == ['0:0:a', '0:1:a', '1:0:a', '1:1:a']
    assert not cirq.control_keys(outer_subcircuit)
    assert not cirq.control_keys(circuit)
    cirq.testing.assert_has_diagram(
        cirq.Circuit(outer_subcircuit),
        """
      [       [ 0: ───M───X─── ]             ]
0: ───[ 0: ───[       ║   ║    ]──────────── ]────────────
      [       [ a: ═══@═══^═══ ](loops=2)    ](loops=2)
""",
        use_unicode_characters=True,
    )
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───────M───X───M───X───M───X───M───X───
          ║   ║   ║   ║   ║   ║   ║   ║
0:0:a: ═══@═══^═══╬═══╬═══╬═══╬═══╬═══╬═══
                  ║   ║   ║   ║   ║   ║
0:1:a: ═══════════@═══^═══╬═══╬═══╬═══╬═══
                          ║   ║   ║   ║
1:0:a: ═══════════════════@═══^═══╬═══╬═══
                                  ║   ║
1:1:a: ═══════════════════════════@═══^═══
""",
        use_unicode_characters=True,
    )
    assert circuit == cirq.Circuit(cirq.decompose(outer_subcircuit))


def test_scope_extern():
    q = cirq.LineQubit(0)
    inner = cirq.Circuit(
        cirq.measure(q, key='a'),
        cirq.X(q).with_classical_controls('b'),
    )
    middle = cirq.Circuit(
        cirq.measure(q, key=cirq.MeasurementKey('b')),
        cirq.CircuitOperation(inner.freeze(), repetitions=2),
    )
    outer_subcircuit = cirq.CircuitOperation(middle.freeze(), repetitions=2)
    circuit = outer_subcircuit.mapped_circuit(deep=True)
    internal_control_keys = [
        str(condition) for op in circuit.all_operations() for condition in cirq.control_keys(op)
    ]
    assert internal_control_keys == ['0:b', '0:b', '1:b', '1:b']
    assert not cirq.control_keys(outer_subcircuit)
    assert not cirq.control_keys(circuit)
    cirq.testing.assert_has_diagram(
        cirq.Circuit(outer_subcircuit),
        """
      [           [ 0: ───M('a')───X─── ]             ]
      [ 0: ───M───[                ║    ]──────────── ]
0: ───[       ║   [ b: ════════════^═══ ](loops=2)    ]────────────
      [       ║   ║                                   ]
      [ b: ═══@═══╩══════════════════════════════════ ](loops=2)
""",
        use_unicode_characters=True,
    )
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ─────M───M('0:0:a')───X───M('0:1:a')───X───M───M('1:0:a')───X───M('1:1:a')───X───
        ║                ║                ║   ║                ║                ║
0:b: ═══@════════════════^════════════════^═══╬════════════════╬════════════════╬═══
                                              ║                ║                ║
1:b: ═════════════════════════════════════════@════════════════^════════════════^═══
""",
        use_unicode_characters=True,
    )
    assert circuit == cirq.Circuit(cirq.decompose(outer_subcircuit))


def test_scope_extern_wrapping_with_non_repeating_subcircuits():
    def wrap(*ops):
        return cirq.CircuitOperation(cirq.FrozenCircuit(*ops))

    def wrap_frozen(*ops):
        return cirq.FrozenCircuit(wrap(*ops))

    q = cirq.LineQubit(0)
    inner = wrap_frozen(
        wrap(cirq.measure(q, key='a')),
        wrap(cirq.X(q).with_classical_controls('b')),
    )
    middle = wrap_frozen(
        wrap(cirq.measure(q, key=cirq.MeasurementKey('b'))),
        wrap(cirq.CircuitOperation(inner, repetitions=2)),
    )
    outer_subcircuit = cirq.CircuitOperation(middle, repetitions=2)
    circuit = outer_subcircuit.mapped_circuit(deep=True)
    internal_control_keys = [
        str(condition) for op in circuit.all_operations() for condition in cirq.control_keys(op)
    ]
    assert internal_control_keys == ['0:b', '0:b', '1:b', '1:b']
    assert not cirq.control_keys(outer_subcircuit)
    assert not cirq.control_keys(circuit)
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ─────M───M('0:0:a')───X───M('0:1:a')───X───M───M('1:0:a')───X───M('1:1:a')───X───
        ║                ║                ║   ║                ║                ║
0:b: ═══@════════════════^════════════════^═══╬════════════════╬════════════════╬═══
                                              ║                ║                ║
1:b: ═════════════════════════════════════════@════════════════^════════════════^═══
""",
        use_unicode_characters=True,
    )
    assert circuit == cirq.Circuit(cirq.decompose(outer_subcircuit))


def test_scope_root():
    q = cirq.LineQubit(0)
    inner = cirq.Circuit(
        cirq.measure(q, key='a'),
        cirq.X(q).with_classical_controls('b'),
    )
    middle = cirq.Circuit(
        cirq.measure(q, key=cirq.MeasurementKey('c')),
        cirq.CircuitOperation(inner.freeze(), repetitions=2),
    )
    outer_subcircuit = cirq.CircuitOperation(middle.freeze(), repetitions=2)
    circuit = outer_subcircuit.mapped_circuit(deep=True)
    internal_control_keys = [
        str(condition) for op in circuit.all_operations() for condition in cirq.control_keys(op)
    ]
    assert internal_control_keys == ['b', 'b', 'b', 'b']
    assert cirq.control_keys(outer_subcircuit) == {cirq.MeasurementKey('b')}
    assert cirq.control_keys(circuit) == {cirq.MeasurementKey('b')}
    cirq.testing.assert_has_diagram(
        cirq.Circuit(outer_subcircuit),
        """
      [                [ 0: ───M('a')───X─── ]             ]
      [ 0: ───M('c')───[                ║    ]──────────── ]
0: ───[                [ b: ════════════^═══ ](loops=2)    ]────────────
      [                ║                                   ]
      [ b: ════════════╩══════════════════════════════════ ](loops=2)
      ║
b: ═══╩═════════════════════════════════════════════════════════════════
""",
        use_unicode_characters=True,
    )
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───M('0:c')───M('0:0:a')───X───M('0:1:a')───X───M('1:c')───M('1:0:a')───X───M('1:1:a')───X───
                              ║                ║                           ║                ║
b: ═══════════════════════════^════════════════^═══════════════════════════^════════════════^═══
""",
        use_unicode_characters=True,
    )
    assert circuit == cirq.Circuit(cirq.decompose(outer_subcircuit))


def test_scope_extern_mismatch():
    q = cirq.LineQubit(0)
    inner = cirq.Circuit(
        cirq.measure(q, key='a'),
        cirq.X(q).with_classical_controls('b'),
    )
    middle = cirq.Circuit(
        cirq.measure(q, key=cirq.MeasurementKey('b', ('0',))),
        cirq.CircuitOperation(inner.freeze(), repetitions=2),
    )
    outer_subcircuit = cirq.CircuitOperation(middle.freeze(), repetitions=2)
    circuit = outer_subcircuit.mapped_circuit(deep=True)
    internal_control_keys = [
        str(condition) for op in circuit.all_operations() for condition in cirq.control_keys(op)
    ]
    assert internal_control_keys == ['b', 'b', 'b', 'b']
    assert cirq.control_keys(outer_subcircuit) == {cirq.MeasurementKey('b')}
    assert cirq.control_keys(circuit) == {cirq.MeasurementKey('b')}
    cirq.testing.assert_has_diagram(
        cirq.Circuit(outer_subcircuit),
        """
      [                  [ 0: ───M('a')───X─── ]             ]
      [ 0: ───M('0:b')───[                ║    ]──────────── ]
0: ───[                  [ b: ════════════^═══ ](loops=2)    ]────────────
      [                  ║                                   ]
      [ b: ══════════════╩══════════════════════════════════ ](loops=2)
      ║
b: ═══╩═══════════════════════════════════════════════════════════════════
""",
        use_unicode_characters=True,
    )
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───M('0:0:b')───M('0:0:a')───X───M('0:1:a')───X───M('1:0:b')───M('1:0:a')───X───M('1:1:a')───X───
                                ║                ║                             ║                ║
b: ═════════════════════════════^════════════════^═════════════════════════════^════════════════^═══
""",
        use_unicode_characters=True,
    )
    assert circuit == cirq.Circuit(cirq.decompose(outer_subcircuit))


def test_repr():
    q0 = cirq.LineQubit(0)
    op = cirq.X(q0).with_classical_controls('a')
    assert repr(op) == (
        "cirq.ClassicallyControlledOperation("
        "cirq.X(cirq.LineQubit(0)), [cirq.KeyCondition(cirq.MeasurementKey(name='a'))]"
        ")"
    )


def test_no_measurement_gates():
    q0 = cirq.LineQubit(0)
    with pytest.raises(ValueError, match='with measurements'):
        _ = cirq.measure(q0).with_classical_controls('a')


def test_unmeasured_condition():
    q0 = cirq.LineQubit(0)
    bad_circuit = cirq.Circuit(cirq.X(q0).with_classical_controls('a'))
    with pytest.raises(
        ValueError, match='Measurement key a missing when testing classical control'
    ):
        _ = cirq.Simulator().simulate(bad_circuit)


def test_layered_circuit_operations_with_controls_in_between():
    q = cirq.LineQubit(0)
    outer_subcircuit = cirq.CircuitOperation(
        cirq.Circuit(
            cirq.CircuitOperation(
                cirq.FrozenCircuit(
                    cirq.X(q),
                    cirq.Y(q),
                )
            ).with_classical_controls('m')
        ).freeze()
    )
    circuit = outer_subcircuit.mapped_circuit(deep=True)
    cirq.testing.assert_has_diagram(
        cirq.Circuit(outer_subcircuit),
        """
      [ 0: ───[ 0: ───X───Y─── ].with_classical_controls(m)─── ]
0: ───[       ║                                                ]───
      [ m: ═══╩═══════════════════════════════════════════════ ]
      ║
m: ═══╩════════════════════════════════════════════════════════════
""",
        use_unicode_characters=True,
    )
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───[ 0: ───X───Y─── ].with_classical_controls(m)───
      ║
m: ═══╩═══════════════════════════════════════════════
""",
        use_unicode_characters=True,
    )
    cirq.testing.assert_has_diagram(
        cirq.Circuit(cirq.decompose(outer_subcircuit)),
        """
0: ───X───Y───
      ║   ║
m: ═══^═══^═══
""",
        use_unicode_characters=True,
    )


def test_sympy():
    q0, q1, q2, q3, q_result = cirq.LineQubit.range(5)
    for i in range(4):
        for j in range(4):
            # Put first two qubits into a state representing bitstring(i), next two qubits into a
            # state representing bitstring(j) and measure those into m_i and m_j respectively. Then
            # add a conditional X(q_result) based on m_i > m_j and measure that.
            bitstring_i = cirq.big_endian_int_to_bits(i, bit_count=2)
            bitstring_j = cirq.big_endian_int_to_bits(j, bit_count=2)
            circuit = cirq.Circuit(
                cirq.X(q0) ** bitstring_i[0],
                cirq.X(q1) ** bitstring_i[1],
                cirq.X(q2) ** bitstring_j[0],
                cirq.X(q3) ** bitstring_j[1],
                cirq.measure(q0, q1, key='m_i'),
                cirq.measure(q2, q3, key='m_j'),
                cirq.X(q_result).with_classical_controls(sympy_parser.parse_expr('m_j > m_i')),
                cirq.measure(q_result, key='m_result'),
            )

            # m_result should now be set iff j > i.
            result = cirq.Simulator().run(circuit)
            assert result.measurements['m_result'][0][0] == (j > i)


def test_sympy_path_prefix():
    q = cirq.LineQubit(0)
    op = cirq.X(q).with_classical_controls(sympy.Symbol('b'))
    prefixed = cirq.with_key_path_prefix(op, ('0',))
    assert cirq.control_keys(prefixed) == {'0:b'}


def test_sympy_scope():
    q = cirq.LineQubit(0)
    a, b, c, d = sympy.symbols('a b c d')
    inner = cirq.Circuit(
        cirq.measure(q, key='a'),
        cirq.X(q).with_classical_controls(a & b).with_classical_controls(c | d),
    )
    middle = cirq.Circuit(
        cirq.measure(q, key='b'),
        cirq.measure(q, key=cirq.MeasurementKey('c', ('0',))),
        cirq.CircuitOperation(inner.freeze(), repetitions=2),
    )
    outer_subcircuit = cirq.CircuitOperation(middle.freeze(), repetitions=2)
    circuit = outer_subcircuit.mapped_circuit(deep=True)
    internal_controls = [str(k) for op in circuit.all_operations() for k in cirq.control_keys(op)]
    assert set(internal_controls) == {'0:0:a', '0:1:a', '1:0:a', '1:1:a', '0:b', '1:b', 'c', 'd'}
    assert cirq.control_keys(outer_subcircuit) == {'c', 'd'}
    assert cirq.control_keys(circuit) == {'c', 'd'}
    assert circuit == cirq.Circuit(cirq.decompose(outer_subcircuit))
    cirq.testing.assert_has_diagram(
        cirq.Circuit(outer_subcircuit),
        """
      [                      [ 0: ───M───X(conditions=[c | d, a & b])─── ]             ]
      [                      [       ║   ║                               ]             ]
      [                      [ a: ═══@═══^══════════════════════════════ ]             ]
      [                      [           ║                               ]             ]
      [ 0: ───M───M('0:c')───[ b: ═══════^══════════════════════════════ ]──────────── ]
      [       ║              [           ║                               ]             ]
      [       ║              [ c: ═══════^══════════════════════════════ ]             ]
0: ───[       ║              [           ║                               ]             ]────────────
      [       ║              [ d: ═══════^══════════════════════════════ ](loops=2)    ]
      [       ║              ║                                                         ]
      [ b: ═══@══════════════╬════════════════════════════════════════════════════════ ]
      [                      ║                                                         ]
      [ c: ══════════════════╬════════════════════════════════════════════════════════ ]
      [                      ║                                                         ]
      [ d: ══════════════════╩════════════════════════════════════════════════════════ ](loops=2)
      ║
c: ═══╬═════════════════════════════════════════════════════════════════════════════════════════════
      ║
d: ═══╩═════════════════════════════════════════════════════════════════════════════════════════════
""",
        use_unicode_characters=True,
    )

    # pylint: disable=line-too-long
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───────M───M('0:0:c')───M───X(conditions=[c | d, 0:0:a & 0:b])───M───X(conditions=[c | d, 0:1:a & 0:b])───M───M('1:0:c')───M───X(conditions=[c | d, 1:0:a & 1:b])───M───X(conditions=[c | d, 1:1:a & 1:b])───
          ║                ║   ║                                    ║   ║                                    ║                ║   ║                                    ║   ║
0:0:a: ═══╬════════════════@═══^════════════════════════════════════╬═══╬════════════════════════════════════╬════════════════╬═══╬════════════════════════════════════╬═══╬════════════════════════════════════
          ║                    ║                                    ║   ║                                    ║                ║   ║                                    ║   ║
0:1:a: ═══╬════════════════════╬════════════════════════════════════@═══^════════════════════════════════════╬════════════════╬═══╬════════════════════════════════════╬═══╬════════════════════════════════════
          ║                    ║                                        ║                                    ║                ║   ║                                    ║   ║
0:b: ═════@════════════════════^════════════════════════════════════════^════════════════════════════════════╬════════════════╬═══╬════════════════════════════════════╬═══╬════════════════════════════════════
                               ║                                        ║                                    ║                ║   ║                                    ║   ║
1:0:a: ════════════════════════╬════════════════════════════════════════╬════════════════════════════════════╬════════════════@═══^════════════════════════════════════╬═══╬════════════════════════════════════
                               ║                                        ║                                    ║                    ║                                    ║   ║
1:1:a: ════════════════════════╬════════════════════════════════════════╬════════════════════════════════════╬════════════════════╬════════════════════════════════════@═══^════════════════════════════════════
                               ║                                        ║                                    ║                    ║                                        ║
1:b: ══════════════════════════╬════════════════════════════════════════╬════════════════════════════════════@════════════════════^════════════════════════════════════════^════════════════════════════════════
                               ║                                        ║                                                         ║                                        ║
c: ════════════════════════════^════════════════════════════════════════^═════════════════════════════════════════════════════════^════════════════════════════════════════^════════════════════════════════════
                               ║                                        ║                                                         ║                                        ║
d: ════════════════════════════^════════════════════════════════════════^═════════════════════════════════════════════════════════^════════════════════════════════════════^════════════════════════════════════
""",
        use_unicode_characters=True,
    )
    # pylint: enable=line-too-long


def test_sympy_scope_simulation():
    q0, q1, q2, q3, q_ignored, q_result = cirq.LineQubit.range(6)
    condition = sympy_parser.parse_expr('a & b | c & d')
    # We set up condition (a & b | c & d) plus an ignored measurement key, and run through the
    # combinations of possible values of those (by doing X(q_i)**bits[i] on each), then verify
    # that the final measurement into m_result is True iff that condition was met.
    for i in range(32):
        bits = cirq.big_endian_int_to_bits(i, bit_count=5)
        inner = cirq.Circuit(
            cirq.X(q0) ** bits[0],
            cirq.measure(q0, key='a'),
            cirq.X(q_result).with_classical_controls(condition),
            cirq.measure(q_result, key='m_result'),
        )
        middle = cirq.Circuit(
            cirq.X(q1) ** bits[1],
            cirq.measure(q1, key='b'),
            cirq.X(q_ignored) ** bits[4],
            cirq.measure(q_ignored, key=cirq.MeasurementKey('c', ('0',))),
            cirq.CircuitOperation(inner.freeze(), repetition_ids=['0']),
        )
        circuit = cirq.Circuit(
            cirq.X(q2) ** bits[2],
            cirq.measure(q2, key='c'),
            cirq.X(q3) ** bits[3],
            cirq.measure(q3, key='d'),
            cirq.CircuitOperation(middle.freeze(), repetition_ids=['0']),
        )
        result = cirq.CliffordSimulator().run(circuit)
        assert result.measurements['0:0:m_result'][0][0] == (
            bits[0] and bits[1] or bits[2] and bits[3]  # bits[4] irrelevant
        )
