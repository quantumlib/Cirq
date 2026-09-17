# Copyright 2026 The Cirq Developers
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

import numpy as np
import pytest
import sympy

import cirq


def test_init() -> None:
    q = cirq.LineQubit(0)
    op = cirq.While('m', cirq.X(q))
    assert op.conditions == (cirq.KeyCondition(cirq.MeasurementKey('m')),)
    assert op.sub_operation == cirq.X(q)
    assert op.qubits == (q,)
    assert op.classical_controls == frozenset([cirq.KeyCondition(cirq.MeasurementKey('m'))])
    assert op.without_classical_controls() == cirq.X(q)


def test_init_condition_types() -> None:
    q = cirq.LineQubit(0)
    key = cirq.MeasurementKey('k')
    cond = cirq.KeyCondition(key)
    sym = sympy.Symbol('s')

    assert cirq.While(key, cirq.X(q)).conditions == (cond,)
    assert cirq.While(cond, cirq.X(q)).conditions == (cond,)
    assert cirq.While(sym, cirq.X(q)).conditions == (cirq.SympyCondition(sym),)


def test_init_multiple_conditions() -> None:
    q = cirq.LineQubit(0)
    op = cirq.While(['a', 'b'], cirq.X(q))
    assert op.conditions == (
        cirq.KeyCondition(cirq.MeasurementKey('a')),
        cirq.KeyCondition(cirq.MeasurementKey('b')),
    )


def test_init_multiple_operations() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    op = cirq.While('m', cirq.X(q0), cirq.Y(q1))
    assert isinstance(op.sub_operation, cirq.CircuitOperation)
    assert op.sub_operation.circuit == cirq.Circuit(cirq.X(q0), cirq.Y(q1))
    assert op.qubits == (q0, q1)

    op_list = cirq.While('m', [cirq.X(q0), cirq.Y(q1)])
    assert isinstance(op_list.sub_operation, cirq.CircuitOperation)
    assert op_list.sub_operation.circuit == cirq.Circuit(cirq.X(q0), cirq.Y(q1))


def test_init_squash_nested_while() -> None:
    q = cirq.LineQubit(0)
    inner = cirq.While('a', cirq.X(q))
    outer = cirq.While('b', inner)
    assert outer.conditions == (
        cirq.KeyCondition(cirq.MeasurementKey('b')),
        cirq.KeyCondition(cirq.MeasurementKey('a')),
    )
    assert outer.sub_operation == cirq.X(q)

    cco = cirq.ClassicallyControlledOperation(cirq.X(q), ['a'])
    outer_cco = cirq.While('b', cco)
    assert outer_cco.conditions == (
        cirq.KeyCondition(cirq.MeasurementKey('b')),
        cirq.KeyCondition(cirq.MeasurementKey('a')),
    )
    assert outer_cco.sub_operation == cirq.X(q)


def test_init_errors() -> None:
    q = cirq.LineQubit(0)
    with pytest.raises(TypeError, match='Unrecognized condition type'):
        _ = cirq.While(123, cirq.X(q))

    with pytest.raises(TypeError, match='Unrecognized condition type'):
        _ = cirq.While(['a', 123], cirq.X(q))

    with pytest.raises(ValueError, match='At least one condition must be provided'):
        _ = cirq.While([], cirq.X(q))


def test_with_qubits() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    op = cirq.While('m', cirq.X(q0))
    new_op = op.with_qubits(q1)
    assert new_op == cirq.While('m', cirq.X(q1))


def test_value_equality() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(
        cirq.While('a', cirq.X(q0)), cirq.While(cirq.MeasurementKey('a'), cirq.X(q0))
    )
    eq.add_equality_group(cirq.While('b', cirq.X(q0)))
    eq.add_equality_group(cirq.While('a', cirq.X(q1)))
    eq.add_equality_group(cirq.While(['a', 'b'], cirq.X(q0)))


def test_str_and_repr() -> None:
    q = cirq.LineQubit(0)
    op1 = cirq.While('m', cirq.X(q))
    assert str(op1) == 'While(m, X(q(0)))'
    assert repr(op1) == (
        "cirq.While(cirq.KeyCondition(cirq.MeasurementKey(name='m')), cirq.X(cirq.LineQubit(0)))"
    )
    assert eval(repr(op1)) == op1

    op2 = cirq.While(['a', 'b'], cirq.X(q))
    assert str(op2) == 'While([a, b], X(q(0)))'
    assert repr(op2) == (
        "cirq.While([cirq.KeyCondition(cirq.MeasurementKey(name='a')), "
        "cirq.KeyCondition(cirq.MeasurementKey(name='b'))], cirq.X(cirq.LineQubit(0)))"
    )
    assert eval(repr(op2)) == op2


def test_parameterized_and_resolve() -> None:
    q = cirq.LineQubit(0)
    sym = sympy.Symbol('theta')
    cond_sym = sympy.Symbol('cond')

    op = cirq.While(cond_sym, cirq.Rx(rads=sym).on(q))
    assert cirq.is_parameterized(op)
    assert cirq.parameter_names(op) == {'theta'}

    resolved = cirq.resolve_parameters(op, cirq.ParamResolver({'theta': np.pi, 'cond': 1}))
    assert not cirq.is_parameterized(resolved)
    assert resolved == cirq.While(cond_sym, cirq.Rx(rads=np.pi).on(q))


def test_diagram() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.measure(q0, key='a'), cirq.While('a', cirq.X(q1)))
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


def test_diagram_multiple_conditions() -> None:
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a'), cirq.measure(q1, key='b'), cirq.While(['a', 'b'], cirq.X(q2))
    )
    cirq.testing.assert_has_diagram(
        circuit,
        """
      ┌──┐
0: ────M─────────
       ║
1: ────╫M────────
       ║║
2: ────╫╫────X───
       ║║    ║
a: ════@╬════^═══
        ║    ║
b: ═════@════^═══
      └──┘
""",
        use_unicode_characters=True,
    )


def test_diagram_sympy_condition() -> None:
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.While(sympy.Symbol('s'), cirq.X(q)))
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───X(While=s)───
      ║
s: ═══^════════════
""",
        use_unicode_characters=True,
    )


def test_diagram_multiple_sympy_conditions() -> None:
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.While([sympy.Symbol('s'), sympy.Symbol('t')], cirq.X(q)))
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───X(While=s, t)───
      ║
s: ═══^═══════════════
      ║
t: ═══^═══════════════
""",
        use_unicode_characters=True,
    )


def test_simulation_countdown() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    # Start in |11> (value 3). Each loop iteration decrements the 2-qubit integer by 1:
    # 3 (11) -> 2 (10) -> 1 (01) -> 0 (00).
    circuit = cirq.Circuit(
        cirq.X(q0),
        cirq.X(q1),
        cirq.measure(q0, q1, key='a'),
        cirq.While('a', cirq.X(q1), cirq.CNOT(q1, q0), cirq.measure(q0, q1, key='a')),
    )
    sim = cirq.Simulator()
    res = sim.simulate(circuit)

    # Final measurement is [0, 0] and state is |00>
    assert list(res.measurements['a']) == [0, 0]
    np.testing.assert_equal(res.state_vector(), [1, 0, 0, 0])

    # Verify the loop ran 3 times (initial measurement + 3 loop measurements)
    assert res._final_simulator_state.classical_data.records[cirq.MeasurementKey('a')] == [
        (1, 1),
        (1, 0),
        (0, 1),
        (0, 0),
    ]


def test_simulation_repeat_until_success() -> None:
    q0 = cirq.LineQubit(0)
    # Repeat-until-success: apply H and measure until we observe |0>.
    circuit = cirq.Circuit(
        cirq.X(q0),
        cirq.measure(q0, key='a'),
        cirq.While('a', cirq.H(q0), cirq.measure(q0, key='a')),
    )
    # With seed=1, the coin tosses measure 1 several times before landing on 0.
    sim = cirq.Simulator(seed=1)
    res = sim.simulate(circuit)

    assert list(res.measurements['a']) == [0]
    cirq.testing.assert_allclose_up_to_global_phase(res.state_vector(), np.array([1, 0]), atol=1e-6)

    # Every recorded measurement before the last one must be (1,), and the last is (0,)
    records = res._final_simulator_state.classical_data.records[cirq.MeasurementKey('a')]
    assert len(records) > 2
    assert all(r == (1,) for r in records[:-1])
    assert records[-1] == (0,)


def test_key_mappings_and_scoping() -> None:
    q = cirq.LineQubit(0)
    op = cirq.While('a', cirq.X(q))

    mapped = cirq.with_measurement_key_mapping(op, {'a': 'b'})
    assert mapped == cirq.While('b', cirq.X(q))

    prefixed = cirq.with_key_path_prefix(op, ('path',))
    assert prefixed == cirq.While('path:a', cirq.X(q))

    rescoped = cirq.with_rescoped_keys(
        op, ('scope',), frozenset([cirq.MeasurementKey.parse_serialized('scope:a')])
    )
    assert rescoped == cirq.While('scope:a', cirq.X(q))

    assert cirq.control_keys(op) == frozenset([cirq.MeasurementKey('a')])


def test_has_unitary() -> None:
    q = cirq.LineQubit(0)
    assert not cirq.has_unitary(cirq.While('m', cirq.X(q)))


def test_qasm() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    op = cirq.While('a', cirq.X(q1))
    circuit = cirq.Circuit(cirq.measure(q0, key='a'), op)
    with pytest.raises(ValueError, match='QASM 2.0 does not support while loops'):
        _ = cirq.qasm(op)
    with pytest.raises(ValueError, match='QASM 2.0 does not support while loops'):
        _ = cirq.qasm(circuit)

    qasm_str = cirq.qasm(circuit, args=cirq.QasmArgs(version='3.0'))
    assert 'while (m_a!=0) x q[1];' in qasm_str

    op_multi = cirq.While(['a', 'b'], cirq.X(q1))
    circuit_multi = cirq.Circuit(cirq.measure(q0, key='a'), cirq.measure(q0, key='b'), op_multi)
    qasm_str_3 = cirq.qasm(circuit_multi, args=cirq.QasmArgs(version='3.0'))
    assert 'while (m_a!=0 && m_b!=0) x q[1];' in qasm_str_3

    op_block = cirq.While('a', cirq.X(q0), cirq.measure(q0, key='a'))
    circuit_block = cirq.Circuit(cirq.measure(q0, key='a'), op_block)
    qasm_block = cirq.qasm(circuit_block, args=cirq.QasmArgs(version='3.0'))
    assert 'while (m_a!=0) {\n  x q[0];\n  m_a[0] = measure q[0];\n}' in qasm_block


def test_qasm_sub_op_no_qasm() -> None:
    class NoQasmOp(cirq.Operation):
        @property
        def qubits(self):
            return (cirq.LineQubit(0),)  # pragma: nocover

        def with_qubits(self, *new_qubits):
            return self  # pragma: nocover

    op = cirq.While('a', NoQasmOp())
    assert cirq.qasm(op, args=cirq.QasmArgs(version='3.0'), default=None) is None

    op_block = cirq.While('a', NoQasmOp(), NoQasmOp())
    assert cirq.qasm(op_block, args=cirq.QasmArgs(version='3.0'), default=None) is None
