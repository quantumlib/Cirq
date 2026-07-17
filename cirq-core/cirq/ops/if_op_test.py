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
    op = cirq.If('m', cirq.X(q))
    assert op.condition == cirq.KeyCondition(cirq.MeasurementKey('m'))
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

    assert cirq.If(key, cirq.X(q)).condition == cond
    assert cirq.If(cond, cirq.X(q)).condition == cond
    assert cirq.If(sym, cirq.X(q)).condition == cirq.SympyCondition(sym)


def test_init_multiple_conditions() -> None:
    q = cirq.LineQubit(0)
    op = cirq.If(['a', 'b'], cirq.X(q))
    assert op.conditions == (
        cirq.KeyCondition(cirq.MeasurementKey('a')),
        cirq.KeyCondition(cirq.MeasurementKey('b')),
    )
    with pytest.raises(ValueError, match='Operation has multiple conditions'):
        _ = op.condition


def test_init_multiple_operations() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    op = cirq.If('m', cirq.X(q0), cirq.Y(q1))
    assert isinstance(op.sub_operation, cirq.CircuitOperation)
    assert op.sub_operation.circuit == cirq.Circuit(cirq.X(q0), cirq.Y(q1))
    assert op.qubits == (q0, q1)

    op_list = cirq.If('m', [cirq.X(q0), cirq.Y(q1)])
    assert isinstance(op_list.sub_operation, cirq.CircuitOperation)
    assert op_list.sub_operation.circuit == cirq.Circuit(cirq.X(q0), cirq.Y(q1))


def test_init_squash_nested_if() -> None:
    q = cirq.LineQubit(0)
    inner = cirq.If('a', cirq.X(q))
    outer = cirq.If('b', inner)
    assert outer.conditions == (
        cirq.KeyCondition(cirq.MeasurementKey('b')),
        cirq.KeyCondition(cirq.MeasurementKey('a')),
    )
    assert outer.sub_operation == cirq.X(q)

    cco = cirq.ClassicallyControlledOperation(cirq.X(q), ['a'])
    outer_cco = cirq.If('b', cco)
    assert outer_cco.conditions == (
        cirq.KeyCondition(cirq.MeasurementKey('b')),
        cirq.KeyCondition(cirq.MeasurementKey('a')),
    )
    assert outer_cco.sub_operation == cirq.X(q)


def test_init_errors() -> None:
    q = cirq.LineQubit(0)
    with pytest.raises(TypeError, match='Unrecognized condition type'):
        _ = cirq.If(123, cirq.X(q))

    with pytest.raises(TypeError, match='Unrecognized condition type'):
        _ = cirq.If(['a', 123], cirq.X(q))

    with pytest.raises(ValueError, match='At least one condition must be provided'):
        _ = cirq.If([], cirq.X(q))

    with pytest.raises(ValueError, match='Cannot conditionally run operations with measurements'):
        _ = cirq.If('m', cirq.measure(q, key='out'))


def test_with_qubits() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    op = cirq.If('m', cirq.X(q0))
    new_op = op.with_qubits(q1)
    assert new_op == cirq.If('m', cirq.X(q1))


def test_decomposition() -> None:
    q = cirq.LineQubit(0)
    op = cirq.If('m', cirq.X(q))
    decomposed = cirq.decompose_once(op)
    expected = [cirq.ClassicallyControlledOperation(cirq.X(q), ['m'])]
    assert decomposed == expected
    assert op._decompose_() == expected[0]

    circuit = cirq.Circuit(cirq.measure(q, key='m'), op)
    full_decomposed = cirq.Circuit(cirq.decompose(circuit))
    assert full_decomposed == cirq.Circuit(
        cirq.measure(q, key='m'), cirq.X(q).with_classical_controls('m')
    )


def test_value_equality() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.If('a', cirq.X(q0)), cirq.If(cirq.MeasurementKey('a'), cirq.X(q0)))
    eq.add_equality_group(cirq.If('b', cirq.X(q0)))
    eq.add_equality_group(cirq.If('a', cirq.X(q1)))
    eq.add_equality_group(cirq.If(['a', 'b'], cirq.X(q0)))


def test_str_and_repr() -> None:
    q = cirq.LineQubit(0)
    op1 = cirq.If('m', cirq.X(q))
    assert str(op1) == 'If(m, X(q(0)))'
    assert repr(op1) == "cirq.If(cirq.KeyCondition(cirq.MeasurementKey(name='m')), cirq.X(cirq.LineQubit(0)))"
    assert eval(repr(op1)) == op1

    op2 = cirq.If(['a', 'b'], cirq.X(q))
    assert str(op2) == 'If((a, b), X(q(0)))'
    assert (
        repr(op2)
        == "cirq.If([cirq.KeyCondition(cirq.MeasurementKey(name='a')), cirq.KeyCondition(cirq.MeasurementKey(name='b'))], cirq.X(cirq.LineQubit(0)))"
    )
    assert eval(repr(op2)) == op2


def test_parameterized_and_resolve() -> None:
    q = cirq.LineQubit(0)
    sym = sympy.Symbol('theta')
    cond_sym = sympy.Symbol('cond')

    op = cirq.If(cond_sym, cirq.Rx(rads=sym).on(q))
    assert cirq.is_parameterized(op)
    assert cirq.parameter_names(op) == {'theta'}

    resolved = cirq.resolve_parameters(op, cirq.ParamResolver({'theta': np.pi, 'cond': 1}))
    assert not cirq.is_parameterized(resolved)
    assert resolved == cirq.If(cond_sym, cirq.Rx(rads=np.pi).on(q))


def test_diagram() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.measure(q0, key='a'), cirq.If('a', cirq.X(q1)))
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
        cirq.measure(q0, key='a'), cirq.measure(q1, key='b'), cirq.If(['a', 'b'], cirq.X(q2))
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
    circuit = cirq.Circuit(cirq.If(sympy.Symbol('s'), cirq.X(q)))
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───X(If=s)───
      ║
s: ═══^═════════
""",
        use_unicode_characters=True,
    )


def test_simulation() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    # If q0 measured as 1, flip q1.
    circuit = cirq.Circuit(cirq.X(q0), cirq.measure(q0, key='a'), cirq.If('a', cirq.X(q1)))
    sim = cirq.Simulator()
    res = sim.simulate(circuit)
    assert res.measurements['a'] == [1]
    # Since 'a' is 1, X(q1) ran, so final state vector should have both q0 and q1 in |1>.
    expected_state = np.zeros(4, dtype=np.complex64)
    expected_state[3] = 1.0
    np.testing.assert_allclose(res.state_vector(), expected_state, atol=1e-6)

    # Now if q0 measured as 0, do not flip q1.
    circuit_zero = cirq.Circuit(cirq.measure(q0, key='a'), cirq.If('a', cirq.X(q1)))
    res_zero = sim.simulate(circuit_zero)
    assert res_zero.measurements['a'] == [0]
    expected_state_zero = np.zeros(4, dtype=np.complex64)
    expected_state_zero[0] = 1.0
    np.testing.assert_allclose(res_zero.state_vector(), expected_state_zero, atol=1e-6)


def test_key_mappings_and_scoping() -> None:
    q = cirq.LineQubit(0)
    op = cirq.If('a', cirq.X(q))

    mapped = cirq.with_measurement_key_mapping(op, {'a': 'b'})
    assert mapped == cirq.If('b', cirq.X(q))

    prefixed = cirq.with_key_path_prefix(op, ('path',))
    assert prefixed == cirq.If('path:a', cirq.X(q))

    rescoped = cirq.with_rescoped_keys(
        op, ('scope',), frozenset([cirq.MeasurementKey.parse_serialized('scope:a')])
    )
    assert rescoped == cirq.If('scope:a', cirq.X(q))

    assert cirq.control_keys(op) == frozenset([cirq.MeasurementKey('a')])


def test_qasm() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    op = cirq.If('a', cirq.X(q1))
    circuit = cirq.Circuit(cirq.measure(q0, key='a'), op)
    qasm_str = cirq.qasm(circuit)
    assert 'if (m_a==1) cx q[1];' in qasm_str or 'if (m_a==1) x q[1];' in qasm_str

    op_multi = cirq.If(['a', 'b'], cirq.X(q1))
    with pytest.raises(ValueError, match='QASM 2.0 does not support multiple conditions'):
        _ = cirq.qasm(op_multi)


def test_json_serialization() -> None:
    q = cirq.LineQubit(0)
    op = cirq.If('m', cirq.X(q))
    cirq.testing.assert_json_roundtrip_works(op)
    op_multi = cirq.If(['a', 'b'], cirq.X(q))
    cirq.testing.assert_json_roundtrip_works(op_multi)


def test_diagram_multiple_sympy_conditions() -> None:
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.If([sympy.Symbol('s'), sympy.Symbol('t')], cirq.X(q)))
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───X(If=s, t)───
      ║
s: ═══^════════════
      ║
t: ═══^════════════
""",
        use_unicode_characters=True,
    )


def test_qasm_sub_op_no_qasm() -> None:
    class NoQasmOp(cirq.Operation):
        @property
        def qubits(self):
            return (cirq.LineQubit(0),)

        def with_qubits(self, *new_qubits):
            return self

    op = cirq.If('a', NoQasmOp())
    assert op.qubits == (cirq.LineQubit(0),)
    assert op.with_qubits(cirq.LineQubit(1)).sub_operation == op.sub_operation
    assert cirq.qasm(op, default=None) is None
