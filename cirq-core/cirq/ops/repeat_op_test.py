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
    op = cirq.Repeat(3, cirq.X(q))
    assert op.count == 3
    assert op.sub_operation == cirq.X(q)
    assert op.qubits == (q,)
    assert op.classical_controls == frozenset()


def test_init_count_types() -> None:
    q = cirq.LineQubit(0)
    assert cirq.Repeat(np.int64(2), cirq.X(q)).count == 2  # type: ignore[arg-type]
    assert type(cirq.Repeat(np.int64(2), cirq.X(q)).count) is int  # type: ignore[arg-type]
    assert cirq.Repeat(0, cirq.X(q)).count == 0

    for bad in (True, 2.0, '2', None):
        with pytest.raises(TypeError, match='count must be an integer'):
            _ = cirq.Repeat(bad, cirq.X(q))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match='count must be non-negative'):
        _ = cirq.Repeat(-1, cirq.X(q))


def test_init_multiple_operations() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    op = cirq.Repeat(2, cirq.X(q0), cirq.Y(q1))
    assert isinstance(op.sub_operation, cirq.CircuitOperation)
    assert op.sub_operation.circuit == cirq.Circuit(cirq.X(q0), cirq.Y(q1))
    assert op.qubits == (q0, q1)

    op_list = cirq.Repeat(2, [cirq.X(q0), cirq.Y(q1)])
    assert isinstance(op_list.sub_operation, cirq.CircuitOperation)
    assert op_list.sub_operation.circuit == cirq.Circuit(cirq.X(q0), cirq.Y(q1))


def test_nested_preserved() -> None:
    target = cirq.LineQubit(0)
    inner = cirq.Repeat(2, cirq.X(target))
    assert cirq.Repeat(3, inner).sub_operation == inner

    inner_while = cirq.While('a', cirq.X(target))
    assert cirq.Repeat(3, inner_while).sub_operation == inner_while

    inner_if = cirq.If('a', cirq.X(target))
    assert cirq.Repeat(3, inner_if).sub_operation == inner_if


def test_classical_controls() -> None:
    q = cirq.LineQubit(0)
    cond = cirq.KeyCondition(cirq.MeasurementKey('a'))
    op = cirq.Repeat(3, cirq.X(q).with_classical_controls('a'))
    assert op.classical_controls == frozenset([cond])
    assert op.without_classical_controls() == cirq.Repeat(3, cirq.X(q))
    assert cirq.control_keys(op) == frozenset([cirq.MeasurementKey('a')])


def test_with_qubits() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    assert cirq.Repeat(3, cirq.X(q0)).with_qubits(q1) == cirq.Repeat(3, cirq.X(q1))


def test_value_equality() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(
        cirq.Repeat(3, cirq.X(q0)), cirq.Repeat(np.int64(3), cirq.X(q0))  # type: ignore[arg-type]
    )
    eq.add_equality_group(cirq.Repeat(2, cirq.X(q0)))
    eq.add_equality_group(cirq.Repeat(3, cirq.X(q1)))
    eq.add_equality_group(
        cirq.Repeat(3, cirq.X(q0), cirq.Y(q1)),
        cirq.Repeat(3, [cirq.X(q0), cirq.Y(q1)]),
        cirq.Repeat(3, cirq.Circuit(cirq.X(q0), cirq.Y(q1))),
        cirq.Repeat(3, cirq.CircuitOperation(cirq.FrozenCircuit(cirq.X(q0), cirq.Y(q1)))),
    )
    eq.add_equality_group(cirq.Repeat(3, cirq.Y(q0), cirq.X(q1)))


def test_str_and_repr() -> None:
    q = cirq.LineQubit(0)
    op = cirq.Repeat(3, cirq.X(q))
    assert str(op) == 'Repeat(3, X(q(0)))'
    assert repr(op) == 'cirq.Repeat(3, cirq.X(cirq.LineQubit(0)))'
    assert eval(repr(op)) == op

    q0, q1 = cirq.LineQubit.range(2)
    op_multiline = cirq.Repeat(2, cirq.X(q0), cirq.X(q1))
    assert str(op_multiline) == """\
Repeat(2, [ 0: ───X─── ]
          [            ]
          [ 1: ───X─── ])"""


def test_parameterized_and_resolve() -> None:
    q = cirq.LineQubit(0)
    sym = sympy.Symbol('theta')
    op = cirq.Repeat(3, cirq.Rx(rads=sym).on(q))
    assert cirq.is_parameterized(op)
    assert cirq.parameter_names(op) == {'theta'}

    resolved = cirq.resolve_parameters(op, cirq.ParamResolver({'theta': np.pi}))
    assert not cirq.is_parameterized(resolved)
    assert resolved == cirq.Repeat(3, cirq.Rx(rads=np.pi).on(q))


def test_diagram() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.Repeat(3, cirq.X(q0)), cirq.Repeat(2, cirq.CNOT(q0, q1)))
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───X(Repeat=3)───@(Repeat=2)───
                    │
1: ─────────────────X─────────────
""",
        use_unicode_characters=True,
    )


@pytest.mark.parametrize('sub_operation', [[], cirq.Circuit()])
def test_diagram_empty_op_tree(sub_operation: cirq.OP_TREE) -> None:
    op = cirq.Repeat(2, sub_operation)
    assert isinstance(op.sub_operation, cirq.CircuitOperation)
    assert op.sub_operation.circuit == cirq.FrozenCircuit()

    args = cirq.CircuitDiagramInfoArgs.UNINFORMED_DEFAULT
    assert op._circuit_diagram_info_(args) is NotImplemented
    assert cirq.circuit_diagram_info(op, default=None) is None

    cirq.testing.assert_has_diagram(
        cirq.Circuit(op),
        """
    Repeat(2, [  ])
""",
        use_unicode_characters=True,
    )


def test_diagram_circuit_operation_body() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.Repeat(2, cirq.X(q0), cirq.CNOT(q0, q1)))
    cirq.testing.assert_has_diagram(
        circuit,
        """
      Repeat(2, [ 0: ───X───@─── ]
0: ───          [           │    ]────
                [ 1: ───────X─── ])
      │
1: ───#2──────────────────────────────
""",
        use_unicode_characters=True,
    )


def test_diagram_nested() -> None:
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.Repeat(2, cirq.Repeat(3, cirq.X(q))))
    cirq.testing.assert_has_diagram(
        circuit, "0: ───X(Repeat=3)(Repeat=2)───", use_unicode_characters=True
    )


def test_diagram_measurement_body() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    cirq.testing.assert_has_diagram(
        cirq.Circuit(cirq.Repeat(2, cirq.measure(q0, key='a'))),
        "0: ───M('a')(Repeat=2)───",
        use_unicode_characters=True,
    )

    circuit = cirq.Circuit(
        cirq.Repeat(2, cirq.measure(q0, key='a')), cirq.X(q1).with_classical_controls('a')
    )
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───M(Repeat=2)───────
      ║
1: ───╫─────────────X───
      ║             ║
a: ═══@═════════════^═══
""",
        use_unicode_characters=True,
    )


def test_decompose() -> None:
    q = cirq.LineQubit(0)
    op = cirq.Repeat(3, cirq.X(q))
    expected = cirq.CircuitOperation(
        cirq.FrozenCircuit(cirq.X(q)), repetitions=3, use_repetition_ids=False
    )
    assert cirq.decompose_once(op) == [expected]
    assert op._decompose_with_context_() == expected
    assert cirq.decompose(op) == [cirq.X(q)] * 3

    op_zero = cirq.Repeat(0, cirq.X(q))
    assert cirq.decompose_once(op_zero) == [
        cirq.CircuitOperation(
            cirq.FrozenCircuit(cirq.X(q)), repetitions=0, use_repetition_ids=False
        )
    ]
    assert cirq.decompose(op_zero) == []

    # Measurement keys are shared across iterations, not suffixed with repetition ids.
    op_m = cirq.Repeat(3, cirq.measure(q, key='a'))
    assert cirq.measurement_key_names(cirq.decompose_once(op_m)[0]) == {'a'}


def test_unitary() -> None:
    q = cirq.LineQubit(0)
    op = cirq.Repeat(3, cirq.Rz(rads=0.1).on(q))
    assert cirq.has_unitary(op)
    np.testing.assert_allclose(cirq.unitary(op), cirq.unitary(cirq.Rz(rads=0.3)))
    np.testing.assert_allclose(cirq.unitary(cirq.Repeat(0, cirq.X(q))), np.eye(2))
    assert not cirq.has_unitary(cirq.Repeat(2, cirq.measure(q, key='m')))


def test_simulation() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    sim = cirq.Simulator()

    res = sim.simulate(cirq.Circuit(cirq.Repeat(3, cirq.X(q0))))
    np.testing.assert_allclose(np.abs(res.final_state_vector), [0, 1])

    res = sim.simulate(cirq.Circuit(cirq.Repeat(4, cirq.X(q0)), cirq.Repeat(0, cirq.X(q1))))
    np.testing.assert_allclose(np.abs(res.final_state_vector), [1, 0, 0, 0])

    dm_res = cirq.DensityMatrixSimulator().run(
        cirq.Circuit(cirq.Repeat(3, cirq.X(q0)), cirq.measure(q0, key='m')), repetitions=3
    )
    np.testing.assert_array_equal(dm_res.records['m'], np.ones((3, 1, 1), dtype=int))


def test_simulation_with_measurement() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.Repeat(3, cirq.X(q0), cirq.measure(q0, q1, key='a')))
    res = cirq.Simulator().run(circuit, repetitions=4)
    expected = np.tile([[[1, 0], [0, 0], [1, 0]]], (4, 1, 1))
    np.testing.assert_array_equal(res.records['a'], expected)

    res0 = cirq.Simulator().run(
        cirq.Circuit(cirq.Repeat(0, cirq.measure(q0, key='a')), cirq.measure(q1, key='b')),
        repetitions=2,
    )
    assert 'a' not in res0.records
    np.testing.assert_array_equal(res0.records['b'], np.zeros((2, 1, 1), dtype=int))


def test_simulation_nested() -> None:
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.Repeat(2, cirq.Repeat(3, cirq.X(q), cirq.measure(q, key='a'))))
    res = cirq.Simulator().run(circuit, repetitions=2)
    expected = np.tile([[[1], [0], [1], [0], [1], [0]]], (2, 1, 1))
    np.testing.assert_array_equal(res.records['a'], expected)


def test_simulation_if_in_body() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    # The If sees the measurement from the current iteration: q0 is reset to 0 after every
    # measurement, so 'a' is 1 each time and q1 is flipped three times.
    circuit = cirq.Circuit(
        cirq.Repeat(3, cirq.X(q0), cirq.measure(q0, key='a'), cirq.If('a', cirq.X(q0), cirq.X(q1))),
        cirq.measure(q1, key='b'),
    )
    res = cirq.Simulator().run(circuit, repetitions=2)
    np.testing.assert_array_equal(res.records['a'], np.ones((2, 3, 1), dtype=int))
    np.testing.assert_array_equal(res.records['b'], np.ones((2, 1, 1), dtype=int))


def test_simulation_inside_while() -> None:
    q = cirq.LineQubit(0)
    # The first Repeat pass measures 0, 1, 0, so the While exits after one iteration.
    circuit = cirq.Circuit(
        cirq.X(q),
        cirq.measure(q, key='a'),
        cirq.While('a', cirq.Repeat(3, cirq.X(q), cirq.measure(q, key='a'))),
    )
    res = cirq.Simulator().run(circuit, repetitions=2)
    expected = np.tile([[[1], [0], [1], [0]]], (2, 1, 1))
    np.testing.assert_array_equal(res.records['a'], expected)


def test_key_mappings_and_scoping() -> None:
    q = cirq.LineQubit(0)
    op = cirq.Repeat(2, cirq.X(q).with_classical_controls('a'))

    assert cirq.with_measurement_key_mapping(op, {'a': 'b'}) == cirq.Repeat(
        2, cirq.X(q).with_classical_controls('b')
    )
    assert cirq.with_key_path_prefix(op, ('path',)) == cirq.Repeat(
        2, cirq.X(q).with_classical_controls('path:a')
    )
    rescoped = cirq.with_rescoped_keys(
        op, ('scope',), frozenset([cirq.MeasurementKey.parse_serialized('scope:a')])
    )
    assert rescoped == cirq.Repeat(2, cirq.X(q).with_classical_controls('scope:a'))
    assert cirq.control_keys(op) == frozenset([cirq.MeasurementKey('a')])
    assert not cirq.is_measurement(op)
    assert cirq.measurement_key_objs(op) == frozenset()

    op_meas = cirq.Repeat(2, cirq.H(q), cirq.measure(q, key='b'))
    assert cirq.is_measurement(op_meas)
    assert cirq.measurement_key_objs(op_meas) == frozenset([cirq.MeasurementKey('b')])
    assert cirq.measurement_key_names(op_meas) == frozenset(['b'])

    op_m = cirq.Repeat(2, cirq.measure(q, key='a'))
    assert cirq.with_measurement_key_mapping(op_m, {'a': 'b'}) == cirq.Repeat(
        2, cirq.measure(q, key='b')
    )
    assert cirq.with_key_path_prefix(op_m, ('path',)) == cirq.Repeat(
        2, cirq.measure(q, key=cirq.MeasurementKey('a', ('path',)))
    )
    assert cirq.Circuit(op_m).all_measurement_key_names() == {'a'}

    # Sub-operations lacking key-mapping support are preserved.
    plain = cirq.Repeat(2, cirq.X(q))
    assert cirq.with_measurement_key_mapping(plain, {'a': 'b'}) == plain
    assert cirq.with_key_path_prefix(plain, ('p',)) == plain


def test_commutes() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    op = cirq.Repeat(3, cirq.X(q0))
    assert not cirq.commutes(op, cirq.Z(q0))
    assert cirq.commutes(op, cirq.Z(q1))
    assert cirq.commutes(cirq.Repeat(2, cirq.X(q0)), cirq.X(q0))


def test_qasm() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    op = cirq.Repeat(3, cirq.X(q1))
    circuit = cirq.Circuit(op)
    with pytest.raises(ValueError, match='QASM 2.0 does not support loops'):
        _ = cirq.qasm(op)
    with pytest.raises(ValueError, match='QASM 2.0 does not support loops'):
        _ = cirq.qasm(circuit)

    qasm_str = cirq.qasm(circuit, args=cirq.QasmArgs(version='3.0'))
    assert 'for int _i in [0:2] {\n  x q[0];\n}' in qasm_str

    op_multi = cirq.Repeat(2, cirq.X(q0), cirq.measure(q1, key='a'))
    # A CircuitOperation body has no QASM of its own, so the exporter unrolls it.
    qasm_multi = cirq.qasm(
        cirq.Circuit(cirq.measure(q1, key='a'), op_multi), args=cirq.QasmArgs(version='3.0')
    )
    assert 'for' not in qasm_multi
    assert qasm_multi.count('x q[0];') == 2

    op_nested = cirq.Repeat(2, cirq.Repeat(3, cirq.X(q0)))
    qasm_nested = cirq.qasm(cirq.Circuit(op_nested), args=cirq.QasmArgs(version='3.0'))
    assert 'for int _i in [0:1] {\n  for int _i in [0:2] {\n    x q[0];\n  }\n}' in qasm_nested


def test_qasm_count_zero() -> None:
    q = cirq.LineQubit(0)
    op = cirq.Repeat(0, cirq.X(q))
    args = cirq.QasmArgs(version='3.0', qubit_id_map={q: 'q[0]'})
    assert cirq.qasm(op, args=args) == ''
    qasm_str = cirq.qasm(cirq.Circuit(op), args=cirq.QasmArgs(version='3.0'))
    assert 'for' not in qasm_str
    assert 'x q' not in qasm_str


def test_qasm_sub_op_no_qasm() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    op_circuit_op = cirq.Repeat(2, cirq.X(q0), cirq.Y(q1))
    assert cirq.qasm(op_circuit_op, args=cirq.QasmArgs(version='3.0'), default=None) is None

    class NoQasmOp(cirq.Operation):
        @property
        def qubits(self) -> tuple[cirq.Qid, ...]:
            return (cirq.LineQubit(0),)  # pragma: no cover

        def with_qubits(self, *new_qubits: cirq.Qid) -> cirq.Operation:
            return self  # pragma: no cover

    op = cirq.Repeat(2, NoQasmOp())
    assert cirq.qasm(op, args=cirq.QasmArgs(version='3.0'), default=None) is None
