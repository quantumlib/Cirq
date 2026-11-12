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
    i = sympy.Symbol('i')

    # With string loop_var
    op1 = cirq.For('i', [0, 1], cirq.X(q))
    assert op1.loop_var == i
    assert op1.values == (0, 1)
    assert op1.sub_operation == cirq.X(q)
    assert op1.qubits == (q,)
    assert op1.classical_controls == frozenset()
    assert op1.without_classical_controls() == op1

    # With SymPy loop_var
    op2 = cirq.For(i, [0, 1], cirq.X(q))
    assert op2.loop_var == i
    assert op2.values == (0, 1)

    # With int count (range(3))
    op3 = cirq.For('i', 3, cirq.X(q))
    assert op3.values == (0, 1, 2)

    # With range
    op4 = cirq.For('i', range(3), cirq.X(q))
    assert op4.values == (0, 1, 2)

    # Without loop_var
    op5 = cirq.For([0, 1], cirq.X(q))
    assert op5.loop_var is None
    assert op5.values == (0, 1)

    op6 = cirq.For(3, cirq.X(q))
    assert op6.loop_var is None
    assert op6.values == (0, 1, 2)

    op7 = cirq.For(range(3), cirq.X(q))
    assert op7.loop_var is None
    assert op7.values == (0, 1, 2)

    # Tuple syntax
    op8 = cirq.For(('i', 3), cirq.X(q))
    assert op8.loop_var == i
    assert op8.values == (0, 1, 2)

    op9 = cirq.For(('i', [0, 1]), cirq.X(q))
    assert op9.loop_var == i
    assert op9.values == (0, 1)

    # Keyword arguments
    op10 = cirq.For(loop_var='i', values=3, sub_operation=cirq.X(q))
    assert op10.loop_var == i
    assert op10.values == (0, 1, 2)

    op11 = cirq.For(values=3, sub_operation=cirq.X(q))
    assert op11.loop_var is None
    assert op11.values == (0, 1, 2)


def test_init_multiple_operations() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    op = cirq.For('i', 3, cirq.X(q0), cirq.Y(q1))
    assert isinstance(op.sub_operation, cirq.CircuitOperation)
    assert op.sub_operation.circuit == cirq.Circuit(cirq.X(q0), cirq.Y(q1))
    assert op.qubits == (q0, q1)

    op_no_var = cirq.For(3, cirq.X(q0), cirq.Y(q1))
    assert isinstance(op_no_var.sub_operation, cirq.CircuitOperation)
    assert op_no_var.sub_operation.circuit == cirq.Circuit(cirq.X(q0), cirq.Y(q1))

    op_list = cirq.For('i', 3, [cirq.X(q0), cirq.Y(q1)])
    assert isinstance(op_list.sub_operation, cirq.CircuitOperation)
    assert op_list.sub_operation.circuit == cirq.Circuit(cirq.X(q0), cirq.Y(q1))


def test_init_errors() -> None:
    q = cirq.LineQubit(0)
    with pytest.raises(TypeError, match='Unrecognized loop_var type'):
        _ = cirq.For(123.45, [0, 1], cirq.X(q))

    with pytest.raises(TypeError, match='Unrecognized values type'):
        _ = cirq.For('i', object(), cirq.X(q))

    with pytest.raises(ValueError, match='Iteration count must be non-negative'):
        _ = cirq.For('i', -1, cirq.X(q))

    with pytest.raises(ValueError, match='At least one sub-operation must be provided'):
        _ = cirq.For('i', 3, None)

    with pytest.raises(ValueError, match='Iteration values must be provided'):
        _ = cirq.For(loop_var='i', values=None, sub_operation=cirq.X(q))


def test_with_qubits() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    op = cirq.For('i', 3, cirq.X(q0))
    new_op = op.with_qubits(q1)
    assert new_op == cirq.For('i', 3, cirq.X(q1))

    op_no_var = cirq.For(3, cirq.X(q0))
    assert op_no_var.with_qubits(q1) == cirq.For(3, cirq.X(q1))


def test_decomposition() -> None:
    q = cirq.LineQubit(0)
    i = sympy.Symbol('i')

    op = cirq.For('i', [0, 1, 2], cirq.X(q) ** i)
    expected = [cirq.X(q) ** 0, cirq.X(q) ** 1, cirq.X(q) ** 2]
    assert cirq.decompose_once(op) == expected
    assert op._decompose_() == expected
    assert op._decompose_with_context_() == expected

    op_no_var = cirq.For(3, cirq.X(q))
    assert cirq.decompose_once(op_no_var) == [cirq.X(q), cirq.X(q), cirq.X(q)]

    # Empty loop
    op_empty = cirq.For(0, cirq.X(q))
    assert cirq.decompose_once(op_empty) == []

    # Inside circuit
    circuit = cirq.Circuit(op)
    assert cirq.Circuit(cirq.decompose(circuit)) == cirq.Circuit(expected)

    # Symbolic values cannot decompose
    n = sympy.Symbol('n')
    op_sym = cirq.For('i', n, cirq.X(q) ** i)
    assert op_sym._decompose_() is NotImplemented


def test_value_equality() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(
        cirq.For('i', [0, 1], cirq.X(q0)),
        cirq.For(sympy.Symbol('i'), [0, 1], cirq.X(q0)),
        cirq.For(('i', [0, 1]), cirq.X(q0)),
    )
    eq.add_equality_group(cirq.For('j', [0, 1], cirq.X(q0)))
    eq.add_equality_group(cirq.For('i', [0, 2], cirq.X(q0)))
    eq.add_equality_group(cirq.For('i', [0, 1], cirq.X(q1)))
    eq.add_equality_group(cirq.For([0, 1], cirq.X(q0)))


def test_str_and_repr() -> None:
    q = cirq.LineQubit(0)
    op1 = cirq.For('i', [0, 1], cirq.X(q))
    assert str(op1) == 'For(i, [0, 1], X(q(0)))'
    assert repr(op1) == "cirq.For(sympy.Symbol('i'), [0, 1], cirq.X(cirq.LineQubit(0)))"
    assert eval(repr(op1)) == op1

    op2 = cirq.For([0, 1], cirq.X(q))
    assert str(op2) == 'For([0, 1], X(q(0)))'
    assert repr(op2) == "cirq.For([0, 1], cirq.X(cirq.LineQubit(0)))"
    assert eval(repr(op2)) == op2


def test_parameterized_and_resolve() -> None:
    q = cirq.LineQubit(0)
    i = sympy.Symbol('i')
    theta = sympy.Symbol('theta')
    n = sympy.Symbol('n')

    # Loop var is bound, so not a free parameter
    op_bound = cirq.For(i, [0, 1], cirq.X(q) ** i)
    assert not cirq.is_parameterized(op_bound)
    assert cirq.parameter_names(op_bound) == frozenset()

    # Free parameter in sub_operation
    op_free = cirq.For(i, [0, 1], cirq.Rx(rads=theta).on(q) ** i)
    assert cirq.is_parameterized(op_free)
    assert cirq.parameter_names(op_free) == {'theta'}

    resolved = cirq.resolve_parameters(op_free, cirq.ParamResolver({'theta': np.pi, 'i': 99}))
    assert not cirq.is_parameterized(resolved)
    assert resolved == cirq.For(i, [0, 1], cirq.Rx(rads=np.pi).on(q) ** i)

    # Parameterized values
    op_sym_vals = cirq.For(i, n, cirq.X(q) ** i)
    assert cirq.is_parameterized(op_sym_vals)
    assert cirq.parameter_names(op_sym_vals) == {'n'}

    resolved_vals = cirq.resolve_parameters(op_sym_vals, cirq.ParamResolver({'n': 2}))
    assert not cirq.is_parameterized(resolved_vals)
    assert resolved_vals == cirq.For(i, [0, 1], cirq.X(q) ** i)

    # Negative resolved count
    with pytest.raises(ValueError, match='Iteration count must be non-negative'):
        _ = cirq.resolve_parameters(op_sym_vals, cirq.ParamResolver({'n': -5}))

    # Sequence with sympy symbols
    a, b = sympy.Symbol('a'), sympy.Symbol('b')
    op_seq = cirq.For(i, [a, b], cirq.X(q) ** i)
    assert cirq.parameter_names(op_seq) == {'a', 'b'}
    resolved_seq = cirq.resolve_parameters(op_seq, cirq.ParamResolver({'a': 0, 'b': 1}))
    assert resolved_seq == cirq.For(i, [0, 1], cirq.X(q) ** i)


def test_diagram() -> None:
    q = cirq.LineQubit(0)
    i = sympy.Symbol('i')
    circuit = cirq.Circuit(cirq.For(i, [0, 1], cirq.X(q) ** i))
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───X(For=i in [0, 1])^i───
""",
        use_unicode_characters=True,
    )

    circuit_no_var = cirq.Circuit(cirq.For([0, 1], cirq.X(q)))
    cirq.testing.assert_has_diagram(
        circuit_no_var,
        """
0: ───X(For=[0, 1])───
""",
        use_unicode_characters=True,
    )


def test_simulation() -> None:
    q = cirq.LineQubit(0)
    i = sympy.Symbol('i')

    # Flip qubit if i is 1, keep if i is 0
    circuit = cirq.Circuit(cirq.For(i, [1, 0], cirq.X(q) ** i))
    sim = cirq.Simulator()
    res = sim.simulate(circuit)
    np.testing.assert_allclose(res.state_vector(), [0, 1])

    # 4 quarter rotations: Rx(pi/2) * 4 = Rx(2pi) = -I, state stays |0> up to global phase
    circuit_rot = cirq.Circuit(cirq.For(4, cirq.Rx(rads=np.pi / 2).on(q)))
    res_rot = sim.simulate(circuit_rot)
    np.testing.assert_allclose(np.abs(res_rot.state_vector()), [1, 0], atol=1e-7)

    # Simulation with measurements inside loop
    circuit_meas = cirq.Circuit(cirq.X(q), cirq.For(3, cirq.measure(q, key='m')))
    res_meas = sim.run(circuit_meas, repetitions=5)
    np.testing.assert_equal(res_meas.records['m'], np.ones((5, 3, 1), dtype=int))

    # Cannot simulate with symbolic ungrounded bounds
    n = sympy.Symbol('n')
    op_err = cirq.For('i', n, cirq.X(q))
    state = cirq.StateVectorSimulationState(qubits=[q])
    with pytest.raises(ValueError, match='Cannot simulate parameterized For loop'):
        op_err._act_on_(state)


def test_key_mappings_and_scoping() -> None:
    q = cirq.LineQubit(0)
    op = cirq.For(3, cirq.measure(q, key='a'))

    mapped = cirq.with_measurement_key_mapping(op, {'a': 'b'})
    assert mapped == cirq.For(3, cirq.measure(q, key='b'))

    prefixed = cirq.with_key_path_prefix(op, ('path',))
    assert prefixed == cirq.For(
        3, cirq.measure(q, key=cirq.MeasurementKey(path=('path',), name='a'))
    )

    rescoped = cirq.with_rescoped_keys(
        op, ('scope',), frozenset([cirq.MeasurementKey.parse_serialized('scope:a')])
    )
    assert rescoped == cirq.For(
        3, cirq.measure(q, key=cirq.MeasurementKey(path=('scope',), name='a'))
    )

    assert cirq.is_measurement(op)
    assert cirq.measurement_key_names(op) == {'a'}
    assert cirq.measurement_key_objs(op) == {cirq.MeasurementKey('a')}
    assert cirq.control_keys(op) == frozenset()


def test_has_unitary() -> None:
    q = cirq.LineQubit(0)
    i = sympy.Symbol('i')
    op_unitary = cirq.For(i, [0, 1], cirq.X(q) ** i)
    assert cirq.has_unitary(op_unitary)
    expected_u = cirq.unitary(cirq.X(q)) @ cirq.unitary(cirq.X(q) ** 0)
    np.testing.assert_allclose(cirq.unitary(op_unitary), expected_u)

    op_non_unitary = cirq.For(2, cirq.measure(q, key='m'))
    assert not cirq.has_unitary(op_non_unitary)


def test_commutes() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    op = cirq.For(1, cirq.X(q0))
    assert cirq.commutes(op, cirq.X(q0))
    assert not cirq.commutes(op, cirq.Z(q0))
    assert cirq.commutes(op, cirq.Z(q1))


def test_qasm() -> None:
    q0 = cirq.LineQubit(0)
    i = sympy.Symbol('i')
    op = cirq.For(i, [0, 1], cirq.X(q0))
    circuit = cirq.Circuit(op)

    with pytest.raises(ValueError, match='QASM 2.0 does not support for loops'):
        _ = cirq.qasm(circuit)

    qasm_str = cirq.qasm(circuit, args=cirq.QasmArgs(version='3.0'))
    assert 'for int i in {0, 1} x q[0];' in qasm_str

    op_no_var = cirq.For([0, 1], cirq.X(q0))
    circuit_no_var = cirq.Circuit(op_no_var)
    qasm_str_no_var = cirq.qasm(circuit_no_var, args=cirq.QasmArgs(version='3.0'))
    assert 'for int _ in {0, 1} x q[0];' in qasm_str_no_var


def test_qasm_sub_op_no_qasm() -> None:
    class NoQasmOp(cirq.Operation):
        @property
        def qubits(self):
            return (cirq.LineQubit(0),)  # pragma: no cover

        def with_qubits(self, *new_qubits):
            return self  # pragma: no cover

    op = cirq.For('i', 2, NoQasmOp())
    assert cirq.qasm(op, args=cirq.QasmArgs(version='3.0'), default=None) is None


def test_qasm_direct_and_symbolic() -> None:
    q0 = cirq.LineQubit(0)
    op = cirq.For('i', 2, cirq.X(q0))
    with pytest.raises(ValueError, match='QASM 2.0 does not support for loops'):
        _ = op._qasm_()

    n = sympy.Symbol('n')
    op_sym = cirq.For('i', n, cirq.X(q0))
    args = cirq.QasmArgs(version='3.0', qubit_id_map={q0: 'q[0]'})
    assert op_sym._qasm_(args=args) == 'for int i in [n] x q[0];\n'


def test_json_dict() -> None:
    q0 = cirq.LineQubit(0)
    op = cirq.For('i', [0, 1], cirq.X(q0))
    assert op._json_dict_() == {
        'loop_var': sympy.Symbol('i'),
        'values': [0, 1],
        'sub_operation': cirq.X(q0),
    }

    n = sympy.Symbol('n')
    op_sym = cirq.For(None, n, cirq.X(q0))
    assert op_sym._json_dict_() == {'loop_var': None, 'values': n, 'sub_operation': cirq.X(q0)}
