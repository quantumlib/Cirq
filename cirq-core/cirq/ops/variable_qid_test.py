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

from __future__ import annotations

import pytest
import sympy

import cirq


def test_variable_qid_init():
    x = sympy.Symbol('x')
    qx = cirq.VariableQid(x)
    assert qx.symbol == x
    assert qx.dimension == 2

    qy = cirq.VariableQid("y")
    assert qy.symbol == sympy.Symbol("y")
    assert qy.dimension == 2

    qx_d3 = cirq.VariableQid(x, dimension=3)
    assert qx_d3.dimension == 3

    with pytest.raises(ValueError, match="Expected a positive integer"):
        _ = cirq.VariableQid(x, dimension=-1)

    with pytest.raises(
        TypeError, match="Only sympy expressions or strings are supported for VariableQid"
    ):
        _ = cirq.VariableQid(4)


def test_variable_qid_comparison():
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    qx = cirq.VariableQid(x)
    qy = cirq.VariableQid(y)

    assert qx < qy
    assert qx == cirq.VariableQid(x)

    qx_d3 = cirq.VariableQid(x, dimension=3)
    assert qx < qx_d3


def test_variable_qid_parameterization():
    x = sympy.Symbol('x')
    qx = cirq.VariableQid(x)
    assert cirq.is_parameterized(qx)
    assert cirq.parameter_names(qx) == {'x'}


def test_variable_qid_with_dimension():
    x = sympy.Symbol('x')
    qx = cirq.VariableQid(x)
    assert qx.with_dimension(2) == qx

    qx_d3 = qx.with_dimension(3)
    assert qx_d3.dimension == 3


def test_variable_qid_basic_resolution():
    x = sympy.Symbol('x')
    qx = cirq.VariableQid(x)

    resolver = cirq.ParamResolver({x: cirq.LineQubit(3)})
    assert cirq.resolve_parameters(qx, resolver) == cirq.LineQubit(3)

    resolver = cirq.ParamResolver({x: cirq.GridQubit(1, 2)})
    assert cirq.resolve_parameters(qx, resolver) == cirq.GridQubit(1, 2)

    resolver = cirq.ParamResolver({x: cirq.NamedQubit('bob')})
    assert cirq.resolve_parameters(qx, resolver) == cirq.NamedQubit('bob')

    # Unresolved (not in resolver)
    resolver = cirq.ParamResolver({})
    assert cirq.resolve_parameters(qx, resolver) == qx

    # Unresolved (resolves to non-Qid)
    resolver = cirq.ParamResolver({x: 3})
    assert cirq.resolve_parameters(qx, resolver) == qx

    # unresolved - cannot resolve constant
    q1 = cirq.VariableQid(x - x + 1)
    assert cirq.resolve_parameters(q1, resolver) == q1

    # unresolved = cannot resolve all variables
    resolver = cirq.ParamResolver({x: cirq.LineQubit(3)})
    qxy = cirq.VariableQid(sympy.Symbol('x') + sympy.Symbol('y'))
    assert cirq.resolve_parameters(qxy, resolver) == qxy


def test_variable_qid_expression_resolution():
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    x1 = x + 1
    qx1 = cirq.VariableQid(x1)
    qxy = cirq.VariableQid(x + y)
    q2x = cirq.VariableQid(2 * x)

    resolver = cirq.ParamResolver({x: cirq.LineQubit(3)})
    assert cirq.resolve_parameters(qx1, resolver) == cirq.LineQubit(4)

    resolver = cirq.ParamResolver({"x": cirq.LineQubit(3)})
    assert cirq.resolve_parameters(qx1, resolver) == cirq.LineQubit(4)

    resolver = cirq.ParamResolver({x: cirq.LineQubit(3)})
    assert cirq.resolve_parameters(q2x, resolver) == cirq.LineQubit(6)

    resolver = cirq.ParamResolver({x: cirq.GridQubit(1, 2), y: cirq.GridQubit(10, 20)})
    assert cirq.resolve_parameters(qxy, resolver) == cirq.GridQubit(11, 22)

    resolver = cirq.ParamResolver({x: cirq.GridQubit(1, 2)})
    assert cirq.resolve_parameters(q2x, resolver) == cirq.GridQubit(2, 4)


def test_variable_qid_resolution_dimension_mismatch():
    x = sympy.Symbol('x')
    qx = cirq.VariableQid(x, dimension=3)

    resolver = cirq.ParamResolver({x: cirq.LineQubit(3)})
    with pytest.raises(
        ValueError,
        match=r"Resolved Qid dimension \(2\) does not match the VariableQid dimension \(3\)",
    ):
        cirq.resolve_parameters(qx, resolver)


def test_variable_qid_resolved_value():
    qx = cirq.VariableQid(sympy.Symbol('x'))
    assert qx._resolved_value_() == NotImplemented


def test_variable_qid_circuit_diagram_info():
    x = sympy.Symbol('x')
    qx = cirq.VariableQid(x, dimension=3)

    info = cirq.circuit_diagram_info(qx)
    assert info == cirq.CircuitDiagramInfo(wire_symbols=('x (d=3)',))


def test_variable_qid_resolution_operation():
    x = sympy.Symbol('x')
    qx = cirq.VariableQid(x)

    class NongateOperation(cirq.Operation):
        def __init__(self, qubits):
            self._qubits = tuple(qubits)

        @property
        def qubits(self) -> tuple[cirq.Qid, ...]:
            """The qubits targeted by the operation."""
            return self._qubits

        def with_qubits(self, *new_qubits: cirq.Qid) -> NongateOperation:
            return NongateOperation(new_qubits)

        def __eq__(self, other):
            return isinstance(other, NongateOperation) and self._qubits == other._qubits

    op_qx = NongateOperation((qx,))
    assert cirq.is_parameterized(op_qx)
    assert cirq.parameter_names(op_qx) == {'x'}
    resolver = cirq.ParamResolver({x: cirq.LineQubit(3)})
    resolved_op_qx = cirq.resolve_parameters(op_qx, resolver)
    assert resolved_op_qx == NongateOperation((cirq.LineQubit(3),))

    q0 = cirq.LineQubit(0)
    op_q0 = NongateOperation((q0,))
    assert not cirq.is_parameterized(op_q0)
    assert cirq.parameter_names(op_q0) == set()
    resolved_op_q0 = cirq.Operation._resolve_parameters_(op_q0, resolver, recursive=True)
    assert resolved_op_q0 == NongateOperation((cirq.LineQubit(0),))


def test_variable_qid_resolution_gate_operation():
    x = sympy.Symbol('x')
    qx = cirq.VariableQid(x)

    op_1 = cirq.X(cirq.LineQubit(1))
    assert not cirq.is_parameterized(op_1)
    assert cirq.parameter_names(op_1) == set()
    resolver = cirq.ParamResolver({x: cirq.LineQubit(3)})
    resolved_op = cirq.resolve_parameters(op_1, resolver)
    assert resolved_op == op_1

    op_x = cirq.X(qx)
    assert cirq.is_parameterized(op_x)
    assert cirq.parameter_names(op_x) == {'x'}
    resolver = cirq.ParamResolver({x: cirq.LineQubit(3)})
    resolved_op = cirq.resolve_parameters(op_x, resolver)
    assert resolved_op == cirq.X(cirq.LineQubit(3))

    theta = sympy.Symbol('theta')
    op_xpow = cirq.XPowGate(exponent=theta)(qx)
    assert cirq.is_parameterized(op_xpow)
    assert cirq.parameter_names(op_xpow) == {'x', 'theta'}
    resolver = cirq.ParamResolver({x: cirq.LineQubit(3), theta: 0.5})
    resolved_op = cirq.resolve_parameters(op_xpow, resolver)
    assert resolved_op == cirq.XPowGate(exponent=0.5)(cirq.LineQubit(3))


def test_variable_qid_resolution_controlled_operation():
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    q0 = cirq.LineQubit(0)
    qx = cirq.VariableQid(x)
    qy = cirq.VariableQid(y)

    cx_x0 = cirq.X(qx).controlled_by(q0)
    cx_0x = cirq.X(q0).controlled_by(qx)
    cx_xy = cirq.X(qx).controlled_by(qy)

    assert cirq.is_parameterized(cx_x0)
    assert cirq.is_parameterized(cx_0x)
    assert cirq.is_parameterized(cx_xy)

    assert cirq.parameter_names(cx_x0) == {'x'}
    assert cirq.parameter_names(cx_0x) == {'x'}
    assert cirq.parameter_names(cx_xy) == {'x', 'y'}

    resolver = cirq.ParamResolver({x: cirq.LineQubit(1), y: cirq.LineQubit(2)})
    resolved_cx_x0 = cirq.resolve_parameters(cx_x0, resolver)
    assert resolved_cx_x0 == cirq.X(cirq.LineQubit(1)).controlled_by(cirq.LineQubit(0))
    resolved_cx_0x = cirq.resolve_parameters(cx_0x, resolver)
    assert resolved_cx_0x == cirq.X(cirq.LineQubit(0)).controlled_by(cirq.LineQubit(1))
    resolved_cx_xy = cirq.resolve_parameters(cx_xy, resolver)
    assert resolved_cx_xy == cirq.X(cirq.LineQubit(1)).controlled_by(cirq.LineQubit(2))


def test_variable_qid_resolution_circuit_operation():
    x = sympy.Symbol('x')
    q0 = cirq.LineQubit(0)
    qx = cirq.VariableQid(x)

    x_qx = cirq.X(qx)
    x_q0 = cirq.X(q0)
    map_00 = {q0: q0}
    map_0x = {q0: qx}
    map_x0 = {qx: q0}
    circuit_x_qx_map00 = cirq.CircuitOperation(cirq.FrozenCircuit(x_qx), qubit_map=map_00)
    circuit_x_q0_map0x = cirq.CircuitOperation(cirq.FrozenCircuit(x_q0), qubit_map=map_0x)
    circuit_x_qx_mapx0 = cirq.CircuitOperation(cirq.FrozenCircuit(x_qx), qubit_map=map_x0)

    assert cirq.is_parameterized(circuit_x_qx_map00)
    assert cirq.is_parameterized(circuit_x_q0_map0x)
    assert not cirq.is_parameterized(circuit_x_qx_mapx0)

    assert cirq.parameter_names(circuit_x_qx_map00) == {'x'}
    assert cirq.parameter_names(circuit_x_q0_map0x) == {'x'}
    assert cirq.parameter_names(circuit_x_qx_mapx0) == set()

    resolver = cirq.ParamResolver({x: cirq.LineQubit(3)})

    resolved_circuit_x_qx_map00 = cirq.resolve_parameters(circuit_x_qx_map00, resolver)
    assert resolved_circuit_x_qx_map00._mapped_any_loop == cirq.Circuit(cirq.X(cirq.LineQubit(3)))

    resolved_circuit_x_q0_map0x = cirq.resolve_parameters(circuit_x_q0_map0x, resolver)
    assert resolved_circuit_x_q0_map0x._mapped_any_loop == cirq.Circuit(cirq.X(cirq.LineQubit(3)))

    resolved_circuit_x_qx_mapx0 = cirq.resolve_parameters(circuit_x_qx_mapx0, resolver)
    assert resolved_circuit_x_qx_mapx0._mapped_any_loop == cirq.Circuit(cirq.X(cirq.LineQubit(0)))


def test_variable_qid_resolution_PauliString():
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    q0 = cirq.LineQubit(0)
    q1 = cirq.LineQubit(1)
    qx = cirq.VariableQid(x)
    qy = cirq.VariableQid(y)

    qxqy_xy = cirq.PauliString({qx: cirq.X, qy: cirq.Y})
    assert cirq.is_parameterized(qxqy_xy)
    assert cirq.parameter_names(qxqy_xy) == {'x', 'y'}

    resolved = cirq.resolve_parameters(qxqy_xy, cirq.ParamResolver({x: q0, y: q1}))
    assert resolved == cirq.PauliString({q0: cirq.X, q1: cirq.Y})

    with pytest.raises(ValueError, match="Duplicate qubits during parameter resolution"):
        _ = cirq.resolve_parameters(qxqy_xy, cirq.ParamResolver({x: q1, y: q1}))


def test_variable_qid_addition_two_vqid():
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    qx = cirq.VariableQid(x)
    qy = cirq.VariableQid(y)

    qxy_plus = qx + qy
    qxy_minus = qx - qy

    assert qxy_plus.symbol == x + y
    assert qxy_minus.symbol == x - y

    resolver = cirq.ParamResolver({x: cirq.LineQubit(10), y: cirq.LineQubit(1)})
    assert cirq.resolve_parameters(qxy_plus, resolver) == cirq.LineQubit(11)
    assert cirq.resolve_parameters(qxy_minus, resolver) == cirq.LineQubit(9)

    resolver = cirq.ParamResolver({x: cirq.GridQubit(10, 20), y: cirq.GridQubit(1, 2)})
    assert cirq.resolve_parameters(qxy_plus, resolver) == cirq.GridQubit(11, 22)
    assert cirq.resolve_parameters(qxy_minus, resolver) == cirq.GridQubit(9, 18)

    resolver = cirq.ParamResolver({x: cirq.LineQubit(10), y: cirq.GridQubit(10, 20)})
    with pytest.raises(TypeError, match="unsupported operand type"):
        cirq.resolve_parameters(qxy_plus, resolver)

    with pytest.raises(TypeError, match="unsupported operand type"):
        _ = qx + cirq.LineQubit(10)
    with pytest.raises(TypeError, match="unsupported operand type"):
        _ = qx - cirq.LineQubit(10)
    with pytest.raises(TypeError, match="unsupported operand type"):
        _ = "bob" - qx

    with pytest.raises(TypeError, match="Can only add VariableQids with identical dimension"):
        _ = cirq.VariableQid(x) + cirq.VariableQid(y, dimension=3)

    with pytest.raises(TypeError, match="Can only subtract VariableQids with identical dimension"):
        _ = cirq.VariableQid(x) - cirq.VariableQid(y, dimension=3)


def test_variable_qid_addition_vqid_expr():
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    qxy_plus = cirq.VariableQid(x) + y
    qxy_minus = cirq.VariableQid(x) - y

    assert qxy_plus.symbol == x + y
    assert qxy_minus.symbol == x - y

    resolver = cirq.ParamResolver({x: cirq.LineQubit(1), y: cirq.LineQubit(10)})
    assert cirq.resolve_parameters(qxy_plus, resolver) == cirq.LineQubit(11)
    assert cirq.resolve_parameters(qxy_minus, resolver) == cirq.LineQubit(-9)

    resolver = cirq.ParamResolver({x: cirq.GridQubit(1, 2), y: cirq.GridQubit(10, 20)})
    assert cirq.resolve_parameters(qxy_plus, resolver) == cirq.GridQubit(11, 22)
    assert cirq.resolve_parameters(qxy_minus, resolver) == cirq.GridQubit(-9, -18)

    qxy_rplus = x + cirq.VariableQid(y)
    qxy_rminus = x - cirq.VariableQid(y)
    assert qxy_plus == qxy_rplus
    assert qxy_minus == qxy_rminus


def test_variable_qid_multiplication():
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    z = sympy.Symbol('z')
    qx = cirq.VariableQid(x)
    qy = cirq.VariableQid(y)

    qxy_prod = qx * qy
    qxz_prod = qx * z
    qx2_prod = qx * 2
    q2x_prod = 2 * qx

    resolver = cirq.ParamResolver(
        {x: cirq.LineQubit(2), y: cirq.LineQubit(3), z: cirq.LineQubit(4)}
    )
    assert cirq.resolve_parameters(qxy_prod, resolver) == cirq.LineQubit(6)
    assert cirq.resolve_parameters(qxz_prod, resolver) == cirq.LineQubit(8)
    assert cirq.resolve_parameters(qx2_prod, resolver) == cirq.LineQubit(4)
    assert cirq.resolve_parameters(q2x_prod, resolver) == cirq.LineQubit(4)

    resolver = cirq.ParamResolver(
        {x: cirq.GridQubit(2, 3), y: cirq.GridQubit(4, 5), z: cirq.GridQubit(6, 7)}
    )
    assert cirq.resolve_parameters(qxy_prod, resolver) == cirq.GridQubit(8, 15)
    assert cirq.resolve_parameters(qxz_prod, resolver) == cirq.GridQubit(12, 21)
    assert cirq.resolve_parameters(qx2_prod, resolver) == cirq.GridQubit(4, 6)
    assert cirq.resolve_parameters(q2x_prod, resolver) == cirq.GridQubit(4, 6)

    with pytest.raises(TypeError, match="Can only multiply VariableQids with identical dimension"):
        _ = qx * cirq.VariableQid(y, dimension=3)

    with pytest.raises(TypeError, match="unsupported operand"):
        _ = qx * cirq.LineQubit(3)


def test_variable_qid_neg():
    x = sympy.Symbol('x')
    qx = cirq.VariableQid(x)
    nqx = -qx

    resolver = cirq.ParamResolver({x: cirq.LineQubit(1)})
    assert cirq.resolve_parameters(nqx, resolver) == cirq.LineQubit(-1)


def test_repr():
    x = sympy.Symbol('x')
    qx = cirq.VariableQid(x)
    assert repr(qx) == 'cirq.VariableQid(sympy.Symbol(\'x\'), dimension=2)'

    qxy = qx + sympy.Symbol('y')
    assert (
        repr(qxy)
        == 'cirq.VariableQid(sympy.Add(sympy.Symbol(\'x\'), sympy.Symbol(\'y\')), dimension=2)'
    )


def test_str():
    x = sympy.Symbol('x')
    qx = cirq.VariableQid(x)
    assert str(qx) == 'varq(x) (d=2)'

    qxy = qx + sympy.Symbol('y')
    assert str(qxy) == 'varq(x + y) (d=2)'


def test_variable_qid_simulation():
    q0 = cirq.LineQubit(0)
    q1 = cirq.LineQubit(1)
    qx = cirq.VariableQid('x')
    circuit = cirq.Circuit((cirq.Moment(cirq.X(q1)), cirq.Moment(cirq.measure(qx, key='m'))))
    sim = cirq.Simulator()
    result = sim.run_sweep(circuit, params=[{'x': cirq.LineQubit(0)}, {'x': cirq.LineQubit(1)}])
    assert result[0].records['m'][0, 0, 0] == 0
    assert result[1].records['m'][0, 0, 0] == 1

    circuit = cirq.Circuit(cirq.X(q0), cirq.measure(qx, key='m'))
    resolver = cirq.ParamResolver({'x': q0})
    sim = cirq.Simulator()
    with pytest.raises(ValueError, match="Overlapping operations"):
        _ = sim.run(circuit, resolver)
