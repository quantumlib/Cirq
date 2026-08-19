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


def test_variable_line_qid_init():
    x = sympy.Symbol('x')
    qx = cirq.VariableLineQid(x)
    assert qx.x == x
    assert qx.dimension == 2

    qx_d3 = cirq.VariableLineQid(x, dimension=3)
    assert qx_d3.dimension == 3

    with pytest.raises(ValueError, match="Expected a positive integer"):
        _ = cirq.VariableLineQid(x, dimension=-1)

    with pytest.raises(TypeError, match="Only sympy expressions are supported for VariableLineQid"):
        _ = cirq.VariableLineQid([1, 2])


def test_variable_line_qid_comparison():
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    qx = cirq.VariableLineQid(x)
    qy = cirq.VariableLineQid(y)

    assert qx < qy
    assert qx == cirq.VariableLineQid(x)

    qx_d3 = cirq.VariableLineQid(x, dimension=3)
    assert qx < qx_d3


def test_variable_line_qid_parameterization():
    x = sympy.Symbol('x')
    qx = cirq.VariableLineQid(x)
    assert cirq.is_parameterized(qx)
    assert cirq.parameter_names(qx) == {'x'}


def test_variable_line_qid_with_dimension():
    x = sympy.Symbol('x')
    qx = cirq.VariableLineQid(x)
    assert qx.with_dimension(2) == qx

    qx_d3 = qx.with_dimension(3)
    assert qx_d3.dimension == 3


def test_variable_line_qid_basic_resolution():
    x = sympy.Symbol('x')
    qx = cirq.VariableLineQid(x)

    resolver = cirq.ParamResolver({x: 3})
    assert cirq.resolve_parameters(qx, resolver) == cirq.LineQubit(3)

    # Unresolved (not in resolver)
    resolver = cirq.ParamResolver({})
    assert cirq.resolve_parameters(qx, resolver) == qx

    # Error (resolves to non-integer)
    resolver = cirq.ParamResolver({x: 5.3})
    with pytest.raises(ValueError, match="Could not resolve expression 5.3 to a LineQid"):
        _ = cirq.resolve_parameters(qx, resolver)

    # unresolved - cannot resolve constant
    y = sympy.Symbol('y')
    resolver = cirq.ParamResolver({y: 3})
    q1 = cirq.VariableLineQid(x - x + 1)
    assert cirq.resolve_parameters(q1, resolver) == cirq.LineQubit(1)

    # partial resolution
    resolver = cirq.ParamResolver({x: 3})
    qxy = cirq.VariableLineQid(sympy.Symbol('x') + sympy.Symbol('y'))
    qref = cirq.VariableLineQid(sympy.Symbol('y') + 3)
    assert cirq.resolve_parameters(qxy, resolver) == qref


def test_variable_line_qid_expression_resolution():
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    x1 = x + 1
    qx1 = cirq.VariableLineQid(x1)
    qxy = cirq.VariableLineQid(x + y)

    resolver = cirq.ParamResolver({x: 3})
    assert cirq.resolve_parameters(qx1, resolver) == cirq.LineQubit(4)

    resolver = cirq.ParamResolver({"x": 3})
    assert cirq.resolve_parameters(qxy, resolver) == cirq.VariableLineQid(y + 3)


def test_variable_line_qid_circuit_diagram_info():
    x = sympy.Symbol('x')
    qx = cirq.VariableLineQid(x, dimension=3)

    info = cirq.circuit_diagram_info(qx)
    assert info == cirq.CircuitDiagramInfo(wire_symbols=('x (d=3)',))


def test_variable_line_qid_resolution_operation():
    x = sympy.Symbol('x')
    qx = cirq.VariableLineQid(x)

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
    resolver = cirq.ParamResolver({x: 3})
    resolved_op_qx = cirq.resolve_parameters(op_qx, resolver)
    assert resolved_op_qx == NongateOperation((cirq.LineQubit(3),))


def test_variable_line_qid_resolution_gate_operation():
    x = sympy.Symbol('x')
    qx = cirq.VariableLineQid(x)

    op_1 = cirq.X(cirq.LineQubit(1))
    assert not cirq.is_parameterized(op_1)
    assert cirq.parameter_names(op_1) == set()
    resolver = cirq.ParamResolver({x: 3})
    resolved_op = cirq.resolve_parameters(op_1, resolver)
    assert resolved_op == op_1

    op_x = cirq.X(qx)
    assert cirq.is_parameterized(op_x)
    assert cirq.parameter_names(op_x) == {'x'}
    resolver = cirq.ParamResolver({x: 3})
    resolved_op = cirq.resolve_parameters(op_x, resolver)
    assert resolved_op == cirq.X(cirq.LineQubit(3))


def test_variable_line_qid_resolution_controlled_operation():
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    q0 = cirq.LineQubit(0)
    qx = cirq.VariableLineQid(x)
    qy = cirq.VariableLineQid(y)

    cx_x0 = cirq.X(qx).controlled_by(q0)
    cx_0x = cirq.X(q0).controlled_by(qx)
    cx_xy = cirq.X(qx).controlled_by(qy)

    assert cirq.is_parameterized(cx_x0)
    assert cirq.is_parameterized(cx_0x)
    assert cirq.is_parameterized(cx_xy)

    assert cirq.parameter_names(cx_x0) == {'x'}
    assert cirq.parameter_names(cx_0x) == {'x'}
    assert cirq.parameter_names(cx_xy) == {'x', 'y'}

    resolver = cirq.ParamResolver({x: 1, y: 2})
    resolved_cx_x0 = cirq.resolve_parameters(cx_x0, resolver)
    assert resolved_cx_x0 == cirq.X(cirq.LineQubit(1)).controlled_by(cirq.LineQubit(0))
    resolved_cx_0x = cirq.resolve_parameters(cx_0x, resolver)
    assert resolved_cx_0x == cirq.X(cirq.LineQubit(0)).controlled_by(cirq.LineQubit(1))
    resolved_cx_xy = cirq.resolve_parameters(cx_xy, resolver)
    assert resolved_cx_xy == cirq.X(cirq.LineQubit(1)).controlled_by(cirq.LineQubit(2))


def test_variable_line_qid_resolution_circuit_operation():
    x = sympy.Symbol('x')
    q0 = cirq.LineQubit(0)
    qx = cirq.VariableLineQid(x)

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

    resolver = cirq.ParamResolver({x: 3})
    resolved_circuit_x_qx_map00 = cirq.resolve_parameters(circuit_x_qx_map00, resolver)
    assert resolved_circuit_x_qx_map00._mapped_any_loop == cirq.Circuit(cirq.X(cirq.LineQubit(3)))


def test_variable_line_qid_resolution_PauliString():
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    q0 = cirq.LineQubit(0)
    q1 = cirq.LineQubit(1)
    qx = cirq.VariableLineQid(x)
    qy = cirq.VariableLineQid(y)

    qxqy_xy = cirq.PauliString({qx: cirq.X, qy: cirq.Y})
    assert cirq.is_parameterized(qxqy_xy)
    assert cirq.parameter_names(qxqy_xy) == {'x', 'y'}

    resolved = cirq.resolve_parameters(qxqy_xy, cirq.ParamResolver({x: 0, y: 1}))
    assert resolved == cirq.PauliString({q0: cirq.X, q1: cirq.Y})

    with pytest.raises(ValueError, match="Duplicate qubits"):
        cirq.resolve_parameters(qxqy_xy, cirq.ParamResolver({x: 0, y: 0}))


def test_variable_line_qid_repr():
    x = sympy.Symbol('x')
    qx = cirq.VariableLineQid(x)
    assert repr(qx) == "cirq.VariableLineQid(sympy.Symbol('x'), dimension=2)"

    xplusy = x + sympy.Symbol('y')
    qxplusy = cirq.VariableLineQid(xplusy)
    assert (
        repr(qxplusy)
        == "cirq.VariableLineQid(sympy.Add(sympy.Symbol('x'), sympy.Symbol('y')), dimension=2)"
    )


def test_variable_line_qid_str():
    x = sympy.Symbol('x')
    qx = cirq.VariableLineQid(x)
    assert str(qx) == 'varq(x) (d=2)'

    xplusy = x + sympy.Symbol('y')
    qxplusy = cirq.VariableLineQid(xplusy)
    assert str(qxplusy) == 'varq(x + y) (d=2)'


def test_variable_grid_qid_init():
    r = sympy.Symbol('r')
    c = sympy.Symbol('c')
    q = cirq.VariableGridQid(r, c)
    assert q.row == r
    assert q.col == c
    assert q.dimension == 2

    with pytest.raises(ValueError, match=r"VariableGridQid \(1, 2\) is fully resolved already."):
        _ = cirq.VariableGridQid(1, 2)

    q_d3 = cirq.VariableGridQid(r, c, dimension=3)
    assert q_d3.dimension == 3

    with pytest.raises(ValueError, match="Expected a positive integer"):
        _ = cirq.VariableGridQid(r, c, dimension=-1)

    with pytest.raises(
        TypeError, match="Only sympy expressions or ints are supported for VariableGridQid row/col"
    ):
        _ = cirq.VariableGridQid([1], c)


def test_variable_grid_qid_comparison():
    r1, r2 = sympy.symbols('r1 r2')
    c1, c2 = sympy.symbols('c1 c2')
    q11 = cirq.VariableGridQid(r1, c1)
    q12 = cirq.VariableGridQid(r1, c2)
    q21 = cirq.VariableGridQid(r2, c1)
    q13 = cirq.VariableGridQid(r1, 3)

    assert q11 == cirq.VariableGridQid(r1, c1)
    assert q11 < q12
    assert q12 < q21
    assert q13 < q11

    q11_d3 = cirq.VariableGridQid(r1, c1, dimension=3)
    assert q11 < q11_d3


def test_variable_grid_qid_parameterization():
    r, c = sympy.symbols('r c')
    q = cirq.VariableGridQid(r, c)
    assert cirq.is_parameterized(q)
    assert cirq.parameter_names(q) == {'r', 'c'}


def test_variable_grid_qid_with_dimension():
    r, c = sympy.symbols('r c')
    q = cirq.VariableGridQid(r, c)
    assert q.with_dimension(2) == q

    q_d3 = q.with_dimension(3)
    assert q_d3.dimension == 3


def test_variable_grid_qid_resolution():
    r, c = sympy.symbols('r c')
    q = cirq.VariableGridQid(r, c)

    resolver = cirq.ParamResolver({r: 1, "c": 2})
    assert cirq.resolve_parameters(q, resolver) == cirq.GridQubit(1, 2)

    resolver = cirq.ParamResolver({r: 1})
    assert cirq.resolve_parameters(q, resolver) == cirq.VariableGridQid(1, c)

    # Unresolved (not in resolver)
    resolver = cirq.ParamResolver({})
    assert cirq.resolve_parameters(q, resolver) == q

    # Resolution failure
    resolver = cirq.ParamResolver({r: 1.5, c: 2})
    with pytest.raises(ValueError, match="Could not resolve expression"):
        _ = cirq.resolve_parameters(q, resolver)

    q = cirq.VariableGridQid(r + 1, c * 2)
    resolver = cirq.ParamResolver({r: 3, c: 4})
    assert cirq.resolve_parameters(q, resolver) == cirq.GridQubit(4, 8)


def test_variable_grid_qid_circuit_diagram_info():
    r, c = sympy.symbols('r c')
    q = cirq.VariableGridQid(r, c, dimension=3)

    info = cirq.circuit_diagram_info(q)
    assert info == cirq.CircuitDiagramInfo(wire_symbols=('(r, c) (d=3)',))


def test_variable_grid_qid_repr_and_str():
    r, c = sympy.symbols('r c')
    q = cirq.VariableGridQid(r, c)
    assert repr(q) == "cirq.VariableGridQid(sympy.Symbol('r'), sympy.Symbol('c'), dimension=2)"
    assert str(q) == "varq(r, c) (d=2)"


def test_variable_grid_qid_simulation():
    """VariableQid does not work with simulator sweeps

    Since the addition of SetVariable, VariableQids can not be
    resolved during a simulator sweep. If we enable runtime resolution
    of VariableQid, this test should be changed.
    """
    r, c = sympy.symbols('r c')
    q = cirq.VariableGridQid(r, c)
    q00 = cirq.GridQubit(0, 0)
    circuit = cirq.Circuit(cirq.Moment(cirq.X(q00)), cirq.Moment(cirq.measure(q, key='m')))
    sim = cirq.Simulator()
    with pytest.raises(ValueError):
        _ = sim.run_sweep(circuit, params=[{'r': 0, 'c': 0}, {'r': 1, 'c': 2}])
