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

    with pytest.raises(ValueError, match=r"VariableLineQid \(1\) is fully resolved already."):
        _ = cirq.VariableLineQid(sympy.Integer(1))


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

    qx3 = cirq.VariableLineQid(x, dimension=3)
    assert cirq.resolve_parameters(qx3, resolver) == cirq.LineQid(3, dimension=3)

    # Unresolved (not in resolver)
    resolver = cirq.ParamResolver({})
    assert cirq.resolve_parameters(qx, resolver) == qx

    # Error (resolves to non-integer)
    resolver = cirq.ParamResolver({x: 5.3})
    with pytest.raises(ValueError, match="Could not resolve expression 5.3 to a LineQid"):
        _ = cirq.resolve_parameters(qx, resolver)

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
    assert str(qx) == 'v(x) (d=2)'

    xplusy = x + sympy.Symbol('y')
    qxplusy = cirq.VariableLineQid(xplusy)
    assert str(qxplusy) == 'v(x + y) (d=2)'


def test_variable_grid_qid_init():
    r = sympy.Symbol('r')
    c = sympy.Symbol('c')
    q = cirq.VariableGridQid(r, c)
    assert q.row == r
    assert q.col == c
    assert q.dimension == 2

    with pytest.raises(ValueError, match=r"VariableGridQid \(1, 2\) is fully resolved already."):
        _ = cirq.VariableGridQid(1, 2)

    with pytest.raises(ValueError, match=r"VariableGridQid \(1, 2\) is fully resolved already."):
        _ = cirq.VariableGridQid(sympy.Integer(1), sympy.Float(2))

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

    q3 = cirq.VariableGridQid(r, c, dimension=3)
    assert cirq.resolve_parameters(q3, resolver) == cirq.GridQid(1, 2, dimension=3)

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
    assert str(q) == "v(r, c) (d=2)"


def test_variable_grid_qid_simulation():
    """VariableQid does not work with simulator sweeps

    Since the addition of SetVariable, VariableQids can not be
    resolved during a simulator sweep. If we enable runtime resolution
    of VariableQid, this test should be changed.
    """
    r, c = sympy.symbols('r c')
    q = cirq.VariableGridQid(r, c)
    q00 = cirq.GridQubit(0, 0)

    statevec_sim = cirq.Simulator(split_untangled_states=False)
    prod_statevec_sim = cirq.Simulator()
    dm_sim = cirq.DensityMatrixSimulator()
    sims = [statevec_sim, prod_statevec_sim, dm_sim]

    circuit = cirq.Circuit(cirq.Moment(cirq.X(q00)), cirq.Moment(cirq.measure(q, key='m')))
    for sim in sims:
        results = sim.run_sweep(circuit, params=[{'r': 0, 'c': 0}, {'r': 1, 'c': 2}])
        assert results[0].measurements['m'] == 1
        assert results[1].measurements['m'] == 0

    circuit = cirq.Circuit(
        cirq.Moment(cirq.X(q00)),
        cirq.Moment(cirq.SetVariable(r, 0), cirq.SetVariable(c, 0)),
        cirq.Moment(cirq.measure(q, key='m')),
    )
    for sim in sims:
        results = sim.run(circuit)
        assert results.measurements['m'] == 1

    circuit = cirq.Circuit(
        cirq.Moment(cirq.X(q00)),
        cirq.Moment(cirq.SetVariable(r, 0), cirq.SetVariable(c, 1)),
        cirq.Moment(cirq.measure(q, key='m')),
    )
    for sim in sims:
        results = statevec_sim.run(circuit)
        assert results.measurements['m'] == 0

    # check that unresolved variableqids cause an error
    circuit = cirq.Circuit(cirq.X(q), cirq.measure(q00, key='m'))
    for sim in sims:
        with pytest.raises(
            ValueError, match="Circuit contains ops whose symbols were not specified"
        ):
            sim.run(circuit)
