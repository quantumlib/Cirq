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
        TypeError, match="Only sympy expressions or strings are supported for cirq.VariableQid"
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
        match=r"Resolved Qid dimension \(2\) does not match the cirq.VariableQid dimension \(3\)",
    ):
        cirq.resolve_parameters(qx, resolver)


def test_variable_qid_resolved_value():
    qx = cirq.VariableQid(sympy.Symbol('x'))
    assert qx._resolved_value_() == NotImplemented


def test_variable_qid_circuit_diagram_info():
    x = sympy.Symbol('x')
    qx = cirq.cirq.VariableQid(x, dimension=3)

    info = cirq.circuit_diagram_info(qx)
    assert info == cirq.CircuitDiagramInfo(wire_symbols=('x (d=3)',))


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

    with pytest.raises(TypeError, match="Can only add cirq.VariableQids with identical dimension"):
        _ = cirq.VariableQid(x) + cirq.VariableQid(y, dimension=3)

    with pytest.raises(TypeError, match="Can only subtract cirq.VariableQids with identical dimension"):
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

    with pytest.raises(TypeError, match="Can only multiply cirq.VariableQids with identical dimension"):
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
    assert repr(qx) == 'cirq.cirq.VariableQid(sympy.Symbol(\'x\'), dimension=2)'

    qxy = qx + sympy.Symbol('y')
    assert (
        repr(qxy)
        == 'cirq.cirq.VariableQid(sympy.Add(sympy.Symbol(\'x\'), sympy.Symbol(\'y\')), dimension=2)'
    )


def test_str():
    x = sympy.Symbol('x')
    qx = cirq.VariableQid(x)
    assert str(qx) == 'varq(x) (d=2)'

    qxy = qx + sympy.Symbol('y')
    assert str(qxy) == 'varq(x + y) (d=2)'
