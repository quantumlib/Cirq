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


def test_construction_unparameterized() -> None:
    q = cirq.q(0)
    c = cirq.Circuit(cirq.X(q))
    cf = cirq.CircuitFunction("test_function", c)
    assert cf.name == "test_function"
    assert cf.circuit == c.freeze()
    assert cf.function_params == ()
    assert cirq.parameter_names(cf) == set()
    assert not cirq.is_parameterized(cf)
    assert cf.qubits == frozenset((q,))

    # expose nonfunctional parameter
    theta = sympy.Symbol('theta')
    cf = cirq.CircuitFunction("test_function", c, function_params=[theta])
    assert cf.function_params == (theta,)
    assert cirq.parameter_names(cf) == set()
    assert not cirq.is_parameterized(cf)


def test_construction_basic() -> None:
    # parameter exposed
    q = cirq.q(0)
    theta = sympy.Symbol('theta')
    c = cirq.Circuit(cirq.X(q) ** theta)
    cf = cirq.CircuitFunction("test_function", c, function_params=[theta])
    assert cf.name == "test_function"
    assert cf.circuit == c.freeze()
    assert cf.function_params == (theta,)
    assert cirq.parameter_names(cf) == set()
    assert not cirq.is_parameterized(cf)
    assert cf.qubits == frozenset((q,))

    cf = cirq.CircuitFunction("test_function", c)
    assert cf.function_params == (theta,)

    # parameter not exposed
    cf = cirq.CircuitFunction("test_function", c, function_params=[])
    assert cf.function_params == ()
    assert cirq.parameter_names(cf) == {theta.name}
    assert cirq.is_parameterized(cf)

    with pytest.raises(TypeError, match='Function name must be a string'):
        _ = cirq.CircuitFunction(10, c)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match='Function name must be a non-empty string'):
        _ = cirq.CircuitFunction("", c)

    with pytest.raises(TypeError, match='Expected circuit of type AbstractCircuit'):
        _ = cirq.CircuitFunction("test_function", 5)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match='All parameters must be sympy Symbols.'):
        _ = cirq.CircuitFunction("test_function", c, function_params=[5])

    with pytest.raises(ValueError, match='CircuitFunctions may not have duplicate parameters.'):
        _ = cirq.CircuitFunction("test_function", c, function_params=[theta, theta])


def test_construction_vqid() -> None:
    # parameter exposed, vqid not exposed
    x = sympy.Symbol('x')
    q = cirq.VariableLineQid(x)
    theta = sympy.Symbol('theta')
    c = cirq.Circuit(cirq.X(q) ** theta)
    cf = cirq.CircuitFunction("test_function", c, function_params=[theta])
    assert cf.name == "test_function"
    assert cf.circuit == c.freeze()
    assert cf.function_params == (theta,)
    assert cirq.parameter_names(cf) == {x.name}
    assert cirq.is_parameterized(cf)
    assert cf.qubits == frozenset((q,))

    # parameter exposed, vqid exposed
    cf = cirq.CircuitFunction("test_function", c, function_params=[theta, x])
    assert cf.function_params == (theta, x)
    assert cirq.parameter_names(cf) == set()
    assert not cirq.is_parameterized(cf)

    cf = cirq.CircuitFunction("test_function", c)
    assert cf.function_params == (theta, x)

    # parameter not exposed, vqid exposed
    cf = cirq.CircuitFunction("test_function", c, function_params=[x])
    assert cf.function_params == (x,)
    assert cirq.parameter_names(cf) == {theta.name}
    assert cirq.is_parameterized(cf)

    # parameter not exposed, vqid not exposed
    cf = cirq.CircuitFunction("test_function", c, function_params=[])
    assert cf.function_params == ()
    assert cirq.parameter_names(cf) == {theta.name, x.name}
    assert cirq.is_parameterized(cf)


def test_repr() -> None:
    x = sympy.Symbol('x')
    qx = cirq.VariableLineQid(x)
    q0 = cirq.q(0)
    theta = sympy.Symbol('theta')
    cx = cirq.Circuit(cirq.X(qx) ** theta)
    c0 = cirq.Circuit(cirq.X(q0) ** theta)
    cirq.testing.assert_equivalent_repr(
        cirq.CircuitFunction("test_function", c0, function_params=[theta])
    )
    cirq.testing.assert_equivalent_repr(
        cirq.CircuitFunction("test_function", cx, function_params=[])
    )
    cirq.testing.assert_equivalent_repr(
        cirq.CircuitFunction("test_function", cx, function_params=[theta])
    )
    cirq.testing.assert_equivalent_repr(
        cirq.CircuitFunction("test_function", cx, function_params=[theta, x])
    )


def test_str() -> None:
    x = sympy.Symbol('x')
    qx = cirq.VariableLineQid(x)
    theta = sympy.Symbol('theta')
    cx = cirq.Circuit(cirq.X(qx) ** theta)
    assert str(cirq.CircuitFunction("test_function", cx, function_params=[])) == "test_function()"
    assert (
        str(cirq.CircuitFunction("test_function", cx, function_params=[theta]))
        == "test_function(theta)"
    )
    assert str(cirq.CircuitFunction("test_function", cx)) == "test_function(theta, x)"


def test_eq() -> None:
    x = sympy.Symbol('x')
    qx = cirq.VariableLineQid(x)
    theta = sympy.Symbol('theta')
    cx = cirq.Circuit(cirq.X(qx) ** theta)

    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(
        cirq.CircuitFunction("test_function", cx),
        cirq.CircuitFunction("test_function", cx, function_params=[theta, x]),
    )
    eq.add_equality_group(cirq.CircuitFunction("test_function2", cx))
    eq.add_equality_group(
        cirq.CircuitFunction("test_function", cx, function_params=[]),
        cirq.CircuitFunction("test_function", cx.freeze(), function_params=[]),
    )


def test_call() -> None:
    x = sympy.Symbol('x')
    qx = cirq.VariableLineQid(x)
    theta = sympy.Symbol('theta')
    cx = cirq.Circuit(cirq.X(qx) ** theta)
    cf = cirq.CircuitFunction("test_function", cx, function_params=[x, theta])

    ref_circuit = cirq.Circuit(cirq.X(cirq.q(2)) ** 3).freeze()
    assert cf(2, 3) == ref_circuit

    with pytest.raises(
        TypeError, match="CircuitFunction test_function called with the wrong number of parameters"
    ):
        _ = cf(2)
