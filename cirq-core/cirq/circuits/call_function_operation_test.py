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


def test_construction_basic() -> None:
    q = cirq.q(0)
    theta = sympy.Symbol('theta')
    c = cirq.Circuit(cirq.X(q) ** theta)
    cf = cirq.CircuitFunction("test_function", c, function_params=[theta])

    op = cirq.CallFunctionOperation(cf, [0.5])
    assert op.circuit_function == cf
    assert op.function_args == (0.5,)
    assert cirq.parameter_names(op) == set()
    assert op.param_resolver == cirq.ParamResolver()
    assert op.qubits == (q,)

    with pytest.raises(TypeError, match="Expected circuit_function of type CircuitFunction"):
        _ = cirq.CallFunctionOperation(c, [0.5])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Expected 1 parameters"):
        _ = cirq.CallFunctionOperation(cf, [0.5, 2])


def test_construction_parameterized() -> None:
    x = sympy.Symbol('x')
    q = cirq.VariableLineQid(x)
    theta = sympy.Symbol('theta')
    c = cirq.Circuit(cirq.X(q) ** theta)
    cf = cirq.CircuitFunction("test_function", c, function_params=[theta])

    op = cirq.CallFunctionOperation(cf, [0.5], param_resolver={x: 1})
    assert op.circuit_function == cf
    assert op.function_args == (0.5,)
    assert cirq.parameter_names(op) == set()
    assert op.param_resolver == cirq.ParamResolver({x: 1})
    assert not cirq.is_parameterized(op)
    assert op.qubits == (cirq.q(1),)


def test_parameterized() -> None:
    x = sympy.Symbol('x')
    q = cirq.VariableLineQid(x)
    theta = sympy.Symbol('theta')
    c = cirq.Circuit(cirq.X(q) ** theta)

    cf = cirq.CircuitFunction("test_function", c, function_params=[theta, x])
    op = cirq.CallFunctionOperation(cf, [0.5, 1])
    assert not cirq.is_parameterized(op)

    cf = cirq.CircuitFunction("test_function", c, function_params=[theta])
    op = cirq.CallFunctionOperation(cf, [0.5])
    assert cirq.is_parameterized(op)


def test_parameter_resolution() -> None:
    x = sympy.Symbol('x')
    q = cirq.VariableLineQid(x)
    theta = sympy.Symbol('theta')
    c = cirq.Circuit(cirq.X(q) ** theta)
    cf = cirq.CircuitFunction("test_function", c, function_params=[theta])

    op = cirq.CallFunctionOperation(cf, [0.5])
    op_resolved = cirq.resolve_parameters(op, {x: 1})
    assert op_resolved.circuit_function == cf
    assert op_resolved.function_args == (0.5,)
    assert cirq.parameter_names(op) == {'x'}
    assert cirq.parameter_names(op_resolved) == set()
    assert op_resolved.param_resolver == cirq.ParamResolver({x: 1})
    assert cirq.is_parameterized(op)
    assert not cirq.is_parameterized(op_resolved)
    assert op.qubits == (q,)
    assert op_resolved.qubits == (cirq.q(1),)


def test_decompose() -> None:
    x = sympy.Symbol('x')
    q = cirq.VariableLineQid(x)
    theta = sympy.Symbol('theta')
    c = cirq.Circuit(cirq.X(q) ** theta)
    cf = cirq.CircuitFunction("test_function", c, function_params=[theta])
    op = cirq.CallFunctionOperation(cf, [0.5])
    test_circ = cirq.decompose(op)
    ref_circ = [cirq.X(q) ** 0.5]
    assert test_circ == ref_circ

    resolved_op = cirq.resolve_parameters(op, {x: 4})
    test_circ = cirq.decompose(resolved_op)
    assert test_circ == [cirq.X(cirq.q(4)) ** 0.5]

    cf = cirq.CircuitFunction("test_function", c, function_params=[theta, x])
    op = cirq.CallFunctionOperation(cf, [0.5, 3])
    test_circ = cirq.decompose(op)
    ref_circ = [cirq.X(cirq.q(3)) ** 0.5]
    assert test_circ == ref_circ

    op = cirq.CallFunctionOperation(cf, [2, 5])
    test_circ = cirq.decompose(op)
    ref_circ = [cirq.X(cirq.q(5)) ** 2]
    assert test_circ == ref_circ

    y = sympy.Symbol('y')
    op = cirq.CallFunctionOperation(cf, [2, y])
    test_circ = cirq.decompose(op)
    ref_circ = [cirq.X(cirq.VariableLineQid(y)) ** 2]
    assert test_circ == ref_circ


def test_with_qubits() -> None:
    q = cirq.q(0)
    theta = sympy.Symbol('theta')
    c = cirq.Circuit(cirq.X(q) ** theta)
    cf = cirq.CircuitFunction("test_function", c, function_params=[theta])
    op = cirq.CallFunctionOperation(cf, [0.5])

    assert op.with_qubits(cirq.q(0)) == op

    with pytest.raises(ValueError, match="CallFunctionOperation does not support `with_qubits`"):
        _ = op.with_qubits(cirq.q(1))


def test_repr() -> None:
    x = sympy.Symbol('x')
    q = cirq.VariableLineQid(x)
    theta = sympy.Symbol('theta')
    c = cirq.Circuit(cirq.X(q) ** theta)
    cf = cirq.CircuitFunction("test_function", c, function_params=[theta])
    op = cirq.CallFunctionOperation(cf, [0.5])
    cirq.testing.assert_equivalent_repr(op)

    cf = cirq.CircuitFunction("test_function", c, function_params=[theta, x])
    op = cirq.CallFunctionOperation(cf, [0.5, 3])
    cirq.testing.assert_equivalent_repr(op)

    cf = cirq.CircuitFunction("test_function", c, function_params=[theta])
    op = cirq.CallFunctionOperation(cf, [0.5], param_resolver={'x': 1})
    cirq.testing.assert_equivalent_repr(op)


def test_json_serializes_shared_circuit_function_once() -> None:
    x = sympy.Symbol('x')
    c = cirq.Circuit(cirq.X(cirq.VariableLineQid(x)))
    cf = cirq.CircuitFunction("test_function", c, function_params=[x])
    circuit = cirq.Circuit(cirq.CallFunctionOperation(cf, [i]) for i in range(5))

    json_text = cirq.to_json(circuit)
    assert json_text.count('"cirq_type": "CircuitFunction"') == 1


def test_str() -> None:
    x, y = sympy.symbols('x y')
    qx = cirq.VariableLineQid(x)
    qy = cirq.VariableLineQid(y)
    theta = sympy.Symbol('theta')
    c = cirq.Circuit(cirq.X(qx) ** theta, cirq.X(qy))
    cf = cirq.CircuitFunction("test_function", c, function_params=[theta])
    op = cirq.CallFunctionOperation(cf, [0.5])
    assert str(op) == "test_function(0.5)"

    op = cirq.CallFunctionOperation(cf, [0.5], param_resolver={'x': 2})
    assert str(op) == "test_function(0.5) [x=2]"

    op = cirq.CallFunctionOperation(cf, [0.5], param_resolver={'x': 2, y: 3})
    assert str(op) == "test_function(0.5) [x=2, y=3]"

    cf = cirq.CircuitFunction("test_function", c, function_params=[theta, x])
    op = cirq.CallFunctionOperation(cf, [0.5, 3])
    assert str(op) == "test_function(0.5, 3)"


def test_circuit_diagram() -> None:
    x = sympy.Symbol('x')
    q = cirq.VariableLineQid(x)
    theta = sympy.Symbol('theta')
    c = cirq.Circuit(cirq.X(q) ** theta)
    cf = cirq.CircuitFunction("test_function", c, function_params=[theta])
    op = cirq.CallFunctionOperation(cf, [0.5])

    circuit = cirq.Circuit(op)
    diagram = circuit.to_text_diagram()
    assert "test_function(0.5)" in diagram

    c = cirq.Circuit(cirq.GlobalPhaseGate(1).on())
    cf = cirq.CircuitFunction("test_function", c)
    op = cirq.CallFunctionOperation(cf, [])
    circuit = cirq.Circuit(op)
    diagram = circuit.to_text_diagram()
    assert diagram.strip() == "test_function"


def test_unitary() -> None:
    x = sympy.Symbol('x')
    q = cirq.VariableLineQid(x)
    theta = sympy.Symbol('theta')
    c = cirq.Circuit(cirq.X(q) ** theta)

    cf = cirq.CircuitFunction("test_function", c, function_params=[theta])
    op = cirq.CallFunctionOperation(cf, [0.5])
    assert not cirq.has_unitary(op)

    cf = cirq.CircuitFunction("test_function", c, function_params=[theta, x])
    op = cirq.CallFunctionOperation(cf, [0.5, 1])
    assert cirq.has_unitary(op)

    c = cirq.Circuit(cirq.measure(cirq.q(0), key='m0'))
    cf = cirq.CircuitFunction("test_function", c)
    op = cirq.CallFunctionOperation(cf, [])
    assert not cirq.has_unitary(op)


def test_measurement() -> None:
    x = sympy.Symbol('x')
    q = cirq.VariableLineQid(x)
    theta = sympy.Symbol('theta')
    c = cirq.Circuit(
        cirq.X(q) ** theta,
        cirq.measure(cirq.q(1), key='m0'),
        cirq.measure(q, key='m1'),
        cirq.Y(q).with_classical_controls('m2'),
    )
    cf = cirq.CircuitFunction("test_function", c, function_params=[theta])
    op = cirq.CallFunctionOperation(cf, [0.5])

    assert cirq.is_measurement(op)
    assert cirq.measurement_key_objs(op) == {cirq.MeasurementKey('m0'), cirq.MeasurementKey('m1')}
    assert cirq.measurement_key_names(op) == {'m0', 'm1'}
    assert cirq.control_keys(op) == {'m2'}


def test_protocols() -> None:
    theta0, theta1, x = sympy.symbols('theta0 theta1 x')
    c = cirq.Circuit(cirq.X(cirq.q(0)) ** theta0, cirq.Y(cirq.q(1)) ** theta1)
    cf = cirq.CircuitFunction("test_function", c, function_params=[theta0])
    op = cirq.CallFunctionOperation(cf, [0.5])
    cirq.testing.assert_implements_consistent_protocols(op)

    cf = cirq.CircuitFunction("test_function", c, function_params=[theta0, theta1])
    op = cirq.CallFunctionOperation(cf, [0.5, 1])
    cirq.testing.assert_implements_consistent_protocols(op)

    op = cirq.CallFunctionOperation(cf, [0.5, x])
    cirq.testing.assert_implements_consistent_protocols(op)

    cf = cirq.CircuitFunction("test_function", c, function_params=[])
    op = cirq.CallFunctionOperation(cf, [], param_resolver={theta0: 0.5, 'theta1': 1})
    cirq.testing.assert_implements_consistent_protocols(op)
