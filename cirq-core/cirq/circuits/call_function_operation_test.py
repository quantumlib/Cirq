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

    op_overlapping_resolver = cirq.CallFunctionOperation(cf, [0.5], {theta: 0.25})
    assert cirq.parameter_names(op_overlapping_resolver) == {'x'}
    assert op_overlapping_resolver.param_resolver == cirq.ParamResolver()


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

    resolved_op = cirq.resolve_parameters(op, {theta: 0.25})
    test_circ = cirq.decompose(resolved_op)
    assert test_circ == [cirq.X(q) ** 0.5]

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


def test_same_qubits() -> None:
    x = sympy.Symbol('x')
    c = cirq.Circuit(cirq.Moment(cirq.X(cirq.q(0))), cirq.Moment(cirq.Z(cirq.VariableLineQid(x))))
    cf = cirq.CircuitFunction('f', c, function_params=[x])
    op = cirq.CallFunctionOperation(cf, [0])
    assert op.qubits == (cirq.q(0),)

    c = cirq.Circuit(cirq.Moment(cirq.X(cirq.q(0)), cirq.Z(cirq.VariableLineQid(x))))
    cf = cirq.CircuitFunction('f', c, function_params=[x])
    op = cirq.CallFunctionOperation(cf, [0])
    assert op.qubits == (cirq.q(0),)
    with pytest.raises(ValueError, match='Overlapping operations'):
        cirq.decompose(op)

    c = cirq.Circuit(cirq.CNOT(cirq.q(0), cirq.VariableLineQid(x)))
    cf = cirq.CircuitFunction('f', c, function_params=[x])
    op = cirq.CallFunctionOperation(cf, [0])
    assert op.qubits == (cirq.q(0),)
    with pytest.raises(ValueError, match='Duplicate qids for <cirq.CNOT'):
        cirq.decompose(op)


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

    cf = cirq.CircuitFunction("test_function", c, function_params=[theta])
    op = cirq.CallFunctionOperation(cf, [0.5])
    swapped_op = cirq.resolve_parameters_once(op, {x: y, y: x})
    assert str(swapped_op) == "test_function(0.5) [x=y, y=x]"

    chained_op = cirq.resolve_parameters_once(op, {x: y, y: 2})
    assert str(chained_op) == "test_function(0.5) [x=y, y=2]"

    c_measured = cirq.Circuit(cirq.X(qx) ** theta, cirq.X(qy), cirq.measure(qx, key='m0'))
    cf_measured = cirq.CircuitFunction("test_function", c_measured, function_params=[theta])
    all_params_op = cirq.CallFunctionOperation(
        circuit_function=cf_measured,
        function_args=[0.5],
        param_resolver={'x': 2},
        measurement_key_map={'m0': 'a'},
        parent_path=('p',),
    )
    assert str(all_params_op) == "test_function(0.5) [x=2] key_map={m0: a} parent_path=('p',)"


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

    op = cirq.CallFunctionOperation(
        cf, [0.5], measurement_key_map={'m0': 'a', 'm2': 'c'}, parent_path=('p',)
    )
    assert cirq.measurement_key_objs(op) == {
        cirq.MeasurementKey('a', path=('p',)),
        cirq.MeasurementKey('m1', path=('p',)),
    }
    assert cirq.measurement_key_names(op) == {'p:a', 'p:m1'}
    assert cirq.control_keys(op) == {cirq.MeasurementKey('c')}

    with pytest.raises(ValueError, match='Mapping to invalid key'):
        _ = cirq.CallFunctionOperation(cf, [0.5], measurement_key_map={'m0': 'a:b'})


def test_with_measurement_key_mapping() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.measure(q0, key='a'), cirq.measure(q1, key='b'))
    cf = cirq.CircuitFunction("test_function", c)
    op = cirq.CallFunctionOperation(cf, [])

    mapped_op = cirq.with_measurement_key_mapping(op, {'a': 'x'})
    assert mapped_op.measurement_key_map == {'a': 'x'}
    assert cirq.measurement_key_names(mapped_op) == {'x', 'b'}
    assert cirq.decompose(mapped_op) == [cirq.measure(q0, key='x'), cirq.measure(q1, key='b')]

    remapped_op = cirq.with_measurement_key_mapping(mapped_op, {'x': 'y', 'zz': 'w'})
    assert remapped_op.measurement_key_map == {'a': 'y'}
    assert cirq.measurement_key_names(remapped_op) == {'y', 'b'}

    with pytest.raises(ValueError, match='Collision in measurement key map composition'):
        _ = cirq.with_measurement_key_mapping(op, {'a': 'b'})


def test_with_key_path() -> None:
    q0 = cirq.q(0)
    c = cirq.Circuit(cirq.measure(q0, key='m'))
    cf = cirq.CircuitFunction("test_function", c)
    op = cirq.CallFunctionOperation(cf, [])

    prefixed_op = cirq.with_key_path_prefix(op, ('p',))
    assert prefixed_op.parent_path == ('p',)
    assert cirq.measurement_key_names(prefixed_op) == {'p:m'}
    assert cirq.decompose(prefixed_op) == [cirq.measure(q0, key=cirq.MeasurementKey('m', ('p',)))]

    assert cirq.with_key_path_prefix(prefixed_op, ('outer',)).parent_path == ('outer', 'p')
    assert cirq.with_key_path(prefixed_op, ('x',)).parent_path == ('x',)


def test_circuit_operation_matches_simulator() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit(
        cirq.X(q0),
        cirq.measure(q0, key='m'),
        cirq.X(q1).with_classical_controls('m'),
        cirq.measure(q1, key='n'),
    )
    cf = cirq.CircuitFunction("test_function", c)
    op = cirq.CallFunctionOperation(cf, [])
    repeated_circuit = cirq.Circuit(
        cirq.CircuitOperation(cirq.FrozenCircuit(op), repetitions=2, use_repetition_ids=True)
    )
    assert cirq.measurement_key_names(repeated_circuit) == {'0:m', '0:n', '1:m', '1:n'}
    result = cirq.Simulator().run(repeated_circuit, repetitions=3)
    assert set(result.measurements) == cirq.measurement_key_names(repeated_circuit)


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

    c = cirq.Circuit(cirq.X(cirq.q(0)) ** theta0, cirq.measure(cirq.q(0), key='m'))
    cf = cirq.CircuitFunction("test_function", c, function_params=[theta0])
    op = cirq.CallFunctionOperation(
        cf, [0.5], measurement_key_map={'m': 'n'}, parent_path=('p', 'q')
    )
    cirq.testing.assert_implements_consistent_protocols(op)
