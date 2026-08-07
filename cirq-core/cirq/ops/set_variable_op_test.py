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


def test_set_variable_init_and_properties():
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    op = cirq.SetVariable(a, b + 1)
    assert op.target == a
    assert op.expression == b + 1
    assert op.qubits == ()
    assert op.with_qubits() == op

    with pytest.raises(ValueError, match="does not support qubits"):
        op.with_qubits(cirq.LineQubit(0))

    with pytest.raises(TypeError, match="target must be a sympy.Symbol"):
        cirq.SetVariable("not_a_symbol", 1)


def test_set_variable_repr_and_str():
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    op = cirq.SetVariable(a, b + 1)
    assert str(op) == "SetVariable(a, b + 1)"
    assert (
        repr(op)
        == "cirq.SetVariable(sympy.Symbol('a'), sympy.Add(sympy.Symbol('b'), sympy.Integer(1)))"
    )


def test_set_variable_parameter_names():
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    op = cirq.SetVariable(a, b + 1)
    assert cirq.parameter_names(op) == {'b'}
    assert cirq.is_parameterized(op)
    assert not cirq.is_parameterized(cirq.SetVariable(a, 1.0))


def test_set_variable_simulation_constant():
    q = cirq.LineQubit(0)
    a = sympy.Symbol('a')
    circuit = cirq.Circuit(cirq.SetVariable(a, 1.0), cirq.Rx(rads=a).on(q))
    sim = cirq.Simulator()
    result = sim.simulate(circuit)
    expected_state = cirq.Circuit(cirq.Rx(rads=1.0).on(q)).final_state_vector()
    np.testing.assert_allclose(result.final_state_vector, expected_state, atol=1e-6)


def test_set_variable_simulation_expression():
    q = cirq.LineQubit(0)
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    circuit = cirq.Circuit(cirq.SetVariable(b, a * 2), cirq.Rx(rads=b).on(q))
    sim = cirq.Simulator()
    result = sim.simulate(circuit, param_resolver={'a': 0.5})
    expected_state = cirq.Circuit(cirq.Rx(rads=1.0).on(q)).final_state_vector()
    np.testing.assert_allclose(result.final_state_vector, expected_state, atol=1e-6)


def test_set_variable_run_simulation():
    q0 = cirq.LineQubit(0)
    a = sympy.Symbol('a')
    circuit = cirq.Circuit(cirq.SetVariable(a, 1), cirq.measure(q0, key='m'))
    sim = cirq.Simulator()
    result = sim.run(circuit, repetitions=3)
    assert len(result.measurements['m']) == 3


def test_set_variable_simulation_measurement_dependency():
    q0, q1 = cirq.LineQubit.range(2)
    a = sympy.Symbol('a')

    # We must explicitly separate X, measure, SetVariable, and Rx into separate moments
    # to guarantee correct execution order (since SetVariable has no qubits and would
    # otherwise be aligned alongside measurement).
    circuit = cirq.Circuit(
        cirq.Moment(cirq.X(q0)),
        cirq.Moment(cirq.measure(q0, key='m')),
        cirq.Moment(cirq.SetVariable(a, sympy.Symbol('m') * np.pi)),
        cirq.Moment(cirq.Rx(rads=a).on(q1)),
    )
    sim = cirq.Simulator()
    result = sim.simulate(circuit)
    # q0 is 1, so m=1, a=pi, Rx(pi) on q1 rotates |0> to -i|1>.
    # So q1 final state is |1> (up to phase).
    expected_state = cirq.Circuit(cirq.X(q0), cirq.Rx(rads=np.pi).on(q1)).final_state_vector()
    np.testing.assert_allclose(np.abs(result.final_state_vector), np.abs(expected_state), atol=1e-6)


def test_set_variable_act_on_edge_cases():
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    c = sympy.Symbol('c')
    q0 = cirq.LineQubit(0)

    # 1. Parameter resolver lookup in _act_on_ (by name or symbol)
    op = cirq.SetVariable(a, b * 2)
    state = cirq.StateVectorSimulationState(
        qubits=[q0], param_resolver=cirq.ParamResolver({'b': 3})
    )
    assert op._act_on_(state)
    assert state.param_resolver.value_of('a') == 6

    state_symbol = cirq.StateVectorSimulationState(
        qubits=[q0], param_resolver=cirq.ParamResolver({b: 4})
    )
    assert op._act_on_(state_symbol)
    assert state_symbol.param_resolver.value_of(a) == 8

    # 2. Target symbol already in current_params by string name vs symbol
    op_update = cirq.SetVariable(a, 10)
    assert op_update._act_on_(state)
    assert state.param_resolver.value_of('a') == 10

    assert op_update._act_on_(state_symbol)
    assert state_symbol.param_resolver.value_of(a) == 10

    # 3. Unresolved symbol error
    op_unresolved = cirq.SetVariable(a, c + 1)
    empty_state = cirq.StateVectorSimulationState(qubits=[q0])
    with pytest.raises(ValueError, match="could not be resolved"):
        op_unresolved._act_on_(empty_state)

    # 4. Expression did not evaluate to a number
    op_non_numeric = cirq.SetVariable(a, sympy.Symbol('unresolved'))
    with pytest.raises(ValueError, match="did not evaluate to a number"):
        op_non_numeric._act_on_(
            cirq.StateVectorSimulationState(
                qubits=[q0],
                param_resolver=cirq.ParamResolver({'unresolved': sympy.Symbol('still_symbol')}),
            )
        )

    # 5. Indexed symbol / bitwise measurement in expression
    q1 = cirq.LineQubit(1)
    m_base = sympy.IndexedBase('m')
    op_indexed = cirq.SetVariable(a, m_base[0] + m_base[1])
    indexed_state = cirq.StateVectorSimulationState(qubits=[q0, q1])
    indexed_state.classical_data.record_measurement(cirq.MeasurementKey('m'), [1, 0], [q0, q1])
    assert op_indexed._act_on_(indexed_state)
    assert indexed_state.param_resolver.value_of('a') == 1


def test_set_variable_json():
    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    op = cirq.SetVariable(a, b + 1)
    json_text = cirq.to_json(op)
    restored_op = cirq.read_json(json_text=json_text)
    assert restored_op == op
