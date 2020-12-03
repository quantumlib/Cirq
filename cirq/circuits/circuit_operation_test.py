# Copyright 2018 The Cirq Developers
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

import pytest, sympy

import cirq


def test_properties():
    a, b, c = cirq.LineQubit.range(3)
    circuit = cirq.FrozenCircuit(
        cirq.X(a),
        cirq.Y(b),
        cirq.H(c),
        cirq.CX(a, b) ** sympy.Symbol('exp'),
        cirq.measure(a, b, c, key='m'),
    )
    op = cirq.CircuitOperation(circuit)
    assert op.circuit is circuit
    assert op.qubits == (a, b, c)
    assert op.qubit_map == {a: a, b: b, c: c}
    assert op.measurement_key_map == {'m': 'm'}
    assert op.param_resolver == cirq.ParamResolver({sympy.Symbol('exp'): sympy.Symbol('exp')})
    assert op.repetitions == 1


def test_with_qubits():
    a, b, c, d = cirq.LineQubit.range(4)
    circuit = cirq.FrozenCircuit(cirq.H(a), cirq.CX(a, b))
    op_base = cirq.CircuitOperation(circuit)

    op_with_qubits = op_base.with_qubits(d, c)
    assert op_with_qubits.base_operation() == op_base
    assert op_with_qubits.qubits == (d, c)
    assert op_with_qubits.qubit_map == {a: d, b: c}

    assert op_base.with_qubit_mapping({a: d, b: c, d: a}) == op_with_qubits
    with pytest.raises(ValueError):
        _ = op_base.with_qubits(c, d, b)


def test_with_measurement_keys():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.FrozenCircuit(
        cirq.X(a),
        cirq.measure(b, key='mb'),
        cirq.measure(a, key='ma'),
    )
    op_base = cirq.CircuitOperation(circuit)

    op_with_keys = op_base.with_measurement_key_mapping({'ma': 'pa', 'x': 'z'})
    assert op_with_keys.base_operation() == op_base
    assert op_with_keys.measurement_key_map == {'ma': 'pa', 'mb': 'mb'}
    assert cirq.measurement_keys(op_with_keys) == {'pa', 'mb'}

    assert cirq.with_measurement_key_mapping(op_base, {'ma': 'pa'}) == op_with_keys


def test_with_params():
    a = cirq.LineQubit(0)
    z_exp = sympy.Symbol('z_exp')
    x_exp = sympy.Symbol('x_exp')
    delta = sympy.Symbol('delta')
    theta = sympy.Symbol('theta')
    circuit = cirq.FrozenCircuit(cirq.Z(a) ** z_exp, cirq.X(a) ** x_exp, cirq.Z(a) ** delta)
    op_base = cirq.CircuitOperation(circuit)

    param_dict = {
        z_exp: 2,
        x_exp: theta,
        sympy.Symbol('k'): sympy.Symbol('phi'),
    }
    op_with_params = op_base.with_params(param_dict)
    assert op_with_params.base_operation() == op_base
    assert op_with_params.param_resolver == cirq.ParamResolver(
        {
            z_exp: 2,
            x_exp: theta,
            delta: delta,
            sympy.Symbol('k'): sympy.Symbol('phi'),
        }
    )
    assert cirq.parameter_names(op_with_params) == {'theta', 'delta'}

    assert cirq.resolve_parameters(op_base, cirq.ParamResolver(param_dict)) == op_with_params


def test_repetition():
    a, b = cirq.LineQubit.range(2)
    # The correct modulus for this circuit is 8.
    circuit = cirq.FrozenCircuit(cirq.H(a), cirq.CX(a, b))
    op_base = cirq.CircuitOperation(circuit)

    op_with_reps = op_base.repeat(-5)
    assert op_with_reps.repetitions == -5
    assert op_base ** -5 == op_with_reps

    op_with_reps = op_base.repeat(-5, modulus=4)
    assert op_with_reps.repetitions == 3

    op_with_reps = op_base.repeat(-5, modulus=4, allow_invert=True)
    assert op_with_reps.repetitions == -1

    with pytest.raises(ValueError):
        _ = op_base.repeat(-5, modulus=4, validate_modulus=True)

    op_with_reps = op_base.repeat(-5, modulus=8, validate_modulus=True)
    assert op_with_reps.repetitions == 3


def test_with_circuit():
    a, b, c, d = cirq.LineQubit.range(4)
    exp = sympy.Symbol('exp')
    theta = sympy.Symbol('theta')
    base_circuit = cirq.FrozenCircuit(
        cirq.X(a),
        cirq.Y(b),
        cirq.H(c),
        cirq.CX(a, b) ** exp,
        cirq.measure(a, b, c, key='m'),
    )
    op_base = (
        cirq.CircuitOperation(base_circuit)
        .with_qubits(d, a, c)
        .with_measurement_key_mapping({'m': 'p'})
        .with_params({exp: theta})
    )

    new_circuit = cirq.FrozenCircuit(
        cirq.H(a),
        cirq.H(b),
        cirq.H(c),
        cirq.CCZ(a, b, c) ** exp,
        cirq.measure(c, key='m'),
    )
    new_op = op_base.with_circuit(new_circuit)
    assert new_op.qubit_map == op_base.qubit_map
    assert new_op.measurement_key_map == op_base.measurement_key_map
    assert new_op.param_resolver == op_base.param_resolver
    assert new_op.repetitions == op_base.repetitions


def test_repeat_measurement_fails():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.FrozenCircuit(
        cirq.X(a),
        cirq.measure(b, key='mb'),
        cirq.measure(a, key='ma'),
    )
    op = cirq.CircuitOperation(circuit)
    with pytest.raises(NotImplementedError):
        _ = op.repeat(3)

    rep_circuit = cirq.FrozenCircuit(cirq.H(a), cirq.CX(a, b))
    rep_op = cirq.CircuitOperation(rep_circuit).repeat(3)
    with pytest.raises(NotImplementedError):
        _ = rep_op.with_circuit(circuit)


def test_qid_shape():
    circuit = cirq.FrozenCircuit(
        cirq.IdentityGate(qid_shape=(q.dimension,)).on(q)
        for q in cirq.LineQid.for_qid_shape((1, 2, 3, 4))
    )
    op = cirq.CircuitOperation(circuit)
    assert cirq.qid_shape(op) == (1, 2, 3, 4)
    assert cirq.num_qubits(op) == 4

    id_circuit = cirq.FrozenCircuit(cirq.I(q) for q in cirq.LineQubit.range(3))
    id_op = cirq.CircuitOperation(id_circuit)
    assert cirq.qid_shape(id_op) == (2, 2, 2)
    assert cirq.num_qubits(id_op) == 3


def test_string_format():
    x, y, z = cirq.LineQubit.range(3)

    fc0 = cirq.FrozenCircuit()
    op0 = cirq.CircuitOperation(fc0)
    assert (
        str(op0)
        == f"""\
Circuit_{hash(op0.circuit) % int(1e6):06d}:
[             ]"""
    )

    fc1 = cirq.FrozenCircuit(cirq.X(x), cirq.H(y), cirq.CX(y, z), cirq.measure(x, y, z, key='m'))
    op1 = cirq.CircuitOperation(fc1).with_qubits(x, y, z)
    assert (
        str(op1)
        == f"""\
Circuit_{hash(op1.circuit) % int(1e6):06d}:
[ 0: ───X───────M('m')─── ]
[               │         ]
[ 1: ───H───@───M──────── ]
[           │   │         ]
[ 2: ───────X───M──────── ]"""
    )
    assert (
        repr(op1)
        == f"""\
cirq.CircuitOperation(cirq.FrozenCircuit([
    cirq.Moment(
        cirq.X(cirq.LineQubit(0)),
        cirq.H(cirq.LineQubit(1)),
    ),
    cirq.Moment(
        cirq.CNOT(cirq.LineQubit(1), cirq.LineQubit(2)),
    ),
    cirq.Moment(
        cirq.measure(cirq.LineQubit(0), cirq.LineQubit(1), cirq.LineQubit(2), key='m'),
    ),
]))"""
    )

    fc2 = cirq.FrozenCircuit(cirq.X(x), cirq.H(y), cirq.CX(y, x), name='new_gate')
    op2 = cirq.CircuitOperation(fc2).with_qubits(x, z).repeat(3)
    assert (
        str(op2)
        == """\
new_gate:
[ 0: ───X───X─── ]
[           │    ]
[ 1: ───H───@─── ](qubit_map={0: 0, 1: 2}, loops=3)"""
    )
    assert (
        repr(op2)
        == """\
cirq.CircuitOperation(cirq.FrozenCircuit([
    cirq.Moment(
        cirq.X(cirq.LineQubit(0)),
        cirq.H(cirq.LineQubit(1)),
    ),
    cirq.Moment(
        cirq.CNOT(cirq.LineQubit(1), cirq.LineQubit(0)),
    ),
], name='new_gate')).with_qubit_mapping({
    cirq.LineQubit(0): cirq.LineQubit(0),
    cirq.LineQubit(1): cirq.LineQubit(2),
}).repeat(3)"""
    )

    fc3 = cirq.FrozenCircuit(
        cirq.X(x) ** sympy.Symbol('b'),
        cirq.measure(x, key='m'),
    )
    op3 = (
        cirq.CircuitOperation(fc3)
        .with_qubits(y)
        .with_measurement_key_mapping({'m': 'p'})
        .with_params({sympy.Symbol('b'): 2})
    )
    assert (
        str(op3)
        == f"""\
Circuit_{hash(op3.circuit) % int(1e6):06d}:
[ 0: ───X^b───M('m')─── ](qubit_map={{0: 1}}, \
key_map={{m: p}}, params={{b: 2}})"""
    )
    assert (
        repr(op3)
        == f"""\
cirq.CircuitOperation({fc3!r}).with_qubit_mapping({{
    cirq.LineQubit(0): cirq.LineQubit(1),
}}).with_measurement_key_mapping({{
    'm': 'p',
}}).with_params(cirq.ParamResolver({{sympy.Symbol('b'): 2}}))"""
    )


def test_json_dict():
    a, b, c = cirq.LineQubit.range(3)
    circuit = cirq.FrozenCircuit(
        cirq.X(a),
        cirq.Y(b),
        cirq.H(c),
        cirq.CX(a, b) ** sympy.Symbol('exp'),
        cirq.measure(a, b, c, key='m'),
    )
    op = (
        cirq.CircuitOperation(circuit)
        .with_qubit_mapping({c: b, b: c})
        .with_measurement_key_mapping({'m': 'p'})
        .with_params({'exp': 'theta'})
    )

    assert op._json_dict_() == {
        'cirq_type': 'CircuitOperation',
        'circuit': circuit,
        'repetitions': 1,
        'qubit_map': op.qubit_map,
        'measurement_key_map': op.measurement_key_map,
        'param_resolver': op.param_resolver,
    }


def test_terminal_matches():
    a, b = cirq.LineQubit.range(2)
    fc = cirq.FrozenCircuit(
        cirq.H(a),
        cirq.measure(b, key='m1'),
    )
    op = cirq.CircuitOperation(fc)

    c = cirq.Circuit(cirq.X(a), op)
    assert c.are_all_measurements_terminal()

    c = cirq.Circuit(cirq.X(b), op)
    assert c.are_all_measurements_terminal()

    c = cirq.Circuit(cirq.measure(a), op)
    assert not c.are_all_measurements_terminal()

    c = cirq.Circuit(cirq.measure(b), op)
    assert not c.are_all_measurements_terminal()

    c = cirq.Circuit(op, cirq.X(a))
    assert c.are_all_measurements_terminal()

    c = cirq.Circuit(op, cirq.X(b))
    assert not c.are_all_measurements_terminal()

    c = cirq.Circuit(op, cirq.measure(a))
    assert c.are_all_measurements_terminal()

    c = cirq.Circuit(op, cirq.measure(b))
    assert not c.are_all_measurements_terminal()


def test_nonterminal_in_subcircuit():
    a, b = cirq.LineQubit.range(2)
    fc = cirq.FrozenCircuit(
        cirq.H(a),
        cirq.measure(b, key='m1'),
        cirq.X(b),
    )
    op = cirq.CircuitOperation(fc)
    c = cirq.Circuit(cirq.X(a), op)
    assert isinstance(op, cirq.CircuitOperation)
    assert not c.are_all_measurements_terminal()

    op = op.with_tags('test')
    c = cirq.Circuit(cirq.X(a), op)
    assert not isinstance(op, cirq.CircuitOperation)
    assert not c.are_all_measurements_terminal()


def test_decompose_applies_maps():
    a, b, c = cirq.LineQubit.range(3)
    exp = sympy.Symbol('exp')
    theta = sympy.Symbol('theta')
    circuit = cirq.FrozenCircuit(
        cirq.X(a),
        cirq.Y(b),
        cirq.H(c),
        cirq.CX(a, b) ** exp,
        cirq.measure(a, b, c, key='m'),
    )
    op = cirq.CircuitOperation(circuit)
    op = op.with_qubit_mapping(
        {
            c: b,
            b: c,
        }
    )
    op = op.with_measurement_key_mapping({'m': 'p'})
    op = op.with_params({exp: theta})

    expected_circuit = cirq.Circuit(
        cirq.X(a),
        cirq.Y(c),
        cirq.H(b),
        cirq.CX(a, c) ** theta,
        cirq.measure(a, c, b, key='p'),
    )
    assert cirq.Circuit(cirq.decompose_once(op)) == expected_circuit


def test_decompose_loops():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.FrozenCircuit(
        cirq.H(a),
        cirq.CX(a, b),
    )
    base_op = cirq.CircuitOperation(circuit)

    op = base_op.with_qubits(b, a).repeat(3)
    expected_circuit = cirq.Circuit(
        cirq.H(b),
        cirq.CX(b, a),
        cirq.H(b),
        cirq.CX(b, a),
        cirq.H(b),
        cirq.CX(b, a),
    )
    assert cirq.Circuit(cirq.decompose_once(op)) == expected_circuit

    op = base_op.repeat(-2)
    expected_circuit = cirq.Circuit(
        cirq.CX(a, b),
        cirq.H(a),
        cirq.CX(a, b),
        cirq.H(a),
    )
    assert cirq.Circuit(cirq.decompose_once(op)) == expected_circuit


def test_decompose_nested():
    a, b, c, d = cirq.LineQubit.range(4)
    exp1 = sympy.Symbol('exp1')
    exp_half = sympy.Symbol('exp_half')
    exp_one = sympy.Symbol('exp_one')
    exp_two = sympy.Symbol('exp_two')
    circuit1 = cirq.FrozenCircuit(cirq.X(a) ** exp1, cirq.measure(a, key='m1'))
    op1 = cirq.CircuitOperation(circuit1)
    circuit2 = cirq.FrozenCircuit(
        op1.with_qubits(a).with_measurement_key_mapping({'m1': 'ma'}),
        op1.with_qubits(b).with_measurement_key_mapping({'m1': 'mb'}),
        op1.with_qubits(c).with_measurement_key_mapping({'m1': 'mc'}),
        op1.with_qubits(d).with_measurement_key_mapping({'m1': 'md'}),
    )
    op2 = cirq.CircuitOperation(circuit2)
    circuit3 = cirq.FrozenCircuit(
        op2.with_params({exp1: exp_half}),
        op2.with_params({exp1: exp_one}),
        op2.with_params({exp1: exp_two}),
    )
    op3 = cirq.CircuitOperation(circuit3)

    final_op = op3.with_params({exp_half: 0.5, exp_one: 1.0, exp_two: 2.0})

    expected_circuit1 = cirq.Circuit(
        op2.with_params({exp1: 0.5, exp_half: 0.5, exp_one: 1.0, exp_two: 2.0}),
        op2.with_params({exp1: 1.0, exp_half: 0.5, exp_one: 1.0, exp_two: 2.0}),
        op2.with_params({exp1: 2.0, exp_half: 0.5, exp_one: 1.0, exp_two: 2.0}),
    )

    result_ops1 = cirq.decompose_once(final_op)
    assert cirq.Circuit(result_ops1) == expected_circuit1

    expected_circuit = cirq.Circuit(
        cirq.X(a) ** 0.5,
        cirq.measure(a, key='ma'),
        cirq.X(b) ** 0.5,
        cirq.measure(b, key='mb'),
        cirq.X(c) ** 0.5,
        cirq.measure(c, key='mc'),
        cirq.X(d) ** 0.5,
        cirq.measure(d, key='md'),
        cirq.X(a) ** 1.0,
        cirq.measure(a, key='ma'),
        cirq.X(b) ** 1.0,
        cirq.measure(b, key='mb'),
        cirq.X(c) ** 1.0,
        cirq.measure(c, key='mc'),
        cirq.X(d) ** 1.0,
        cirq.measure(d, key='md'),
        cirq.X(a) ** 2.0,
        cirq.measure(a, key='ma'),
        cirq.X(b) ** 2.0,
        cirq.measure(b, key='mb'),
        cirq.X(c) ** 2.0,
        cirq.measure(c, key='mc'),
        cirq.X(d) ** 2.0,
        cirq.measure(d, key='md'),
    )
    assert cirq.Circuit(cirq.decompose(final_op)) == expected_circuit


def test_tag_propagation():
    # Tags are not propagated from the CircuitOperation to its components.
    # TODO: support tag propagation for better serialization.
    a, b, c = cirq.LineQubit.range(3)
    circuit = cirq.FrozenCircuit(
        cirq.X(a),
        cirq.H(b),
        cirq.H(c),
        cirq.CZ(a, c),
    )
    op = cirq.CircuitOperation(circuit)
    test_tag = 'test_tag'
    op = op.with_tags(test_tag)

    assert test_tag in op.tags

    # TODO: Tags must propagate during decomposition.
    sub_ops = cirq.decompose(op)
    for op in sub_ops:
        assert test_tag not in op.tags


# TODO: Operation has a "gate" property. What is this for a CircuitOperation?
