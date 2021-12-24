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
from cirq.circuits.circuit_operation import _full_join_string_lists


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
    assert op.qubit_map == {}
    assert op.measurement_key_map == {}
    assert op.param_resolver == cirq.ParamResolver()
    assert op.repetitions == 1
    assert op.repetition_ids is None
    # Despite having the same decomposition, these objects are not equal.
    assert op != circuit
    assert op == circuit.to_op()


def test_circuit_type():
    a, b, c = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.X(a),
        cirq.Y(b),
        cirq.H(c),
        cirq.CX(a, b) ** sympy.Symbol('exp'),
        cirq.measure(a, b, c, key='m'),
    )
    with pytest.raises(TypeError, match='Expected circuit of type FrozenCircuit'):
        _ = cirq.CircuitOperation(circuit)


def test_non_invertible_circuit():
    a, b, c = cirq.LineQubit.range(3)
    circuit = cirq.FrozenCircuit(
        cirq.X(a),
        cirq.Y(b),
        cirq.H(c),
        cirq.CX(a, b) ** sympy.Symbol('exp'),
        cirq.measure(a, b, c, key='m'),
    )
    with pytest.raises(ValueError, match='circuit is not invertible'):
        _ = cirq.CircuitOperation(circuit, repetitions=-2)


def test_repetitions_and_ids_length_mismatch():
    a, b, c = cirq.LineQubit.range(3)
    circuit = cirq.FrozenCircuit(
        cirq.X(a),
        cirq.Y(b),
        cirq.H(c),
        cirq.CX(a, b) ** sympy.Symbol('exp'),
        cirq.measure(a, b, c, key='m'),
    )
    with pytest.raises(ValueError, match='Expected repetition_ids to be a list of length 2'):
        _ = cirq.CircuitOperation(circuit, repetitions=2, repetition_ids=['a', 'b', 'c'])


def test_is_measurement_memoization():
    a = cirq.LineQubit(0)
    circuit = cirq.FrozenCircuit(cirq.measure(a, key='m'))
    c_op = cirq.CircuitOperation(circuit)
    assert circuit._has_measurements is None
    # Memoize `_has_measurements` in the circuit.
    assert cirq.is_measurement(c_op)
    assert circuit._has_measurements is True


def test_invalid_measurement_keys():
    a = cirq.LineQubit(0)
    circuit = cirq.FrozenCircuit(cirq.measure(a, key='m'))
    c_op = cirq.CircuitOperation(circuit)
    # Invalid key remapping
    with pytest.raises(ValueError, match='Mapping to invalid key: m:a'):
        _ = c_op.with_measurement_key_mapping({'m': 'm:a'})

    # Invalid key remapping nested CircuitOperation
    with pytest.raises(ValueError, match='Mapping to invalid key: m:a'):
        _ = cirq.CircuitOperation(cirq.FrozenCircuit(c_op), measurement_key_map={'m': 'm:a'})

    # Originally invalid key
    with pytest.raises(ValueError, match='Invalid key name: m:a'):
        _ = cirq.CircuitOperation(cirq.FrozenCircuit(cirq.measure(a, key='m:a')))

    # Remapped to valid key
    _ = cirq.CircuitOperation(circuit, measurement_key_map={'m:a': 'ma'})


def test_invalid_qubit_mapping():
    q = cirq.LineQubit(0)
    q3 = cirq.LineQid(1, dimension=3)

    # Invalid qid remapping dict in constructor
    with pytest.raises(ValueError, match='Qid dimension conflict'):
        _ = cirq.CircuitOperation(cirq.FrozenCircuit(), qubit_map={q: q3})

    # Invalid qid remapping dict in with_qubit_mapping call
    c_op = cirq.CircuitOperation(cirq.FrozenCircuit(cirq.X(q)))
    with pytest.raises(ValueError, match='Qid dimension conflict'):
        _ = c_op.with_qubit_mapping({q: q3})

    # Invalid qid remapping function in with_qubit_mapping call
    with pytest.raises(ValueError, match='Qid dimension conflict'):
        _ = c_op.with_qubit_mapping(lambda q: q3)


def test_circuit_sharing():
    a, b, c = cirq.LineQubit.range(3)
    circuit = cirq.FrozenCircuit(
        cirq.X(a),
        cirq.Y(b),
        cirq.H(c),
        cirq.CX(a, b) ** sympy.Symbol('exp'),
        cirq.measure(a, b, c, key='m'),
    )
    op1 = cirq.CircuitOperation(circuit)
    op2 = cirq.CircuitOperation(op1.circuit)
    op3 = circuit.to_op()
    assert op1.circuit is circuit
    assert op2.circuit is circuit
    assert op3.circuit is circuit

    assert hash(op1) == hash(op2)
    assert hash(op1) == hash(op3)


def test_with_qubits():
    a, b, c, d = cirq.LineQubit.range(4)
    circuit = cirq.FrozenCircuit(cirq.H(a), cirq.CX(a, b))
    op_base = cirq.CircuitOperation(circuit)

    op_with_qubits = op_base.with_qubits(d, c)
    assert op_with_qubits.base_operation() == op_base
    assert op_with_qubits.qubits == (d, c)
    assert op_with_qubits.qubit_map == {a: d, b: c}

    assert op_base.with_qubit_mapping({a: d, b: c, d: a}) == op_with_qubits

    def map_fn(qubit: 'cirq.Qid') -> 'cirq.Qid':
        if qubit == a:
            return d
        if qubit == b:
            return c
        return qubit

    fn_op = op_base.with_qubit_mapping(map_fn)
    assert fn_op == op_with_qubits
    # map_fn does not affect qubits c and d.
    assert fn_op.with_qubit_mapping(map_fn) == op_with_qubits

    # with_qubits must receive the same number of qubits as the circuit contains.
    with pytest.raises(ValueError, match='Expected 2 qubits, got 3'):
        _ = op_base.with_qubits(c, d, b)

    # Two qubits cannot be mapped onto the same target qubit.
    with pytest.raises(ValueError, match='Collision in qubit map'):
        _ = op_base.with_qubit_mapping({a: b})

    # Two qubits cannot be transformed into the same target qubit.
    with pytest.raises(ValueError, match='Collision in qubit map'):
        _ = op_base.with_qubit_mapping(lambda q: b)
    # with_qubit_mapping requires exactly one argument.
    with pytest.raises(TypeError, match='must be a function or dict'):
        _ = op_base.with_qubit_mapping('bad arg')


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
    assert op_with_keys.measurement_key_map == {'ma': 'pa'}
    assert cirq.measurement_key_names(op_with_keys) == {'pa', 'mb'}

    assert cirq.with_measurement_key_mapping(op_base, {'ma': 'pa'}) == op_with_keys

    # Two measurement keys cannot be mapped onto the same target string.
    with pytest.raises(ValueError):
        _ = op_base.with_measurement_key_mapping({'ma': 'mb'})


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
            # As 'k' is irrelevant to the circuit, it does not appear here.
        }
    )
    assert cirq.parameter_names(op_with_params) == {'theta', 'delta'}

    assert (
        cirq.resolve_parameters(op_base, cirq.ParamResolver(param_dict), recursive=False)
        == op_with_params
    )

    # Recursive parameter resolution is rejected.
    with pytest.raises(ValueError, match='Use "recursive=False"'):
        _ = cirq.resolve_parameters(op_base, cirq.ParamResolver(param_dict))


@pytest.mark.parametrize('add_measurements', [True, False])
@pytest.mark.parametrize('use_default_ids_for_initial_rep', [True, False])
def test_repeat(add_measurements, use_default_ids_for_initial_rep):
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(a), cirq.CX(a, b))
    if add_measurements:
        circuit.append([cirq.measure(b, key='mb'), cirq.measure(a, key='ma')])
    op_base = cirq.CircuitOperation(circuit.freeze())
    assert op_base.repeat(1) is op_base
    assert op_base.repeat(1, ['0']) != op_base
    assert op_base.repeat(1, ['0']) == op_base.repeat(repetition_ids=['0'])
    assert op_base.repeat(1, ['0']) == op_base.with_repetition_ids(['0'])

    initial_repetitions = -3
    if add_measurements:
        with pytest.raises(ValueError, match='circuit is not invertible'):
            _ = op_base.repeat(initial_repetitions)
        initial_repetitions = abs(initial_repetitions)

    op_with_reps = None  # type: cirq.CircuitOperation
    rep_ids = []
    if use_default_ids_for_initial_rep:
        op_with_reps = op_base.repeat(initial_repetitions)
        rep_ids = ['0', '1', '2']
        assert op_base ** initial_repetitions == op_with_reps
    else:
        rep_ids = ['a', 'b', 'c']
        op_with_reps = op_base.repeat(initial_repetitions, rep_ids)
        assert op_base ** initial_repetitions != op_with_reps
        assert (op_base ** initial_repetitions).replace(repetition_ids=rep_ids) == op_with_reps
    assert op_with_reps.repetitions == initial_repetitions
    assert op_with_reps.repetition_ids == rep_ids
    assert op_with_reps.repeat(1) is op_with_reps

    final_repetitions = 2 * initial_repetitions

    op_with_consecutive_reps = op_with_reps.repeat(2)
    assert op_with_consecutive_reps.repetitions == final_repetitions
    assert op_with_consecutive_reps.repetition_ids == _full_join_string_lists(['0', '1'], rep_ids)
    assert op_base ** final_repetitions != op_with_consecutive_reps

    op_with_consecutive_reps = op_with_reps.repeat(2, ['a', 'b'])
    assert op_with_reps.repeat(repetition_ids=['a', 'b']) == op_with_consecutive_reps
    assert op_with_consecutive_reps.repetitions == final_repetitions
    assert op_with_consecutive_reps.repetition_ids == _full_join_string_lists(['a', 'b'], rep_ids)

    with pytest.raises(ValueError, match='length to be 2'):
        _ = op_with_reps.repeat(2, ['a', 'b', 'c'])

    with pytest.raises(
        ValueError, match='At least one of repetitions and repetition_ids must be set'
    ):
        _ = op_base.repeat()

    with pytest.raises(TypeError, match='Only integer repetitions are allowed'):
        _ = op_base.repeat(1.3)


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
    assert str(op0) == f"[  ]"

    fc0_global_phase_inner = cirq.FrozenCircuit(
        cirq.global_phase_operation(1j), cirq.global_phase_operation(1j)
    )
    op0_global_phase_inner = cirq.CircuitOperation(fc0_global_phase_inner)
    fc0_global_phase_outer = cirq.FrozenCircuit(
        op0_global_phase_inner, cirq.global_phase_operation(1j)
    )
    op0_global_phase_outer = cirq.CircuitOperation(fc0_global_phase_outer)
    assert (
        str(op0_global_phase_outer)
        == f"""\
[                       ]
[                       ]
[ global phase:   -0.5π ]"""
    )

    fc1 = cirq.FrozenCircuit(cirq.X(x), cirq.H(y), cirq.CX(y, z), cirq.measure(x, y, z, key='m'))
    op1 = cirq.CircuitOperation(fc1)
    assert (
        str(op1)
        == f"""\
[ 0: ───X───────M('m')─── ]
[               │         ]
[ 1: ───H───@───M──────── ]
[           │   │         ]
[ 2: ───────X───M──────── ]"""
    )
    assert (
        repr(op1)
        == """\
cirq.CircuitOperation(
    circuit=cirq.FrozenCircuit([
        cirq.Moment(
            cirq.X(cirq.LineQubit(0)),
            cirq.H(cirq.LineQubit(1)),
        ),
        cirq.Moment(
            cirq.CNOT(cirq.LineQubit(1), cirq.LineQubit(2)),
        ),
        cirq.Moment(
            cirq.measure(cirq.LineQubit(0), cirq.LineQubit(1), cirq.LineQubit(2), key=cirq.MeasurementKey(name='m')),
        ),
    ]),
)"""
    )

    fc2 = cirq.FrozenCircuit(cirq.X(x), cirq.H(y), cirq.CX(y, x))
    op2 = cirq.CircuitOperation(
        circuit=fc2,
        qubit_map=({y: z}),
        repetitions=3,
        parent_path=('outer', 'inner'),
        repetition_ids=['a', 'b', 'c'],
    )
    assert (
        str(op2)
        == f"""\
[ 0: ───X───X─── ]
[           │    ]
[ 1: ───H───@─── ](qubit_map={{1: 2}}, parent_path=('outer', 'inner'),\
 repetition_ids=['a', 'b', 'c'])"""
    )
    assert (
        repr(op2)
        == """\
cirq.CircuitOperation(
    circuit=cirq.FrozenCircuit([
        cirq.Moment(
            cirq.X(cirq.LineQubit(0)),
            cirq.H(cirq.LineQubit(1)),
        ),
        cirq.Moment(
            cirq.CNOT(cirq.LineQubit(1), cirq.LineQubit(0)),
        ),
    ]),
    repetitions=3,
    qubit_map={cirq.LineQubit(1): cirq.LineQubit(2)},
    parent_path=('outer', 'inner'),
    repetition_ids=['a', 'b', 'c'],
)"""
    )

    fc3 = cirq.FrozenCircuit(
        cirq.X(x) ** sympy.Symbol('b'),
        cirq.measure(x, key='m'),
    )
    op3 = cirq.CircuitOperation(
        circuit=fc3,
        qubit_map={x: y},
        measurement_key_map={'m': 'p'},
        param_resolver={sympy.Symbol('b'): 2},
    )
    indented_fc3_repr = repr(fc3).replace('\n', '\n    ')
    assert (
        str(op3)
        == f"""\
[ 0: ───X^b───M('m')─── ](qubit_map={{0: 1}}, \
key_map={{m: p}}, params={{b: 2}})"""
    )
    assert (
        repr(op3)
        == f"""\
cirq.CircuitOperation(
    circuit={indented_fc3_repr},
    qubit_map={{cirq.LineQubit(0): cirq.LineQubit(1)}},
    measurement_key_map={{'m': 'p'}},
    param_resolver=cirq.ParamResolver({{sympy.Symbol('b'): 2}}),
)"""
    )

    fc4 = cirq.FrozenCircuit(cirq.X(y))
    op4 = cirq.CircuitOperation(fc4)
    fc5 = cirq.FrozenCircuit(cirq.X(x), op4)
    op5 = cirq.CircuitOperation(fc5)
    assert (
        repr(op5)
        == """\
cirq.CircuitOperation(
    circuit=cirq.FrozenCircuit([
        cirq.Moment(
            cirq.X(cirq.LineQubit(0)),
            cirq.CircuitOperation(
                circuit=cirq.FrozenCircuit([
                    cirq.Moment(
                        cirq.X(cirq.LineQubit(1)),
                    ),
                ]),
            ),
        ),
    ]),
)"""
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
    op = cirq.CircuitOperation(
        circuit=circuit,
        qubit_map={c: b, b: c},
        measurement_key_map={'m': 'p'},
        param_resolver={'exp': 'theta'},
        parent_path=('nested', 'path'),
    )

    assert op._json_dict_() == {
        'circuit': circuit,
        'repetitions': 1,
        'qubit_map': sorted([(k, v) for k, v in op.qubit_map.items()]),
        'measurement_key_map': op.measurement_key_map,
        'param_resolver': op.param_resolver,
        'parent_path': op.parent_path,
        'repetition_ids': None,
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
    assert c.are_any_measurements_terminal()

    c = cirq.Circuit(cirq.X(b), op)
    assert c.are_all_measurements_terminal()
    assert c.are_any_measurements_terminal()

    c = cirq.Circuit(cirq.measure(a), op)
    assert not c.are_all_measurements_terminal()
    assert c.are_any_measurements_terminal()

    c = cirq.Circuit(cirq.measure(b), op)
    assert not c.are_all_measurements_terminal()
    assert c.are_any_measurements_terminal()

    c = cirq.Circuit(op, cirq.X(a))
    assert c.are_all_measurements_terminal()
    assert c.are_any_measurements_terminal()

    c = cirq.Circuit(op, cirq.X(b))
    assert not c.are_all_measurements_terminal()
    assert not c.are_any_measurements_terminal()

    c = cirq.Circuit(op, cirq.measure(a))
    assert c.are_all_measurements_terminal()
    assert c.are_any_measurements_terminal()

    c = cirq.Circuit(op, cirq.measure(b))
    assert not c.are_all_measurements_terminal()
    assert c.are_any_measurements_terminal()


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
    assert not c.are_any_measurements_terminal()

    op = op.with_tags('test')
    c = cirq.Circuit(cirq.X(a), op)
    assert not isinstance(op, cirq.CircuitOperation)
    assert not c.are_all_measurements_terminal()
    assert not c.are_any_measurements_terminal()


def test_decompose_applies_maps():
    a, b, c = cirq.LineQubit.range(3)
    exp = sympy.Symbol('exp')
    theta = sympy.Symbol('theta')
    circuit = cirq.FrozenCircuit(
        cirq.X(a) ** theta,
        cirq.Y(b),
        cirq.H(c),
        cirq.CX(a, b) ** exp,
        cirq.measure(a, b, c, key='m'),
    )
    op = cirq.CircuitOperation(
        circuit=circuit,
        qubit_map={
            c: b,
            b: c,
        },
        measurement_key_map={'m': 'p'},
        param_resolver={exp: theta, theta: exp},
    )

    expected_circuit = cirq.Circuit(
        cirq.X(a) ** exp,
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


def test_decompose_loops_with_measurements():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.FrozenCircuit(
        cirq.H(a),
        cirq.CX(a, b),
        cirq.measure(a, b, key='m'),
    )
    base_op = cirq.CircuitOperation(circuit)

    op = base_op.with_qubits(b, a).repeat(3)
    expected_circuit = cirq.Circuit(
        cirq.H(b),
        cirq.CX(b, a),
        cirq.measure(b, a, key=cirq.MeasurementKey.parse_serialized('0:m')),
        cirq.H(b),
        cirq.CX(b, a),
        cirq.measure(b, a, key=cirq.MeasurementKey.parse_serialized('1:m')),
        cirq.H(b),
        cirq.CX(b, a),
        cirq.measure(b, a, key=cirq.MeasurementKey.parse_serialized('2:m')),
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
        op2.with_params({exp1: exp_one})
        .with_measurement_key_mapping({'ma': 'ma1'})
        .with_measurement_key_mapping({'mb': 'mb1'})
        .with_measurement_key_mapping({'mc': 'mc1'})
        .with_measurement_key_mapping({'md': 'md1'}),
        op2.with_params({exp1: exp_two})
        .with_measurement_key_mapping({'ma': 'ma2'})
        .with_measurement_key_mapping({'mb': 'mb2'})
        .with_measurement_key_mapping({'mc': 'mc2'})
        .with_measurement_key_mapping({'md': 'md2'}),
    )
    op3 = cirq.CircuitOperation(circuit3)

    final_op = op3.with_params({exp_half: 0.5, exp_one: 1.0, exp_two: 2.0})

    expected_circuit1 = cirq.Circuit(
        op2.with_params({exp1: 0.5, exp_half: 0.5, exp_one: 1.0, exp_two: 2.0}),
        op2.with_params({exp1: 1.0, exp_half: 0.5, exp_one: 1.0, exp_two: 2.0})
        .with_measurement_key_mapping({'ma': 'ma1'})
        .with_measurement_key_mapping({'mb': 'mb1'})
        .with_measurement_key_mapping({'mc': 'mc1'})
        .with_measurement_key_mapping({'md': 'md1'}),
        op2.with_params({exp1: 2.0, exp_half: 0.5, exp_one: 1.0, exp_two: 2.0})
        .with_measurement_key_mapping({'ma': 'ma2'})
        .with_measurement_key_mapping({'mb': 'mb2'})
        .with_measurement_key_mapping({'mc': 'mc2'})
        .with_measurement_key_mapping({'md': 'md2'}),
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
        cirq.measure(a, key='ma1'),
        cirq.X(b) ** 1.0,
        cirq.measure(b, key='mb1'),
        cirq.X(c) ** 1.0,
        cirq.measure(c, key='mc1'),
        cirq.X(d) ** 1.0,
        cirq.measure(d, key='md1'),
        cirq.X(a) ** 2.0,
        cirq.measure(a, key='ma2'),
        cirq.X(b) ** 2.0,
        cirq.measure(b, key='mb2'),
        cirq.X(c) ** 2.0,
        cirq.measure(c, key='mc2'),
        cirq.X(d) ** 2.0,
        cirq.measure(d, key='md2'),
    )
    assert cirq.Circuit(cirq.decompose(final_op)) == expected_circuit
    # Verify that mapped_circuit gives the same operations.
    assert final_op.mapped_circuit(deep=True) == expected_circuit


def test_decompose_repeated_nested_measurements():
    # Details of this test described at
    # https://tinyurl.com/measurement-repeated-circuitop#heading=h.sbgxcsyin9wt.
    a = cirq.LineQubit(0)

    op1 = (
        cirq.CircuitOperation(cirq.FrozenCircuit(cirq.measure(a, key='A')))
        .with_measurement_key_mapping({'A': 'B'})
        .repeat(2, ['zero', 'one'])
    )

    op2 = (
        cirq.CircuitOperation(cirq.FrozenCircuit(cirq.measure(a, key='P'), op1))
        .with_measurement_key_mapping({'B': 'C', 'P': 'Q'})
        .repeat(2, ['zero', 'one'])
    )

    op3 = (
        cirq.CircuitOperation(cirq.FrozenCircuit(cirq.measure(a, key='X'), op2))
        .with_measurement_key_mapping({'C': 'D', 'X': 'Y'})
        .repeat(2, ['zero', 'one'])
    )

    expected_measurement_keys_in_order = [
        'zero:Y',
        'zero:zero:Q',
        'zero:zero:zero:D',
        'zero:zero:one:D',
        'zero:one:Q',
        'zero:one:zero:D',
        'zero:one:one:D',
        'one:Y',
        'one:zero:Q',
        'one:zero:zero:D',
        'one:zero:one:D',
        'one:one:Q',
        'one:one:zero:D',
        'one:one:one:D',
    ]
    assert cirq.measurement_key_names(op3) == set(expected_measurement_keys_in_order)

    expected_circuit = cirq.Circuit()
    for key in expected_measurement_keys_in_order:
        expected_circuit.append(cirq.measure(a, key=cirq.MeasurementKey.parse_serialized(key)))

    assert cirq.Circuit(cirq.decompose(op3)) == expected_circuit
    assert cirq.measurement_key_names(expected_circuit) == set(expected_measurement_keys_in_order)

    # Verify that mapped_circuit gives the same operations.
    assert op3.mapped_circuit(deep=True) == expected_circuit


def test_keys_under_parent_path():
    a = cirq.LineQubit(0)
    op1 = cirq.CircuitOperation(cirq.FrozenCircuit(cirq.measure(a, key='A')))
    assert cirq.measurement_key_names(op1) == {'A'}
    op2 = op1.with_key_path(('B',))
    assert cirq.measurement_key_names(op2) == {'B:A'}
    op3 = op2.repeat(2)
    assert cirq.measurement_key_names(op3) == {'B:0:A', 'B:1:A'}


def test_mapped_circuit_preserves_moments():
    q0, q1 = cirq.LineQubit.range(2)
    fc = cirq.FrozenCircuit(cirq.Moment(cirq.X(q0)), cirq.Moment(cirq.X(q1)))
    op = cirq.CircuitOperation(fc)
    assert op.mapped_circuit() == fc
    assert op.repeat(3).mapped_circuit(deep=True) == fc * 3


def test_mapped_op():
    q0, q1 = cirq.LineQubit.range(2)
    a, b = (sympy.Symbol(x) for x in 'ab')
    fc1 = cirq.FrozenCircuit(cirq.X(q0) ** a, cirq.measure(q0, q1, key='m'))
    op1 = (
        cirq.CircuitOperation(fc1)
        .with_params({'a': 'b'})
        .with_qubits(q1, q0)
        .with_measurement_key_mapping({'m': 'k'})
    )
    fc2 = cirq.FrozenCircuit(cirq.X(q1) ** b, cirq.measure(q1, q0, key='k'))
    op2 = cirq.CircuitOperation(fc2)

    assert op1.mapped_op() == op2


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


def test_mapped_circuit_keeps_keys_under_parent_path():
    q = cirq.LineQubit(0)
    op1 = cirq.CircuitOperation(
        cirq.FrozenCircuit(
            cirq.measure(q, key='A'),
            cirq.measure_single_paulistring(cirq.X(q), key='B'),
            cirq.MixedUnitaryChannel.from_mixture(cirq.bit_flip(0.5), key='C').on(q),
            cirq.KrausChannel.from_channel(cirq.phase_damp(0.5), key='D').on(q),
        )
    )
    op2 = op1.with_key_path(('X',))
    assert cirq.measurement_key_names(op2.mapped_circuit()) == {'X:A', 'X:B', 'X:C', 'X:D'}


def test_keys_conflict_no_repetitions():
    q = cirq.LineQubit(0)
    op1 = cirq.CircuitOperation(
        cirq.FrozenCircuit(
            cirq.measure(q, key='A'),
        )
    )
    op2 = cirq.CircuitOperation(cirq.FrozenCircuit(op1, op1))
    with pytest.raises(ValueError, match='Conflicting measurement keys found: A'):
        _ = op2.mapped_circuit(deep=True)


def test_keys_conflict_locally():
    q = cirq.LineQubit(0)
    op1 = cirq.measure(q, key='A')
    op2 = cirq.CircuitOperation(cirq.FrozenCircuit(op1, op1))
    with pytest.raises(ValueError, match='Conflicting measurement keys found: A'):
        _ = op2.mapped_circuit()


# TODO: Operation has a "gate" property. What is this for a CircuitOperation?
