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
import unittest.mock as mock
from typing import Optional

import numpy as np
import pytest
import sympy

import cirq
import cirq.circuits.circuit_operation as circuit_operation
from cirq import _compat
from cirq.circuits.circuit_operation import _full_join_string_lists

ALL_SIMULATORS = (cirq.Simulator(), cirq.DensityMatrixSimulator(), cirq.CliffordSimulator())


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
    cache_name = _compat._method_cache_name(circuit._is_measurement_)
    assert not hasattr(circuit, cache_name)
    # Memoize `_is_measurement_` in the circuit.
    assert cirq.is_measurement(c_op)
    assert hasattr(circuit, cache_name)


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

    def map_fn(qubit: cirq.Qid) -> cirq.Qid:
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
    circuit = cirq.FrozenCircuit(cirq.X(a), cirq.measure(b, key='mb'), cirq.measure(a, key='ma'))
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

    param_dict = {z_exp: 2, x_exp: theta, sympy.Symbol('k'): sympy.Symbol('phi')}
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


def test_recursive_params():
    q = cirq.LineQubit(0)
    a, a2, b, b2 = sympy.symbols('a a2 b b2')
    circuitop = cirq.CircuitOperation(
        cirq.FrozenCircuit(cirq.X(q) ** a, cirq.Z(q) ** b),
        # Not recursive, a and b are swapped.
        param_resolver=cirq.ParamResolver({a: b, b: a}),
    )
    # Recursive, so a->a2->0 and b->b2->1.
    outer_params = {a: a2, a2: 0, b: b2, b2: 1}
    resolved = cirq.resolve_parameters(circuitop, outer_params)
    # Combined, a->b->b2->1, and b->a->a2->0.
    assert resolved.param_resolver.param_dict == {a: 1, b: 0}

    # Non-recursive, so a->a2 and b->b2.
    resolved = cirq.resolve_parameters(circuitop, outer_params, recursive=False)
    # Combined, a->b->b2, and b->a->a2.
    assert resolved.param_resolver.param_dict == {a: b2, b: a2}

    with pytest.raises(RecursionError):
        cirq.resolve_parameters(circuitop, {a: a2, a2: a})

    # Non-recursive, so a->b and b->a.
    resolved = cirq.resolve_parameters(circuitop, {a: b, b: a}, recursive=False)
    # Combined, a->b->a, and b->a->b.
    assert resolved.param_resolver.param_dict == {}

    # First example should behave like an X when simulated
    result = cirq.Simulator().simulate(cirq.Circuit(circuitop), param_resolver=outer_params)
    assert np.allclose(result.state_vector(), [0, 1])


@pytest.mark.parametrize('add_measurements', [True, False])
@pytest.mark.parametrize('use_default_ids_for_initial_rep', [True, False])
def test_repeat(add_measurements: bool, use_default_ids_for_initial_rep: bool) -> None:
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

    op_with_reps: Optional[cirq.CircuitOperation] = None
    rep_ids = []
    if use_default_ids_for_initial_rep:
        op_with_reps = op_base.repeat(initial_repetitions)
        rep_ids = ['0', '1', '2']
        assert op_base**initial_repetitions == op_with_reps
    else:
        rep_ids = ['a', 'b', 'c']
        op_with_reps = op_base.repeat(initial_repetitions, rep_ids)
        assert op_base**initial_repetitions != op_with_reps
        assert (op_base**initial_repetitions).replace(repetition_ids=rep_ids) == op_with_reps
    assert op_with_reps.repetitions == initial_repetitions
    assert op_with_reps.repetition_ids == rep_ids
    assert op_with_reps.repeat(1) is op_with_reps

    final_repetitions = 2 * initial_repetitions

    op_with_consecutive_reps = op_with_reps.repeat(2)
    assert op_with_consecutive_reps.repetitions == final_repetitions
    assert op_with_consecutive_reps.repetition_ids == _full_join_string_lists(['0', '1'], rep_ids)
    assert op_base**final_repetitions != op_with_consecutive_reps

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

    with pytest.raises(TypeError, match='Only integer or sympy repetitions are allowed'):
        _ = op_base.repeat(1.3)
    assert op_base.repeat(3.00000000001).repetitions == 3
    assert op_base.repeat(2.99999999999).repetitions == 3


# TODO: #7232 - enable and fix immediately after the 1.5.0 release
@pytest.mark.xfail(reason='broken by rollback of use_repetition_ids for #7232')
def test_replace_repetition_ids() -> None:
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(a), cirq.CX(a, b), cirq.M(b, key='mb'), cirq.M(a, key='ma'))
    op = cirq.CircuitOperation(circuit.freeze())
    assert op.repetitions == 1
    assert not op.use_repetition_ids

    op2 = op.replace(repetitions=2)
    assert op2.repetitions == 2
    assert not op2.use_repetition_ids

    op3 = op.replace(repetitions=3, repetition_ids=None)
    assert op3.repetitions == 3
    assert not op3.use_repetition_ids

    # Passing `repetition_ids` will also enable `use_repetition_ids`
    op4 = op.replace(repetitions=4, repetition_ids=['a', 'b', 'c', 'd'])
    assert op4.repetitions == 4
    assert op4.use_repetition_ids
    assert op4.repetition_ids == ['a', 'b', 'c', 'd']


@pytest.mark.parametrize('add_measurements', [True, False])
@pytest.mark.parametrize('use_repetition_ids', [True, False])
@pytest.mark.parametrize('initial_reps', [0, 1, 2, 3])
def test_repeat_zero_times(add_measurements, use_repetition_ids, initial_reps):
    q = cirq.LineQubit(0)
    subcircuit = cirq.Circuit(cirq.X(q))
    if add_measurements:
        subcircuit.append(cirq.measure(q))

    op = cirq.CircuitOperation(
        subcircuit.freeze(), repetitions=initial_reps, use_repetition_ids=use_repetition_ids
    )
    result = cirq.Simulator().simulate(cirq.Circuit(op))
    assert np.allclose(result.state_vector(), [0, 1] if initial_reps % 2 else [1, 0])
    result = cirq.Simulator().simulate(cirq.Circuit(op**0))
    assert np.allclose(result.state_vector(), [1, 0])


def test_no_repetition_ids():
    def default_repetition_ids(self):  # pragma: no cover
        assert False, "Should not call default_repetition_ids"

    with mock.patch.object(circuit_operation, 'default_repetition_ids', new=default_repetition_ids):
        q = cirq.LineQubit(0)
        op = cirq.CircuitOperation(
            cirq.Circuit(cirq.X(q), cirq.measure(q)).freeze(),
            repetitions=1_000_000,
            use_repetition_ids=False,
        )
        assert op.repetitions == 1_000_000
        assert op.repetition_ids is None
        _ = repr(op)
        _ = str(op)

        op2 = op.repeat(10)
        assert op2.repetitions == 10_000_000
        assert op2.repetition_ids is None


def test_parameterized_repeat():
    q = cirq.LineQubit(0)
    op = cirq.CircuitOperation(cirq.FrozenCircuit(cirq.X(q))) ** sympy.Symbol('a')
    assert cirq.parameter_names(op) == {'a'}
    assert not cirq.has_unitary(op)
    result = cirq.Simulator().simulate(cirq.Circuit(op), param_resolver={'a': 0})
    assert np.allclose(result.state_vector(), [1, 0])
    result = cirq.Simulator().simulate(cirq.Circuit(op), param_resolver={'a': 1})
    assert np.allclose(result.state_vector(), [0, 1])
    result = cirq.Simulator().simulate(cirq.Circuit(op), param_resolver={'a': 2})
    assert np.allclose(result.state_vector(), [1, 0])
    result = cirq.Simulator().simulate(cirq.Circuit(op), param_resolver={'a': -1})
    assert np.allclose(result.state_vector(), [0, 1])
    with pytest.raises(TypeError, match='Only integer or sympy repetitions are allowed'):
        cirq.Simulator().simulate(cirq.Circuit(op), param_resolver={'a': 1.5})
    with pytest.raises(ValueError, match='Circuit contains ops whose symbols were not specified'):
        cirq.Simulator().simulate(cirq.Circuit(op))
    op = op**-1
    assert cirq.parameter_names(op) == {'a'}
    assert not cirq.has_unitary(op)
    result = cirq.Simulator().simulate(cirq.Circuit(op), param_resolver={'a': 0})
    assert np.allclose(result.state_vector(), [1, 0])
    result = cirq.Simulator().simulate(cirq.Circuit(op), param_resolver={'a': 1})
    assert np.allclose(result.state_vector(), [0, 1])
    result = cirq.Simulator().simulate(cirq.Circuit(op), param_resolver={'a': 2})
    assert np.allclose(result.state_vector(), [1, 0])
    result = cirq.Simulator().simulate(cirq.Circuit(op), param_resolver={'a': -1})
    assert np.allclose(result.state_vector(), [0, 1])
    with pytest.raises(TypeError, match='Only integer or sympy repetitions are allowed'):
        cirq.Simulator().simulate(cirq.Circuit(op), param_resolver={'a': 1.5})
    with pytest.raises(ValueError, match='Circuit contains ops whose symbols were not specified'):
        cirq.Simulator().simulate(cirq.Circuit(op))
    op = op ** sympy.Symbol('b')
    assert cirq.parameter_names(op) == {'a', 'b'}
    assert not cirq.has_unitary(op)
    result = cirq.Simulator().simulate(cirq.Circuit(op), param_resolver={'a': 1, 'b': 1})
    assert np.allclose(result.state_vector(), [0, 1])
    result = cirq.Simulator().simulate(cirq.Circuit(op), param_resolver={'a': 2, 'b': 1})
    assert np.allclose(result.state_vector(), [1, 0])
    result = cirq.Simulator().simulate(cirq.Circuit(op), param_resolver={'a': 1, 'b': 2})
    assert np.allclose(result.state_vector(), [1, 0])
    with pytest.raises(TypeError, match='Only integer or sympy repetitions are allowed'):
        cirq.Simulator().simulate(cirq.Circuit(op), param_resolver={'a': 1.5, 'b': 1})
    with pytest.raises(ValueError, match='Circuit contains ops whose symbols were not specified'):
        cirq.Simulator().simulate(cirq.Circuit(op))
    op = op**2.0
    assert cirq.parameter_names(op) == {'a', 'b'}
    assert not cirq.has_unitary(op)
    result = cirq.Simulator().simulate(cirq.Circuit(op), param_resolver={'a': 1, 'b': 1})
    assert np.allclose(result.state_vector(), [1, 0])
    result = cirq.Simulator().simulate(cirq.Circuit(op), param_resolver={'a': 1.5, 'b': 1})
    assert np.allclose(result.state_vector(), [0, 1])
    result = cirq.Simulator().simulate(cirq.Circuit(op), param_resolver={'a': 1, 'b': 1.5})
    assert np.allclose(result.state_vector(), [0, 1])
    with pytest.raises(TypeError, match='Only integer or sympy repetitions are allowed'):
        cirq.Simulator().simulate(cirq.Circuit(op), param_resolver={'a': 1.5, 'b': 1.5})
    with pytest.raises(ValueError, match='Circuit contains ops whose symbols were not specified'):
        cirq.Simulator().simulate(cirq.Circuit(op))


def test_parameterized_repeat_side_effects():
    q = cirq.LineQubit(0)
    op = cirq.CircuitOperation(
        cirq.FrozenCircuit(cirq.X(q).with_classical_controls('c'), cirq.measure(q, key='m')),
        repetitions=sympy.Symbol('a'),
    )

    # Control keys can be calculated because they only "lift" if there's a matching
    # measurement, in which case they're not returned here.
    assert cirq.control_keys(op) == {cirq.MeasurementKey('c')}

    # "local" params do not bind to the repetition param.
    assert cirq.parameter_names(op.with_params({'a': 1})) == {'a'}

    # Check errors that require unrolling the circuit.
    with pytest.raises(
        ValueError, match='Cannot unroll circuit due to nondeterministic repetitions'
    ):
        cirq.measurement_key_objs(op)
    with pytest.raises(
        ValueError, match='Cannot unroll circuit due to nondeterministic repetitions'
    ):
        cirq.measurement_key_names(op)
    with pytest.raises(
        ValueError, match='Cannot unroll circuit due to nondeterministic repetitions'
    ):
        op.mapped_circuit()
    with pytest.raises(
        ValueError, match='Cannot unroll circuit due to nondeterministic repetitions'
    ):
        cirq.decompose(op)

    # Not compatible with repetition ids
    with pytest.raises(ValueError, match='repetition ids with parameterized repetitions'):
        op.with_repetition_ids(['x', 'y'])
    with pytest.raises(ValueError, match='repetition ids with parameterized repetitions'):
        op.repeat(repetition_ids=['x', 'y'])

    # TODO(daxfohl): This should work, but likely requires a new protocol that returns *just* the
    # name of the measurement keys. (measurement_key_names returns the full serialized string).
    with pytest.raises(
        ValueError, match='Cannot unroll circuit due to nondeterministic repetitions'
    ):
        cirq.with_measurement_key_mapping(op, {'m': 'm2'})

    # Everything should work once resolved
    op = cirq.resolve_parameters(op, {'a': 2})
    assert set(map(str, cirq.measurement_key_objs(op))) == {'0:m', '1:m'}
    assert op.mapped_circuit() == cirq.Circuit(
        cirq.X(q).with_classical_controls('c'),
        cirq.measure(q, key=cirq.MeasurementKey.parse_serialized('0:m')),
        cirq.X(q).with_classical_controls('c'),
        cirq.measure(q, key=cirq.MeasurementKey.parse_serialized('1:m')),
    )
    assert cirq.decompose(op) == cirq.decompose(
        cirq.Circuit(
            cirq.X(q).with_classical_controls('c'),
            cirq.measure(q, key=cirq.MeasurementKey.parse_serialized('0:m')),
            cirq.X(q).with_classical_controls('c'),
            cirq.measure(q, key=cirq.MeasurementKey.parse_serialized('1:m')),
        )
    )


def test_parameterized_repeat_side_effects_when_not_using_rep_ids():
    q = cirq.LineQubit(0)
    op = cirq.CircuitOperation(
        cirq.FrozenCircuit(cirq.X(q).with_classical_controls('c'), cirq.measure(q, key='m')),
        repetitions=sympy.Symbol('a'),
        use_repetition_ids=False,
    )
    assert cirq.control_keys(op) == {cirq.MeasurementKey('c')}
    assert cirq.parameter_names(op.with_params({'a': 1})) == {'a'}
    assert set(map(str, cirq.measurement_key_objs(op))) == {'m'}
    assert cirq.measurement_key_names(op) == {'m'}
    assert cirq.measurement_key_names(cirq.with_measurement_key_mapping(op, {'m': 'm2'})) == {'m2'}
    with pytest.raises(
        ValueError, match='Cannot unroll circuit due to nondeterministic repetitions'
    ):
        op.mapped_circuit()
    with pytest.raises(
        ValueError, match='Cannot unroll circuit due to nondeterministic repetitions'
    ):
        cirq.decompose(op)
    with pytest.raises(ValueError, match='repetition ids with parameterized repetitions'):
        op.with_repetition_ids(['x', 'y'])
    with pytest.raises(ValueError, match='repetition ids with parameterized repetitions'):
        op.repeat(repetition_ids=['x', 'y'])


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
    assert str(op0) == "[  ]"

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
        == """\
[                       ]
[                       ]
[ global phase:   -0.5π ]"""
    )

    fc1 = cirq.FrozenCircuit(cirq.X(x), cirq.H(y), cirq.CX(y, z), cirq.measure(x, y, z, key='m'))
    op1 = cirq.CircuitOperation(fc1)
    assert (
        str(op1)
        == """\
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
        == """\
[ 0: ───X───X─── ]
[           │    ]
[ 1: ───H───@─── ](qubit_map={q(1): q(2)}, parent_path=('outer', 'inner'),\
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

    fc3 = cirq.FrozenCircuit(cirq.X(x) ** sympy.Symbol('b'), cirq.measure(x, key='m'))
    op3 = cirq.CircuitOperation(
        circuit=fc3,
        qubit_map={x: y},
        measurement_key_map={'m': 'p'},
        param_resolver={sympy.Symbol('b'): 2},
    )
    indented_fc3_repr = repr(fc3).replace('\n', '\n    ')
    assert (
        str(op3)
        == """\
[ 0: ───X^b───M('m')─── ](qubit_map={q(0): q(1)}, \
key_map={m: p}, params={b: 2})"""
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
    op6 = cirq.CircuitOperation(fc5, use_repetition_ids=False)
    assert (
        repr(op6)
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
    use_repetition_ids=False,
)"""
    )
    op7 = cirq.CircuitOperation(
        cirq.FrozenCircuit(cirq.measure(x, key='a')),
        use_repetition_ids=False,
        repeat_until=cirq.KeyCondition(cirq.MeasurementKey('a')),
    )
    assert (
        repr(op7)
        == """\
cirq.CircuitOperation(
    circuit=cirq.FrozenCircuit([
        cirq.Moment(
            cirq.measure(cirq.LineQubit(0), key=cirq.MeasurementKey(name='a')),
        ),
    ]),
    use_repetition_ids=False,
    repeat_until=cirq.KeyCondition(cirq.MeasurementKey(name='a')),
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
    fc = cirq.FrozenCircuit(cirq.H(a), cirq.measure(b, key='m1'))
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
    fc = cirq.FrozenCircuit(cirq.H(a), cirq.measure(b, key='m1'), cirq.X(b))
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
        qubit_map={c: b, b: c},
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
    circuit = cirq.FrozenCircuit(cirq.H(a), cirq.CX(a, b))
    base_op = cirq.CircuitOperation(circuit)

    op = base_op.with_qubits(b, a).repeat(3)
    expected_circuit = cirq.Circuit(
        cirq.H(b), cirq.CX(b, a), cirq.H(b), cirq.CX(b, a), cirq.H(b), cirq.CX(b, a)
    )
    assert cirq.Circuit(cirq.decompose_once(op)) == expected_circuit

    op = base_op.repeat(-2)
    expected_circuit = cirq.Circuit(cirq.CX(a, b), cirq.H(a), cirq.CX(a, b), cirq.H(a))
    assert cirq.Circuit(cirq.decompose_once(op)) == expected_circuit


def test_decompose_loops_with_measurements():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.FrozenCircuit(cirq.H(a), cirq.CX(a, b), cirq.measure(a, b, key='m'))
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
    op3 = cirq.with_key_path_prefix(op2, ('C',))
    assert cirq.measurement_key_names(op3) == {'C:B:A'}
    op4 = op3.repeat(2)
    assert cirq.measurement_key_names(op4) == {'C:B:0:A', 'C:B:1:A'}


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
    circuit = cirq.FrozenCircuit(cirq.X(a), cirq.H(b), cirq.H(c), cirq.CZ(a, c))
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


def test_mapped_circuit_allows_repeated_keys():
    q = cirq.LineQubit(0)
    op1 = cirq.CircuitOperation(cirq.FrozenCircuit(cirq.measure(q, key='A')))
    op2 = cirq.CircuitOperation(cirq.FrozenCircuit(op1, op1))
    circuit = op2.mapped_circuit(deep=True)
    cirq.testing.assert_has_diagram(
        circuit, "0: ───M('A')───M('A')───", use_unicode_characters=True
    )
    op1 = cirq.measure(q, key='A')
    op2 = cirq.CircuitOperation(cirq.FrozenCircuit(op1, op1))
    circuit = op2.mapped_circuit()
    cirq.testing.assert_has_diagram(
        circuit, "0: ───M('A')───M('A')───", use_unicode_characters=True
    )


@pytest.mark.parametrize('sim', ALL_SIMULATORS)
def test_simulate_no_repetition_ids_both_levels(sim):
    q = cirq.LineQubit(0)
    inner = cirq.Circuit(cirq.measure(q, key='a'))
    middle = cirq.Circuit(
        cirq.CircuitOperation(inner.freeze(), repetitions=2, use_repetition_ids=False)
    )
    outer_subcircuit = cirq.CircuitOperation(
        middle.freeze(), repetitions=2, use_repetition_ids=False
    )
    circuit = cirq.Circuit(outer_subcircuit)
    result = sim.run(circuit)
    assert result.records['a'].shape == (1, 4, 1)


@pytest.mark.parametrize('sim', ALL_SIMULATORS)
def test_simulate_no_repetition_ids_outer(sim):
    q = cirq.LineQubit(0)
    inner = cirq.Circuit(cirq.measure(q, key='a'))
    middle = cirq.Circuit(cirq.CircuitOperation(inner.freeze(), repetitions=2))
    outer_subcircuit = cirq.CircuitOperation(
        middle.freeze(), repetitions=2, use_repetition_ids=False
    )
    circuit = cirq.Circuit(outer_subcircuit)
    result = sim.run(circuit)
    assert result.records['0:a'].shape == (1, 2, 1)
    assert result.records['1:a'].shape == (1, 2, 1)


@pytest.mark.parametrize('sim', ALL_SIMULATORS)
def test_simulate_no_repetition_ids_inner(sim):
    q = cirq.LineQubit(0)
    inner = cirq.Circuit(cirq.measure(q, key='a'))
    middle = cirq.Circuit(
        cirq.CircuitOperation(inner.freeze(), repetitions=2, use_repetition_ids=False)
    )
    outer_subcircuit = cirq.CircuitOperation(middle.freeze(), repetitions=2)
    circuit = cirq.Circuit(outer_subcircuit)
    result = sim.run(circuit)
    assert result.records['0:a'].shape == (1, 2, 1)
    assert result.records['1:a'].shape == (1, 2, 1)


@pytest.mark.parametrize('sim', ALL_SIMULATORS)
def test_repeat_until(sim):
    q = cirq.LineQubit(0)
    key = cirq.MeasurementKey('m')
    c = cirq.Circuit(
        cirq.X(q),
        cirq.CircuitOperation(
            cirq.FrozenCircuit(cirq.X(q), cirq.measure(q, key=key)),
            use_repetition_ids=False,
            repeat_until=cirq.KeyCondition(key),
        ),
    )
    measurements = sim.run(c).records['m'][0]
    assert len(measurements) == 2
    assert measurements[0] == (0,)
    assert measurements[1] == (1,)


@pytest.mark.parametrize('sim', ALL_SIMULATORS)
def test_repeat_until_sympy(sim):
    q1, q2 = cirq.LineQubit.range(2)
    circuitop = cirq.CircuitOperation(
        cirq.FrozenCircuit(cirq.X(q2), cirq.measure(q2, key='b')),
        use_repetition_ids=False,
        repeat_until=cirq.SympyCondition(sympy.Eq(sympy.Symbol('a'), sympy.Symbol('b'))),
    )
    c = cirq.Circuit(cirq.measure(q1, key='a'), circuitop)
    # Validate commutation
    assert len(c) == 2
    assert cirq.control_keys(circuitop) == {cirq.MeasurementKey('a')}
    measurements = sim.run(c).records['b'][0]
    assert len(measurements) == 2
    assert measurements[0] == (1,)
    assert measurements[1] == (0,)


@pytest.mark.parametrize('sim', [cirq.Simulator(), cirq.DensityMatrixSimulator()])
def test_post_selection(sim):
    q = cirq.LineQubit(0)
    key = cirq.MeasurementKey('m')
    c = cirq.Circuit(
        cirq.CircuitOperation(
            cirq.FrozenCircuit(cirq.X(q) ** 0.2, cirq.measure(q, key=key)),
            use_repetition_ids=False,
            repeat_until=cirq.KeyCondition(key),
        )
    )
    result = sim.run(c)
    assert result.records['m'][0][-1] == (1,)
    for i in range(len(result.records['m'][0]) - 1):
        assert result.records['m'][0][i] == (0,)


def test_repeat_until_diagram():
    q = cirq.LineQubit(0)
    key = cirq.MeasurementKey('m')
    c = cirq.Circuit(
        cirq.CircuitOperation(
            cirq.FrozenCircuit(cirq.X(q) ** 0.2, cirq.measure(q, key=key)),
            use_repetition_ids=False,
            repeat_until=cirq.KeyCondition(key),
        )
    )
    cirq.testing.assert_has_diagram(
        c,
        """
0: ───[ 0: ───X^0.2───M('m')─── ](no_rep_ids, until=m)───
""",
        use_unicode_characters=True,
    )


def test_repeat_until_error():
    q = cirq.LineQubit(0)
    with pytest.raises(ValueError, match='Cannot use repetitions with repeat_until'):
        cirq.CircuitOperation(
            cirq.FrozenCircuit(),
            use_repetition_ids=True,
            repeat_until=cirq.KeyCondition(cirq.MeasurementKey('a')),
        )
    with pytest.raises(ValueError, match='Infinite loop'):
        cirq.CircuitOperation(
            cirq.FrozenCircuit(cirq.measure(q, key='m')),
            use_repetition_ids=False,
            repeat_until=cirq.KeyCondition(cirq.MeasurementKey('a')),
        )


def test_repeat_until_protocols():
    q = cirq.LineQubit(0)
    op = cirq.CircuitOperation(
        cirq.FrozenCircuit(cirq.H(q) ** sympy.Symbol('p'), cirq.measure(q, key='a')),
        repeat_until=cirq.SympyCondition(sympy.Eq(sympy.Symbol('a'), 0)),
        # TODO: #7232 - remove immediately after the 1.5.0 release
        use_repetition_ids=False,
    )
    scoped = cirq.with_rescoped_keys(op, ('0',))
    # Ensure the _repeat_until has been mapped, the measurement has been mapped to the same key,
    # and the control keys of the subcircuit is empty (because the control key of the condition is
    # bound to the measurement).
    assert scoped._mapped_repeat_until.keys == (cirq.MeasurementKey('a', ('0',)),)
    assert cirq.measurement_key_objs(scoped) == {cirq.MeasurementKey('a', ('0',))}
    assert not cirq.control_keys(scoped)
    mapped = cirq.with_measurement_key_mapping(scoped, {'a': 'b'})
    assert mapped._mapped_repeat_until.keys == (cirq.MeasurementKey('b', ('0',)),)
    assert cirq.measurement_key_objs(mapped) == {cirq.MeasurementKey('b', ('0',))}
    assert not cirq.control_keys(mapped)
    prefixed = cirq.with_key_path_prefix(mapped, ('1',))
    assert prefixed._mapped_repeat_until.keys == (cirq.MeasurementKey('b', ('1', '0')),)
    assert cirq.measurement_key_objs(prefixed) == {cirq.MeasurementKey('b', ('1', '0'))}
    assert not cirq.control_keys(prefixed)
    setpath = cirq.with_key_path(prefixed, ('2',))
    assert setpath._mapped_repeat_until.keys == (cirq.MeasurementKey('b', ('2',)),)
    assert cirq.measurement_key_objs(setpath) == {cirq.MeasurementKey('b', ('2',))}
    assert not cirq.control_keys(setpath)
    resolved = cirq.resolve_parameters(setpath, {'p': 1})
    assert resolved._mapped_repeat_until.keys == (cirq.MeasurementKey('b', ('2',)),)
    assert cirq.measurement_key_objs(resolved) == {cirq.MeasurementKey('b', ('2',))}
    assert not cirq.control_keys(resolved)


def test_inner_repeat_until_simulate():
    sim = cirq.Simulator()
    q = cirq.LineQubit(0)
    inner_loop = cirq.CircuitOperation(
        cirq.FrozenCircuit(cirq.H(q), cirq.measure(q, key="inner_loop")),
        repeat_until=cirq.SympyCondition(sympy.Eq(sympy.Symbol("inner_loop"), 0)),
        # TODO: #7232 - remove immediately after the 1.5.0 release
        use_repetition_ids=False,
    )
    outer_loop = cirq.Circuit(inner_loop, cirq.X(q), cirq.measure(q, key="outer_loop"))
    circuit = cirq.Circuit(
        cirq.CircuitOperation(
            cirq.FrozenCircuit(outer_loop), repetitions=2, use_repetition_ids=True
        )
    )
    result = sim.run(circuit, repetitions=1)
    assert all(len(v) == 1 and v[0] == 1 for v in result.records['0:inner_loop'][0][:-1])
    assert result.records['0:inner_loop'][0][-1] == [0]
    assert result.records['0:outer_loop'] == [[[1]]]
    assert all(len(v) == 1 and v[0] == 1 for v in result.records['1:inner_loop'][0][:-1])
    assert result.records['1:inner_loop'][0][-1] == [0]
    assert result.records['1:outer_loop'] == [[[1]]]


# TODO: Operation has a "gate" property. What is this for a CircuitOperation?
