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

import numpy as np
import pytest
import sympy

import cirq
import cirq.testing


def test_validation():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    d = cirq.NamedQubit('d')

    _ = cirq.Moment([])
    _ = cirq.Moment([cirq.X(a)])
    _ = cirq.Moment([cirq.CZ(a, b)])
    _ = cirq.Moment([cirq.CZ(b, d)])
    _ = cirq.Moment([cirq.CZ(a, b), cirq.CZ(c, d)])
    _ = cirq.Moment([cirq.CZ(a, c), cirq.CZ(b, d)])
    _ = cirq.Moment([cirq.CZ(a, c), cirq.X(b)])

    with pytest.raises(ValueError):
        _ = cirq.Moment([cirq.X(a), cirq.X(a)])
    with pytest.raises(ValueError):
        _ = cirq.Moment([cirq.CZ(a, c), cirq.X(c)])
    with pytest.raises(ValueError):
        _ = cirq.Moment([cirq.CZ(a, c), cirq.CZ(c, d)])


def test_equality():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    d = cirq.NamedQubit('d')

    eq = cirq.testing.EqualsTester()

    # Default is empty. Iterables get frozen into tuples.
    eq.add_equality_group(cirq.Moment(), cirq.Moment([]), cirq.Moment(()))
    eq.add_equality_group(cirq.Moment([cirq.X(d)]), cirq.Moment((cirq.X(d),)))

    # Equality depends on gate and qubits.
    eq.add_equality_group(cirq.Moment([cirq.X(a)]))
    eq.add_equality_group(cirq.Moment([cirq.X(b)]))
    eq.add_equality_group(cirq.Moment([cirq.Y(a)]))

    # Equality doesn't depend on order.
    eq.add_equality_group(cirq.Moment([cirq.X(a), cirq.X(b)]), cirq.Moment([cirq.X(a), cirq.X(b)]))

    # Two qubit gates.
    eq.make_equality_group(lambda: cirq.Moment([cirq.CZ(c, d)]))
    eq.make_equality_group(lambda: cirq.Moment([cirq.CZ(a, c)]))
    eq.make_equality_group(lambda: cirq.Moment([cirq.CZ(a, b), cirq.CZ(c, d)]))
    eq.make_equality_group(lambda: cirq.Moment([cirq.CZ(a, c), cirq.CZ(b, d)]))


def test_approx_eq():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    assert not cirq.approx_eq(cirq.Moment([cirq.X(a)]), cirq.X(a))

    # Default is empty. Iterables get frozen into tuples.
    assert cirq.approx_eq(cirq.Moment(), cirq.Moment([]))
    assert cirq.approx_eq(cirq.Moment([]), cirq.Moment(()))

    assert cirq.approx_eq(cirq.Moment([cirq.X(a)]), cirq.Moment([cirq.X(a)]))
    assert not cirq.approx_eq(cirq.Moment([cirq.X(a)]), cirq.Moment([cirq.X(b)]))

    assert cirq.approx_eq(
        cirq.Moment([cirq.XPowGate(exponent=0)(a)]), cirq.Moment([cirq.XPowGate(exponent=1e-9)(a)])
    )
    assert not cirq.approx_eq(
        cirq.Moment([cirq.XPowGate(exponent=0)(a)]), cirq.Moment([cirq.XPowGate(exponent=1e-7)(a)])
    )
    assert cirq.approx_eq(
        cirq.Moment([cirq.XPowGate(exponent=0)(a)]),
        cirq.Moment([cirq.XPowGate(exponent=1e-7)(a)]),
        atol=1e-6,
    )


def test_operates_on_single_qubit():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')

    # Empty case.
    assert not cirq.Moment().operates_on_single_qubit(a)
    assert not cirq.Moment().operates_on_single_qubit(b)

    # One-qubit operation case.
    assert cirq.Moment([cirq.X(a)]).operates_on_single_qubit(a)
    assert not cirq.Moment([cirq.X(a)]).operates_on_single_qubit(b)

    # Two-qubit operation case.
    assert cirq.Moment([cirq.CZ(a, b)]).operates_on_single_qubit(a)
    assert cirq.Moment([cirq.CZ(a, b)]).operates_on_single_qubit(b)
    assert not cirq.Moment([cirq.CZ(a, b)]).operates_on_single_qubit(c)

    # Multiple operations case.
    assert cirq.Moment([cirq.X(a), cirq.X(b)]).operates_on_single_qubit(a)
    assert cirq.Moment([cirq.X(a), cirq.X(b)]).operates_on_single_qubit(b)
    assert not cirq.Moment([cirq.X(a), cirq.X(b)]).operates_on_single_qubit(c)


def test_operates_on():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')

    # Empty case.
    assert not cirq.Moment().operates_on([])
    assert not cirq.Moment().operates_on([a])
    assert not cirq.Moment().operates_on([b])
    assert not cirq.Moment().operates_on([a, b])

    # One-qubit operation case.
    assert not cirq.Moment([cirq.X(a)]).operates_on([])
    assert cirq.Moment([cirq.X(a)]).operates_on([a])
    assert not cirq.Moment([cirq.X(a)]).operates_on([b])
    assert cirq.Moment([cirq.X(a)]).operates_on([a, b])

    # Two-qubit operation case.
    assert not cirq.Moment([cirq.CZ(a, b)]).operates_on([])
    assert cirq.Moment([cirq.CZ(a, b)]).operates_on([a])
    assert cirq.Moment([cirq.CZ(a, b)]).operates_on([b])
    assert cirq.Moment([cirq.CZ(a, b)]).operates_on([a, b])
    assert not cirq.Moment([cirq.CZ(a, b)]).operates_on([c])
    assert cirq.Moment([cirq.CZ(a, b)]).operates_on([a, c])
    assert cirq.Moment([cirq.CZ(a, b)]).operates_on([a, b, c])

    # Multiple operations case.
    assert not cirq.Moment([cirq.X(a), cirq.X(b)]).operates_on([])
    assert cirq.Moment([cirq.X(a), cirq.X(b)]).operates_on([a])
    assert cirq.Moment([cirq.X(a), cirq.X(b)]).operates_on([b])
    assert cirq.Moment([cirq.X(a), cirq.X(b)]).operates_on([a, b])
    assert not cirq.Moment([cirq.X(a), cirq.X(b)]).operates_on([c])
    assert cirq.Moment([cirq.X(a), cirq.X(b)]).operates_on([a, c])
    assert cirq.Moment([cirq.X(a), cirq.X(b)]).operates_on([a, b, c])


def test_operation_at():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')

    # No operation on that qubit
    assert cirq.Moment().operation_at(a) is None

    # One Operation on the quibt
    assert cirq.Moment([cirq.X(a)]).operation_at(a) == cirq.X(a)

    # Multiple Operations on the qubits
    assert cirq.Moment([cirq.CZ(a, b), cirq.X(c)]).operation_at(a) == cirq.CZ(a, b)


def test_from_ops():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    assert cirq.Moment.from_ops(cirq.X(a), cirq.Y(b)) == cirq.Moment(cirq.X(a), cirq.Y(b))


def test_with_operation():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    assert cirq.Moment().with_operation(cirq.X(a)) == cirq.Moment([cirq.X(a)])

    assert cirq.Moment([cirq.X(a)]).with_operation(cirq.X(b)) == cirq.Moment([cirq.X(a), cirq.X(b)])

    # One-qubit operation case.
    with pytest.raises(ValueError):
        _ = cirq.Moment([cirq.X(a)]).with_operation(cirq.X(a))

    # Two-qubit operation case.
    with pytest.raises(ValueError):
        _ = cirq.Moment([cirq.CZ(a, b)]).with_operation(cirq.X(a))
    with pytest.raises(ValueError):
        _ = cirq.Moment([cirq.CZ(a, b)]).with_operation(cirq.X(b))

    # Multiple operations case.
    with pytest.raises(ValueError):
        _ = cirq.Moment([cirq.X(a), cirq.X(b)]).with_operation(cirq.X(a))
    with pytest.raises(ValueError):
        _ = cirq.Moment([cirq.X(a), cirq.X(b)]).with_operation(cirq.X(b))


def test_with_operations():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')

    assert cirq.Moment().with_operations(cirq.X(a)) == cirq.Moment([cirq.X(a)])
    assert cirq.Moment().with_operations(cirq.X(a), cirq.X(b)) == cirq.Moment(
        [cirq.X(a), cirq.X(b)]
    )

    assert cirq.Moment([cirq.X(a)]).with_operations(cirq.X(b)) == cirq.Moment(
        [cirq.X(a), cirq.X(b)]
    )
    assert cirq.Moment([cirq.X(a)]).with_operations(cirq.X(b), cirq.X(c)) == cirq.Moment(
        [cirq.X(a), cirq.X(b), cirq.X(c)]
    )

    # One-qubit operation case.
    with pytest.raises(ValueError):
        _ = cirq.Moment([cirq.X(a)]).with_operations(cirq.X(a))

    # Two-qubit operation case.
    with pytest.raises(ValueError):
        _ = cirq.Moment([cirq.CZ(a, b)]).with_operations(cirq.X(a))
    with pytest.raises(ValueError):
        _ = cirq.Moment([cirq.CZ(a, b)]).with_operations(cirq.X(b))

    # Multiple operations case.
    with pytest.raises(ValueError):
        _ = cirq.Moment([cirq.X(a), cirq.X(b)]).with_operations(cirq.X(a))
    with pytest.raises(ValueError):
        _ = cirq.Moment([cirq.X(a), cirq.X(b)]).with_operations(cirq.X(b))


def test_without_operations_touching():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')

    # Empty case.
    assert cirq.Moment().without_operations_touching([]) == cirq.Moment()
    assert cirq.Moment().without_operations_touching([a]) == cirq.Moment()
    assert cirq.Moment().without_operations_touching([a, b]) == cirq.Moment()

    # One-qubit operation case.
    assert cirq.Moment([cirq.X(a)]).without_operations_touching([]) == cirq.Moment([cirq.X(a)])
    assert cirq.Moment([cirq.X(a)]).without_operations_touching([a]) == cirq.Moment()
    assert cirq.Moment([cirq.X(a)]).without_operations_touching([b]) == cirq.Moment([cirq.X(a)])

    # Two-qubit operation case.
    assert cirq.Moment([cirq.CZ(a, b)]).without_operations_touching([]) == cirq.Moment(
        [cirq.CZ(a, b)]
    )
    assert cirq.Moment([cirq.CZ(a, b)]).without_operations_touching([a]) == cirq.Moment()
    assert cirq.Moment([cirq.CZ(a, b)]).without_operations_touching([b]) == cirq.Moment()
    assert cirq.Moment([cirq.CZ(a, b)]).without_operations_touching([c]) == cirq.Moment(
        [cirq.CZ(a, b)]
    )

    # Multiple operation case.
    assert cirq.Moment([cirq.CZ(a, b), cirq.X(c)]).without_operations_touching([]) == cirq.Moment(
        [cirq.CZ(a, b), cirq.X(c)]
    )
    assert cirq.Moment([cirq.CZ(a, b), cirq.X(c)]).without_operations_touching([a]) == cirq.Moment(
        [cirq.X(c)]
    )
    assert cirq.Moment([cirq.CZ(a, b), cirq.X(c)]).without_operations_touching([b]) == cirq.Moment(
        [cirq.X(c)]
    )
    assert cirq.Moment([cirq.CZ(a, b), cirq.X(c)]).without_operations_touching([c]) == cirq.Moment(
        [cirq.CZ(a, b)]
    )
    assert cirq.Moment([cirq.CZ(a, b), cirq.X(c)]).without_operations_touching(
        [a, b]
    ) == cirq.Moment([cirq.X(c)])
    assert (
        cirq.Moment([cirq.CZ(a, b), cirq.X(c)]).without_operations_touching([a, c]) == cirq.Moment()
    )


def test_is_parameterized():
    a, b = cirq.LineQubit.range(2)
    moment = cirq.Moment(cirq.X(a) ** sympy.Symbol('v'), cirq.Y(b) ** sympy.Symbol('w'))
    assert cirq.is_parameterized(moment)
    assert not cirq.is_parameterized(cirq.Moment(cirq.X(a), cirq.Y(b)))


def test_resolve_parameters():
    a, b = cirq.LineQubit.range(2)
    moment = cirq.Moment(cirq.X(a) ** sympy.Symbol('v'), cirq.Y(b) ** sympy.Symbol('w'))
    resolved_moment = cirq.resolve_parameters(moment, cirq.ParamResolver({'v': 0.1, 'w': 0.2}))
    assert resolved_moment == cirq.Moment(cirq.X(a) ** 0.1, cirq.Y(b) ** 0.2)
    # sympy constant is resolved to a Python number
    moment = cirq.Moment(cirq.Rz(rads=sympy.pi).on(a))
    resolved_moment = cirq.resolve_parameters(moment, {'pi': np.pi})
    assert resolved_moment == cirq.Moment(cirq.Rz(rads=np.pi).on(a))
    resolved_gate = resolved_moment.operations[0].gate
    assert not isinstance(resolved_gate.exponent, sympy.Basic)
    assert isinstance(resolved_gate.exponent, float)
    assert not cirq.is_parameterized(resolved_moment)


def test_resolve_parameters_no_change():
    a, b = cirq.LineQubit.range(2)
    moment = cirq.Moment(cirq.X(a), cirq.Y(b))
    resolved_moment = cirq.resolve_parameters(moment, cirq.ParamResolver({'v': 0.1, 'w': 0.2}))
    assert resolved_moment is moment

    moment = cirq.Moment(cirq.X(a) ** sympy.Symbol('v'), cirq.Y(b) ** sympy.Symbol('w'))
    resolved_moment = cirq.resolve_parameters(moment, cirq.ParamResolver({}))
    assert resolved_moment is moment


def test_parameter_names():
    a, b = cirq.LineQubit.range(2)
    moment = cirq.Moment(cirq.X(a) ** sympy.Symbol('v'), cirq.Y(b) ** sympy.Symbol('w'))
    assert cirq.parameter_names(moment) == {'v', 'w'}
    assert cirq.parameter_names(cirq.Moment(cirq.X(a), cirq.Y(b))) == set()


def test_with_measurement_keys():
    a, b = cirq.LineQubit.range(2)
    m = cirq.Moment(cirq.measure(a, key='m1'), cirq.measure(b, key='m2'))

    new_moment = cirq.with_measurement_key_mapping(m, {'m1': 'p1', 'm2': 'p2', 'x': 'z'})

    assert new_moment.operations[0] == cirq.measure(a, key='p1')
    assert new_moment.operations[1] == cirq.measure(b, key='p2')


def test_with_key_path():
    a, b = cirq.LineQubit.range(2)
    m = cirq.Moment(cirq.measure(a, key='m1'), cirq.measure(b, key='m2'))

    new_moment = cirq.with_key_path(m, ('a', 'b'))

    assert new_moment.operations[0] == cirq.measure(
        a, key=cirq.MeasurementKey.parse_serialized('a:b:m1')
    )
    assert new_moment.operations[1] == cirq.measure(
        b, key=cirq.MeasurementKey.parse_serialized('a:b:m2')
    )


def test_with_key_path_prefix():
    a, b, c = cirq.LineQubit.range(3)
    m = cirq.Moment(cirq.measure(a, key='m1'), cirq.measure(b, key='m2'), cirq.X(c))
    mb = cirq.with_key_path_prefix(m, ('b',))
    mab = cirq.with_key_path_prefix(mb, ('a',))
    assert mab.operations[0] == cirq.measure(a, key=cirq.MeasurementKey.parse_serialized('a:b:m1'))
    assert mab.operations[1] == cirq.measure(b, key=cirq.MeasurementKey.parse_serialized('a:b:m2'))
    assert mab.operations[2] is m.operations[2]


def test_copy():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    original = cirq.Moment([cirq.CZ(a, b)])
    copy = original.__copy__()
    assert original == copy
    assert id(original) != id(copy)


def test_qubits():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    assert cirq.Moment([cirq.X(a), cirq.X(b)]).qubits == {a, b}
    assert cirq.Moment([cirq.X(a)]).qubits == {a}
    assert cirq.Moment([cirq.CZ(a, b)]).qubits == {a, b}


def test_container_methods():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    m = cirq.Moment([cirq.H(a), cirq.H(b)])
    assert list(m) == list(m.operations)
    # __iter__
    assert list(iter(m)) == list(m.operations)
    # __contains__ for free.
    assert cirq.H(b) in m

    assert len(m) == 2


def test_decompose():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    m = cirq.Moment(cirq.X(a), cirq.X(b))
    assert list(cirq.decompose(m)) == list(m.operations)


def test_measurement_keys():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    m = cirq.Moment(cirq.X(a), cirq.X(b))
    assert cirq.measurement_key_names(m) == set()
    assert not cirq.is_measurement(m)

    m2 = cirq.Moment(cirq.measure(a, b, key='foo'))
    assert cirq.measurement_key_objs(m2) == {cirq.MeasurementKey('foo')}
    assert cirq.measurement_key_names(m2) == {'foo'}
    assert cirq.is_measurement(m2)


def test_measurement_key_objs_caching():
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    m = cirq.Moment(cirq.measure(q0, key='foo'))
    assert m._measurement_key_objs is None
    key_objs = cirq.measurement_key_objs(m)
    assert m._measurement_key_objs == key_objs

    # Make sure it gets updated when adding an operation.
    m = m.with_operation(cirq.measure(q1, key='bar'))
    assert m._measurement_key_objs == {
        cirq.MeasurementKey(name='bar'),
        cirq.MeasurementKey(name='foo'),
    }
    # Or multiple operations.
    m = m.with_operations(cirq.measure(q2, key='doh'), cirq.measure(q3, key='baz'))
    assert m._measurement_key_objs == {
        cirq.MeasurementKey(name='bar'),
        cirq.MeasurementKey(name='foo'),
        cirq.MeasurementKey(name='doh'),
        cirq.MeasurementKey(name='baz'),
    }


def test_control_keys_caching():
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    m = cirq.Moment(cirq.X(q0).with_classical_controls('foo'))
    assert m._control_keys is None
    keys = cirq.control_keys(m)
    assert m._control_keys == keys

    # Make sure it gets updated when adding an operation.
    m = m.with_operation(cirq.X(q1).with_classical_controls('bar'))
    assert m._control_keys == {cirq.MeasurementKey(name='bar'), cirq.MeasurementKey(name='foo')}
    # Or multiple operations.
    m = m.with_operations(
        cirq.X(q2).with_classical_controls('doh'), cirq.X(q3).with_classical_controls('baz')
    )
    assert m._control_keys == {
        cirq.MeasurementKey(name='bar'),
        cirq.MeasurementKey(name='foo'),
        cirq.MeasurementKey(name='doh'),
        cirq.MeasurementKey(name='baz'),
    }


def test_bool():
    assert not cirq.Moment()
    a = cirq.NamedQubit('a')
    assert cirq.Moment([cirq.X(a)])


def test_repr():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    cirq.testing.assert_equivalent_repr(cirq.Moment())
    cirq.testing.assert_equivalent_repr(cirq.Moment(cirq.CZ(a, b)))
    cirq.testing.assert_equivalent_repr(cirq.Moment(cirq.X(a), cirq.Y(b)))


def test_json_dict():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    mom = cirq.Moment([cirq.CZ(a, b)])
    assert mom._json_dict_() == {'operations': (cirq.CZ(a, b),)}


def test_inverse():
    a, b, c = cirq.LineQubit.range(3)
    m = cirq.Moment([cirq.S(a), cirq.CNOT(b, c)])
    assert m**1 is m
    assert m**-1 == cirq.Moment([cirq.S(a) ** -1, cirq.CNOT(b, c)])
    assert m**0.5 == cirq.Moment([cirq.T(a), cirq.CNOT(b, c) ** 0.5])
    assert cirq.inverse(m) == m**-1
    assert cirq.inverse(cirq.inverse(m)) == m
    assert cirq.inverse(cirq.Moment([cirq.measure(a)]), default=None) is None


def test_immutable_moment():
    with pytest.raises(AttributeError):
        q1, q2 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(cirq.X(q1))
        moment = circuit.moments[0]
        moment.operations += (cirq.Y(q2),)


def test_add():
    a, b, c = cirq.LineQubit.range(3)
    expected_circuit = cirq.Circuit([cirq.CNOT(a, b), cirq.X(a), cirq.Y(b)])

    circuit1 = cirq.Circuit([cirq.CNOT(a, b), cirq.X(a)])
    circuit1[1] += cirq.Y(b)
    assert circuit1 == expected_circuit

    circuit2 = cirq.Circuit(cirq.CNOT(a, b), cirq.Y(b))
    circuit2[1] += cirq.X(a)
    assert circuit2 == expected_circuit

    m1 = cirq.Moment([cirq.X(a)])
    m2 = cirq.Moment([cirq.CNOT(a, b)])
    m3 = cirq.Moment([cirq.X(c)])
    assert m1 + m3 == cirq.Moment([cirq.X(a), cirq.X(c)])
    assert m2 + m3 == cirq.Moment([cirq.CNOT(a, b), cirq.X(c)])
    with pytest.raises(ValueError, match='Overlap'):
        _ = m1 + m2

    assert m1 + [[[[cirq.Y(b)]]]] == cirq.Moment(cirq.X(a), cirq.Y(b))
    assert m1 + [] == m1
    assert m1 + [] is m1


def test_sub():
    a, b, c = cirq.LineQubit.range(3)
    m = cirq.Moment(cirq.X(a), cirq.Y(b))
    assert m - [] == m
    assert m - cirq.X(a) == cirq.Moment(cirq.Y(b))
    assert m - [[[[cirq.X(a)]], []]] == cirq.Moment(cirq.Y(b))
    assert m - [cirq.X(a), cirq.Y(b)] == cirq.Moment()
    assert m - [cirq.Y(b)] == cirq.Moment(cirq.X(a))

    with pytest.raises(ValueError, match="missing operations"):
        _ = m - cirq.X(b)
    with pytest.raises(ValueError, match="missing operations"):
        _ = m - [cirq.X(a), cirq.Z(c)]

    # Preserves relative order.
    m2 = cirq.Moment(cirq.X(a), cirq.Y(b), cirq.Z(c))
    assert m2 - cirq.Y(b) == cirq.Moment(cirq.X(a), cirq.Z(c))


def test_op_tree():
    eq = cirq.testing.EqualsTester()
    a, b = cirq.LineQubit.range(2)

    eq.add_equality_group(cirq.Moment(), cirq.Moment([]), cirq.Moment([[], [[[]]]]))

    eq.add_equality_group(
        cirq.Moment(cirq.X(a)), cirq.Moment([cirq.X(a)]), cirq.Moment({cirq.X(a)})
    )

    eq.add_equality_group(cirq.Moment(cirq.X(a), cirq.Y(b)), cirq.Moment([cirq.X(a), cirq.Y(b)]))


def test_indexes_by_qubit():
    a, b, c = cirq.LineQubit.range(3)
    moment = cirq.Moment([cirq.H(a), cirq.CNOT(b, c)])

    assert moment[a] == cirq.H(a)
    assert moment[b] == cirq.CNOT(b, c)
    assert moment[c] == cirq.CNOT(b, c)


def test_throws_when_indexed_by_unused_qubit():
    a, b = cirq.LineQubit.range(2)
    moment = cirq.Moment([cirq.H(a)])

    with pytest.raises(KeyError, match="Moment doesn't act on given qubit"):
        _ = moment[b]


def test_indexes_by_list_of_qubits():
    q = cirq.LineQubit.range(4)
    moment = cirq.Moment([cirq.Z(q[0]), cirq.CNOT(q[1], q[2])])

    assert moment[[q[0]]] == cirq.Moment([cirq.Z(q[0])])
    assert moment[[q[1]]] == cirq.Moment([cirq.CNOT(q[1], q[2])])
    assert moment[[q[2]]] == cirq.Moment([cirq.CNOT(q[1], q[2])])
    assert moment[[q[3]]] == cirq.Moment([])
    assert moment[q[0:2]] == moment
    assert moment[q[1:3]] == cirq.Moment([cirq.CNOT(q[1], q[2])])
    assert moment[q[2:4]] == cirq.Moment([cirq.CNOT(q[1], q[2])])
    assert moment[[q[0], q[3]]] == cirq.Moment([cirq.Z(q[0])])
    assert moment[q] == moment


def test_moment_text_diagram():
    a, b, c, d = cirq.GridQubit.rect(2, 2)
    m = cirq.Moment(cirq.CZ(a, b), cirq.CNOT(c, d))
    assert (
        str(m).strip()
        == """
  ╷ 0 1
╶─┼─────
0 │ @─@
  │
1 │ @─X
  │
    """.strip()
    )

    m = cirq.Moment(cirq.CZ(a, b), cirq.CNOT(c, d))
    cirq.testing.assert_has_diagram(
        m,
        """
   ╷ None 0 1
╶──┼──────────
aa │
   │
0  │      @─@
   │
1  │      @─X
   │
        """,
        extra_qubits=[cirq.NamedQubit("aa")],
    )

    m = cirq.Moment(cirq.S(c), cirq.ISWAP(a, d))
    cirq.testing.assert_has_diagram(
        m,
        """
  ╷ 0     1
╶─┼─────────────
0 │ iSwap─┐
  │       │
1 │ S     iSwap
  │
    """,
    )

    m = cirq.Moment(cirq.S(c) ** 0.1, cirq.ISWAP(a, d) ** 0.5)
    cirq.testing.assert_has_diagram(
        m,
        """
  ╷ 0         1
╶─┼─────────────────
0 │ iSwap^0.5─┐
  │           │
1 │ Z^0.05    iSwap
  │
    """,
    )

    a, b, c = cirq.LineQubit.range(3)
    m = cirq.Moment(cirq.X(a), cirq.SWAP(b, c))
    cirq.testing.assert_has_diagram(
        m,
        """
  ╷ a b c
╶─┼───────
0 │ X
  │
1 │   ×─┐
  │     │
2 │     ×
  │
    """,
        xy_breakdown_func=lambda q: ('abc'[q.x], q.x),
    )

    class EmptyGate(cirq.testing.SingleQubitGate):
        def __str__(self):
            return 'Empty'

    m = cirq.Moment(EmptyGate().on(a))
    cirq.testing.assert_has_diagram(
        m,
        """
  ╷ 0
╶─┼───────
0 │ Empty
  │
    """,
    )


def test_text_diagram_does_not_depend_on_insertion_order():
    q = cirq.LineQubit.range(4)
    ops = [cirq.CNOT(q[0], q[3]), cirq.CNOT(q[1], q[2])]
    m1, m2 = cirq.Moment(ops), cirq.Moment(ops[::-1])
    assert m1 == m2
    assert str(m1) == str(m2)


def test_commutes_moment_and_operation():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    d = cirq.NamedQubit('d')

    moment = cirq.Moment([cirq.X(a), cirq.Y(b), cirq.H(c)])

    assert cirq.commutes(moment, a, default=None) is None

    assert cirq.commutes(moment, cirq.X(a))
    assert cirq.commutes(moment, cirq.Y(b))
    assert cirq.commutes(moment, cirq.H(c))
    assert cirq.commutes(moment, cirq.H(d))

    # X and H do not commute
    assert not cirq.commutes(moment, cirq.H(a))
    assert not cirq.commutes(moment, cirq.H(b))
    assert not cirq.commutes(moment, cirq.X(c))

    # Empty moment commutes with everything
    moment = cirq.Moment()
    assert cirq.commutes(moment, cirq.X(a))
    assert cirq.commutes(moment, cirq.measure(b))

    # Two qubit operation
    moment = cirq.Moment(cirq.Z(a), cirq.Z(b))
    assert cirq.commutes(moment, cirq.XX(a, b))


def test_commutes_moment_and_moment():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')

    # Test cases where individual operations don't commute but moments do
    # Two Z gates (Z⊗Z) commutes with RXX even though individual Z's don't
    assert not cirq.commutes(cirq.Moment(cirq.Z(a)), cirq.Moment(cirq.XX(a, b)))
    assert cirq.commutes(cirq.Moment(cirq.Z(a), cirq.Z(b)), cirq.Moment(cirq.XX(a, b)))

    # Moments that do not commute if acting on same qubits
    assert cirq.commutes(cirq.Moment(cirq.X(a)), cirq.Moment(cirq.Y(b)))
    assert not cirq.commutes(cirq.Moment(cirq.X(a)), cirq.Moment(cirq.Y(a)))

    # Moments commute with themselves
    assert cirq.commutes(
        cirq.Moment([cirq.X(a), cirq.Y(b), cirq.H(c)]),
        cirq.Moment([cirq.X(a), cirq.Y(b), cirq.H(c)]),
    )


def test_commutes_moment_with_controls():
    a, b = cirq.LineQubit.range(2)
    assert cirq.commutes(
        cirq.Moment(cirq.measure(a, key='k0')), cirq.Moment(cirq.X(b).with_classical_controls('k1'))
    )
    assert cirq.commutes(
        cirq.Moment(cirq.X(b).with_classical_controls('k1')), cirq.Moment(cirq.measure(a, key='k0'))
    )
    assert cirq.commutes(
        cirq.Moment(cirq.X(a).with_classical_controls('k0')),
        cirq.Moment(cirq.H(b).with_classical_controls('k0')),
    )
    assert cirq.commutes(
        cirq.Moment(cirq.X(a).with_classical_controls('k0')),
        cirq.Moment(cirq.X(a).with_classical_controls('k0')),
    )
    assert not cirq.commutes(
        cirq.Moment(cirq.measure(a, key='k0')), cirq.Moment(cirq.X(b).with_classical_controls('k0'))
    )
    assert not cirq.commutes(
        cirq.Moment(cirq.X(b).with_classical_controls('k0')), cirq.Moment(cirq.measure(a, key='k0'))
    )
    assert not cirq.commutes(
        cirq.Moment(cirq.X(a).with_classical_controls('k0')),
        cirq.Moment(cirq.H(a).with_classical_controls('k0')),
    )


def test_commutes_moment_and_moment_comprehensive():
    a, b, c, d = cirq.LineQubit.range(4)

    # Basic Z⊗Z commuting with XX at different angles
    m1 = cirq.Moment([cirq.Z(a), cirq.Z(b)])
    m2 = cirq.Moment([cirq.XXPowGate(exponent=0.5)(a, b)])
    assert cirq.commutes(m1, m2)

    # Disjoint qubit sets
    m1 = cirq.Moment([cirq.X(a), cirq.Y(b)])
    m2 = cirq.Moment([cirq.Z(c), cirq.H(d)])
    assert cirq.commutes(m1, m2)

    # Mixed case - some commute individually, some as group
    m1 = cirq.Moment([cirq.Z(a), cirq.Z(b), cirq.X(c)])
    m2 = cirq.Moment([cirq.XXPowGate(exponent=0.5)(a, b), cirq.X(c)])
    assert cirq.commutes(m1, m2)

    # Non-commuting case: X on first qubit, Z on second with XX gate
    m1 = cirq.Moment([cirq.X(a), cirq.Z(b)])
    m2 = cirq.Moment([cirq.XX(a, b)])
    assert not cirq.commutes(m1, m2)

    # Complex case requiring unitary calculation - non-commuting case
    m1 = cirq.Moment([cirq.Z(a), cirq.Z(b), cirq.Z(c)])
    m2 = cirq.Moment([cirq.XXPowGate(exponent=0.5)(a, b), cirq.X(c)])
    assert not cirq.commutes(m1, m2)  # Z⊗Z⊗Z doesn't commute with XX⊗X


def test_commutes_handles_non_unitary_operation():
    a = cirq.NamedQubit('a')
    op_damp_a = cirq.AmplitudeDampingChannel(gamma=0.1).on(a)
    assert cirq.commutes(cirq.Moment(cirq.X(a)), op_damp_a, default=None) is None
    assert cirq.commutes(cirq.Moment(cirq.X(a)), cirq.Moment(op_damp_a), default=None) is None
    assert cirq.commutes(cirq.Moment(op_damp_a), cirq.Moment(op_damp_a))


def test_transform_qubits():
    a, b = cirq.LineQubit.range(2)
    x, y = cirq.GridQubit.rect(2, 1, 10, 20)

    original = cirq.Moment([cirq.X(a), cirq.Y(b)])
    modified = cirq.Moment([cirq.X(x), cirq.Y(y)])

    assert original.transform_qubits({a: x, b: y}) == modified
    assert original.transform_qubits(lambda q: cirq.GridQubit(10 + q.x, 20)) == modified
    with pytest.raises(TypeError, match='must be a function or dict'):
        _ = original.transform_qubits('bad arg')


def test_expand_to():
    a, b = cirq.LineQubit.range(2)
    m1 = cirq.Moment(cirq.H(a))
    m2 = m1.expand_to({a})
    assert m1 == m2

    m3 = m1.expand_to({a, b})
    assert m1 != m3
    assert m3.qubits == {a, b}
    assert m3.operations == (cirq.H(a), cirq.I(b))

    with pytest.raises(ValueError, match='superset'):
        _ = m1.expand_to({b})


def test_kraus():
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.diag([1, -1])

    a, b = cirq.LineQubit.range(2)

    m = cirq.Moment()
    assert cirq.has_kraus(m)
    k = cirq.kraus(m)
    assert len(k) == 1
    assert np.allclose(k[0], np.array([[1.0]]))

    m = cirq.Moment(cirq.S(a))
    assert cirq.has_kraus(m)
    k = cirq.kraus(m)
    assert len(k) == 1
    assert np.allclose(k[0], np.diag([1, 1j]))

    m = cirq.Moment(cirq.CNOT(a, b))
    assert cirq.has_kraus(m)
    k = cirq.kraus(m)
    assert len(k) == 1
    assert np.allclose(k[0], np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]))

    p = 0.1
    m = cirq.Moment(cirq.depolarize(p).on(a))
    assert cirq.has_kraus(m)
    k = cirq.kraus(m)
    assert len(k) == 4
    assert np.allclose(k[0], np.sqrt(1 - p) * I)
    assert np.allclose(k[1], np.sqrt(p / 3) * X)
    assert np.allclose(k[2], np.sqrt(p / 3) * Y)
    assert np.allclose(k[3], np.sqrt(p / 3) * Z)

    p = 0.2
    q = 0.3
    m = cirq.Moment(cirq.bit_flip(p).on(a), cirq.phase_flip(q).on(b))
    assert cirq.has_kraus(m)
    k = cirq.kraus(m)
    assert len(k) == 4
    assert np.allclose(k[0], np.sqrt((1 - p) * (1 - q)) * np.kron(I, I))
    assert np.allclose(k[1], np.sqrt(q * (1 - p)) * np.kron(I, Z))
    assert np.allclose(k[2], np.sqrt(p * (1 - q)) * np.kron(X, I))
    assert np.allclose(k[3], np.sqrt(p * q) * np.kron(X, Z))


def test_kraus_too_big():
    m = cirq.Moment(cirq.IdentityGate(11).on(*cirq.LineQubit.range(11)))
    assert not cirq.has_kraus(m)
    assert not m._has_superoperator_()
    assert m._kraus_() is NotImplemented
    assert m._superoperator_() is NotImplemented
    assert cirq.kraus(m, default=None) is None


def test_op_has_no_kraus():
    class EmptyGate(cirq.testing.SingleQubitGate):
        pass

    m = cirq.Moment(EmptyGate().on(cirq.NamedQubit("a")))
    assert not cirq.has_kraus(m)
    assert not m._has_superoperator_()
    assert m._kraus_() is NotImplemented
    assert m._superoperator_() is NotImplemented
    assert cirq.kraus(m, default=None) is None


def test_superoperator():
    cnot = cirq.unitary(cirq.CNOT)

    a, b = cirq.LineQubit.range(2)

    m = cirq.Moment()
    assert m._has_superoperator_()
    s = m._superoperator_()
    assert np.allclose(s, np.array([[1.0]]))

    m = cirq.Moment(cirq.I(a))
    assert m._has_superoperator_()
    s = m._superoperator_()
    assert np.allclose(s, np.eye(4))

    m = cirq.Moment(cirq.IdentityGate(2).on(a, b))
    assert m._has_superoperator_()
    s = m._superoperator_()
    assert np.allclose(s, np.eye(16))

    m = cirq.Moment(cirq.S(a))
    assert m._has_superoperator_()
    s = m._superoperator_()
    assert np.allclose(s, np.diag([1, -1j, 1j, 1]))

    m = cirq.Moment(cirq.CNOT(a, b))
    assert m._has_superoperator_()
    s = m._superoperator_()
    assert np.allclose(s, np.kron(cnot, cnot))

    m = cirq.Moment(cirq.depolarize(0.75).on(a))
    assert m._has_superoperator_()
    s = m._superoperator_()
    assert np.allclose(s, np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]) / 2)
