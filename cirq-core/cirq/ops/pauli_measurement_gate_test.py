# Copyright 2021 The Cirq Developers
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

import cirq


@pytest.mark.parametrize(
    'key',
    [
        'q0_1_0',
        cirq.MeasurementKey(name='q0_1_0'),
        cirq.MeasurementKey(path=('a', 'b'), name='c'),
    ],
)
def test_eval_repr(key):
    # Basic safeguard against repr-inequality.
    op = cirq.GateOperation(
        gate=cirq.PauliMeasurementGate([cirq.X, cirq.Y], key),
        qubits=[cirq.GridQubit(0, 1), cirq.GridQubit(1, 1)],
    )
    cirq.testing.assert_equivalent_repr(op)
    assert cirq.is_measurement(op)
    assert cirq.measurement_key_name(op) == str(key)


@pytest.mark.parametrize('observable', [[cirq.X], [cirq.Y, cirq.Z], cirq.DensePauliString('XYZ')])
@pytest.mark.parametrize('key', ['a', cirq.MeasurementKey('a')])
def test_init(observable, key):
    g = cirq.PauliMeasurementGate(observable, key)
    assert g.num_qubits() == len(observable)
    assert g.key == 'a'
    assert g.mkey == cirq.MeasurementKey('a')
    assert g._observable == tuple(observable)
    assert cirq.qid_shape(g) == (2,) * len(observable)


def test_measurement_eq():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(
        lambda: cirq.PauliMeasurementGate([cirq.X, cirq.Y], 'a'),
        lambda: cirq.PauliMeasurementGate([cirq.X, cirq.Y], cirq.MeasurementKey('a')),
    )
    eq.add_equality_group(cirq.PauliMeasurementGate([cirq.X, cirq.Y], 'b'))
    eq.add_equality_group(cirq.PauliMeasurementGate([cirq.Y, cirq.X], 'a'))


@pytest.mark.parametrize(
    'protocol,args,key',
    [
        (None, None, 'b'),
        (cirq.with_key_path, ('p', 'q'), 'p:q:a'),
        (cirq.with_measurement_key_mapping, {'a': 'b'}, 'b'),
    ],
)
@pytest.mark.parametrize(
    'gate',
    [
        cirq.PauliMeasurementGate([cirq.X], 'a'),
        cirq.PauliMeasurementGate([cirq.X, cirq.Y, cirq.Z], 'a'),
    ],
)
def test_measurement_with_key(protocol, args, key, gate):
    if protocol:
        gate_with_key = protocol(gate, args)
    else:
        gate_with_key = gate.with_key('b')
    assert gate_with_key.key == key
    assert gate_with_key.num_qubits() == gate.num_qubits()
    assert gate_with_key.observable() == gate.observable()
    assert cirq.qid_shape(gate_with_key) == cirq.qid_shape(gate)
    if protocol:
        same_gate = cirq.with_measurement_key_mapping(gate, {'a': 'a'})
    else:
        same_gate = gate.with_key('a')
    assert same_gate == gate


def test_measurement_gate_diagram():
    # Shows observable & key.
    assert cirq.circuit_diagram_info(
        cirq.PauliMeasurementGate([cirq.X], key='test')
    ) == cirq.CircuitDiagramInfo(("M(X)('test')",))

    # Shows multiple observables.
    assert cirq.circuit_diagram_info(
        cirq.PauliMeasurementGate([cirq.X, cirq.Y, cirq.Z], 'a')
    ) == cirq.CircuitDiagramInfo(("M(X)('a')", 'M(Y)', 'M(Z)'))

    # Omits key when it is the default.
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    cirq.testing.assert_has_diagram(
        cirq.Circuit(cirq.measure_single_paulistring(cirq.X(a) * cirq.Y(b))),
        """
a: ───M(X)───
      │
b: ───M(Y)───
""",
    )
    cirq.testing.assert_has_diagram(
        cirq.Circuit(cirq.measure_single_paulistring(cirq.X(a) * cirq.Y(b), key='test')),
        """
a: ───M(X)('test')───
      │
b: ───M(Y)───────────
""",
    )


@pytest.mark.parametrize('observable', [[cirq.X], [cirq.X, cirq.Y, cirq.Z]])
@pytest.mark.parametrize(
    'key',
    [
        'q0_1_0',
        cirq.MeasurementKey(name='q0_1_0'),
        cirq.MeasurementKey(path=('a', 'b'), name='c'),
    ],
)
def test_consistent_protocols(observable, key):
    gate = cirq.PauliMeasurementGate(observable, key=key)
    cirq.testing.assert_implements_consistent_protocols(gate)
    assert cirq.is_measurement(gate)
    assert cirq.measurement_key_name(gate) == str(key)


def test_op_repr():
    a, b, c = cirq.LineQubit.range(3)
    ps = cirq.X(a) * cirq.Y(b) * cirq.Z(c)
    assert (
        repr(cirq.measure_single_paulistring(ps))
        == 'cirq.measure_single_paulistring((cirq.X(cirq.LineQubit(0))'
        '*cirq.Y(cirq.LineQubit(1))*cirq.Z(cirq.LineQubit(2))))'
    )
    assert (
        repr(cirq.measure_single_paulistring(ps, key='out'))
        == "cirq.measure_single_paulistring((cirq.X(cirq.LineQubit(0))"
        "*cirq.Y(cirq.LineQubit(1))*cirq.Z(cirq.LineQubit(2))), "
        "key=cirq.MeasurementKey(name='out'))"
    )


def test_bad_observable_raises():
    with pytest.raises(ValueError, match='Pauli observable .* is empty'):
        _ = cirq.PauliMeasurementGate([])

    with pytest.raises(ValueError, match=r'Pauli observable .* must be Iterable\[`cirq.Pauli`\]'):
        _ = cirq.PauliMeasurementGate([cirq.I, cirq.X, cirq.Y])

    with pytest.raises(ValueError, match=r'Pauli observable .* must be Iterable\[`cirq.Pauli`\]'):
        _ = cirq.PauliMeasurementGate(cirq.DensePauliString('XYZI'))


def test_with_observable():
    o1 = [cirq.Z, cirq.Y, cirq.X]
    o2 = [cirq.X, cirq.Y, cirq.Z]
    g1 = cirq.PauliMeasurementGate(o1, key='a')
    g2 = cirq.PauliMeasurementGate(o2, key='a')
    assert g1.with_observable(o2) == g2
    assert g1.with_observable(o1) is g1
