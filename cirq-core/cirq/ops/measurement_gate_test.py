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

from typing import cast

import numpy as np
import pytest

import cirq


@pytest.mark.parametrize(
    'key',
    ['q0_1_0', cirq.MeasurementKey(name='q0_1_0'), cirq.MeasurementKey(path=('a', 'b'), name='c')],
)
def test_eval_repr(key):
    # Basic safeguard against repr-inequality.
    op = cirq.GateOperation(gate=cirq.MeasurementGate(1, key), qubits=[cirq.GridQubit(0, 1)])
    cirq.testing.assert_equivalent_repr(op)


@pytest.mark.parametrize('num_qubits', [1, 2, 4])
def test_measure_init(num_qubits):
    assert cirq.MeasurementGate(num_qubits, 'a').num_qubits() == num_qubits
    assert cirq.MeasurementGate(num_qubits, key='a').key == 'a'
    assert cirq.MeasurementGate(num_qubits, key='a').mkey == cirq.MeasurementKey('a')
    assert cirq.MeasurementGate(num_qubits, key=cirq.MeasurementKey('a')).key == 'a'
    assert cirq.MeasurementGate(num_qubits, key=cirq.MeasurementKey('a')) == cirq.MeasurementGate(
        num_qubits, key='a'
    )
    assert cirq.MeasurementGate(num_qubits, 'a', invert_mask=(True,)).invert_mask == (True,)
    assert cirq.qid_shape(cirq.MeasurementGate(num_qubits, 'a')) == (2,) * num_qubits
    cmap = {(0,): np.array([[0, 1], [1, 0]])}
    assert cirq.MeasurementGate(num_qubits, 'a', confusion_map=cmap).confusion_map == cmap


def test_measure_init_num_qubit_agnostic():
    assert cirq.qid_shape(cirq.MeasurementGate(3, 'a', qid_shape=(1, 2, 3))) == (1, 2, 3)
    assert cirq.qid_shape(cirq.MeasurementGate(key='a', qid_shape=(1, 2, 3))) == (1, 2, 3)
    with pytest.raises(ValueError, match='len.* >'):
        cirq.MeasurementGate(5, 'a', invert_mask=(True,) * 6)
    with pytest.raises(ValueError, match='len.* !='):
        cirq.MeasurementGate(5, 'a', qid_shape=(1, 2))
    with pytest.raises(ValueError, match='valid string'):
        cirq.MeasurementGate(2, qid_shape=(1, 2), key=None)
    with pytest.raises(ValueError, match='Confusion matrices have index out of bounds'):
        cirq.MeasurementGate(1, 'a', confusion_map={(1,): np.array([[0, 1], [1, 0]])})
    with pytest.raises(ValueError, match='Specify either'):
        cirq.MeasurementGate()


def test_measurement_has_unitary_returns_false():
    gate = cirq.MeasurementGate(1, 'a')
    assert not cirq.has_unitary(gate)


@pytest.mark.parametrize('num_qubits', [1, 2, 4])
def test_has_stabilizer_effect(num_qubits):
    assert cirq.has_stabilizer_effect(cirq.MeasurementGate(num_qubits, 'a'))


def test_measurement_eq():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(
        lambda: cirq.MeasurementGate(1, 'a'),
        lambda: cirq.MeasurementGate(1, 'a', invert_mask=()),
        lambda: cirq.MeasurementGate(1, 'a', invert_mask=(False,)),
        lambda: cirq.MeasurementGate(1, 'a', qid_shape=(2,)),
        lambda: cirq.MeasurementGate(1, 'a', confusion_map={}),
    )
    eq.add_equality_group(cirq.MeasurementGate(1, 'a', invert_mask=(True,)))
    eq.add_equality_group(
        cirq.MeasurementGate(1, 'a', confusion_map={(0,): np.array([[0, 1], [1, 0]])})
    )
    eq.add_equality_group(cirq.MeasurementGate(1, 'b'))
    eq.add_equality_group(cirq.MeasurementGate(2, 'a'))
    eq.add_equality_group(
        cirq.MeasurementGate(3, 'a'), cirq.MeasurementGate(3, 'a', qid_shape=(2, 2, 2))
    )
    eq.add_equality_group(cirq.MeasurementGate(3, 'a', qid_shape=(1, 2, 3)))


def test_measurement_full_invert_mask():
    assert cirq.MeasurementGate(1, 'a').full_invert_mask() == (False,)
    assert cirq.MeasurementGate(2, 'a', invert_mask=(False, True)).full_invert_mask() == (
        False,
        True,
    )
    assert cirq.MeasurementGate(2, 'a', invert_mask=(True,)).full_invert_mask() == (True, False)


@pytest.mark.parametrize('use_protocol', [False, True])
@pytest.mark.parametrize(
    'gate',
    [
        cirq.MeasurementGate(1, 'a'),
        cirq.MeasurementGate(1, 'a', invert_mask=(True,)),
        cirq.MeasurementGate(1, 'a', qid_shape=(3,)),
        cirq.MeasurementGate(2, 'a', invert_mask=(True, False), qid_shape=(2, 3)),
    ],
)
def test_measurement_with_key(use_protocol, gate):
    if use_protocol:
        gate1 = cirq.with_measurement_key_mapping(gate, {'a': 'b'})
    else:
        gate1 = gate.with_key('b')
    assert gate1.key == 'b'
    assert gate1.num_qubits() == gate.num_qubits()
    assert gate1.invert_mask == gate.invert_mask
    assert cirq.qid_shape(gate1) == cirq.qid_shape(gate)
    if use_protocol:
        gate2 = cirq.with_measurement_key_mapping(gate, {'a': 'a'})
    else:
        gate2 = gate.with_key('a')
    assert gate2 == gate


@pytest.mark.parametrize(
    'num_qubits, mask, bits, flipped',
    [
        (1, (), [0], (True,)),
        (3, (False,), [1], (False, True)),
        (3, (False, False), [0, 2], (True, False, True)),
    ],
)
def test_measurement_with_bits_flipped(num_qubits, mask, bits, flipped):
    gate = cirq.MeasurementGate(num_qubits, key='a', invert_mask=mask, qid_shape=(3,) * num_qubits)

    gate1 = gate.with_bits_flipped(*bits)
    assert gate1.key == gate.key
    assert gate1.num_qubits() == gate.num_qubits()
    assert gate1.invert_mask == flipped
    assert cirq.qid_shape(gate1) == cirq.qid_shape(gate)

    # Flipping bits again restores the mask (but may have extended it).
    gate2 = gate1.with_bits_flipped(*bits)
    assert gate2.full_invert_mask() == gate.full_invert_mask()


def test_qudit_measure_qasm():
    assert (
        cirq.qasm(
            cirq.measure(cirq.LineQid(0, 3), key='a'),
            args=cirq.QasmArgs(),
            default='not implemented',
        )
        == 'not implemented'
    )


def test_confused_measure_qasm():
    q0 = cirq.LineQubit(0)
    assert (
        cirq.qasm(
            cirq.measure(q0, key='a', confusion_map={(0,): np.array([[0, 1], [1, 0]])}),
            args=cirq.QasmArgs(),
            default='not implemented',
        )
        == 'not implemented'
    )


def test_measurement_gate_diagram():
    # Shows key.
    assert cirq.circuit_diagram_info(
        cirq.MeasurementGate(1, key='test')
    ) == cirq.CircuitDiagramInfo(("M('test')",))

    # Uses known qubit count.
    assert cirq.circuit_diagram_info(
        cirq.MeasurementGate(3, 'a'),
        cirq.CircuitDiagramInfoArgs(
            known_qubits=None,
            known_qubit_count=3,
            use_unicode_characters=True,
            precision=None,
            label_map=None,
        ),
    ) == cirq.CircuitDiagramInfo(("M('a')", 'M', 'M'))

    # Shows invert mask.
    assert cirq.circuit_diagram_info(
        cirq.MeasurementGate(2, 'a', invert_mask=(False, True))
    ) == cirq.CircuitDiagramInfo(("M('a')", "!M"))

    # Omits key when it is the default.
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    cirq.testing.assert_has_diagram(
        cirq.Circuit(cirq.measure(a, b)),
        """
a: ───M───
      │
b: ───M───
""",
    )
    cirq.testing.assert_has_diagram(
        cirq.Circuit(cirq.measure(a, b, invert_mask=(True,))),
        """
a: ───!M───
      │
b: ───M────
""",
    )
    cirq.testing.assert_has_diagram(
        cirq.Circuit(cirq.measure(a, b, confusion_map={(1,): np.array([[0, 1], [1, 0]])})),
        """
a: ───M────
      │
b: ───?M───
""",
    )
    cirq.testing.assert_has_diagram(
        cirq.Circuit(
            cirq.measure(
                a, b, invert_mask=(False, True), confusion_map={(1,): np.array([[0, 1], [1, 0]])}
            )
        ),
        """
a: ───M─────
      │
b: ───!?M───
""",
    )
    cirq.testing.assert_has_diagram(
        cirq.Circuit(cirq.measure(a, b, key='test')),
        """
a: ───M('test')───
      │
b: ───M───────────
""",
    )


def test_measurement_channel():
    np.testing.assert_allclose(
        cirq.kraus(cirq.MeasurementGate(1, 'a')),
        (np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])),
    )
    cirq.testing.assert_consistent_channel(cirq.MeasurementGate(1, 'a'))
    assert not cirq.has_mixture(cirq.MeasurementGate(1, 'a'))
    # yapf: disable
    np.testing.assert_allclose(
            cirq.kraus(cirq.MeasurementGate(2, 'a')),
            (np.array([[1, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]]),
             np.array([[0, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]]),
             np.array([[0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 0]]),
             np.array([[0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 1]])))
    np.testing.assert_allclose(
            cirq.kraus(cirq.MeasurementGate(2, 'a', qid_shape=(2, 3))),
            (np.diag([1, 0, 0, 0, 0, 0]),
             np.diag([0, 1, 0, 0, 0, 0]),
             np.diag([0, 0, 1, 0, 0, 0]),
             np.diag([0, 0, 0, 1, 0, 0]),
             np.diag([0, 0, 0, 0, 1, 0]),
             np.diag([0, 0, 0, 0, 0, 1])))
    # yapf: enable


def test_measurement_qubit_count_vs_mask_length():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')

    _ = cirq.MeasurementGate(num_qubits=1, key='a', invert_mask=(True,)).on(a)
    _ = cirq.MeasurementGate(num_qubits=2, key='a', invert_mask=(True, False)).on(a, b)
    _ = cirq.MeasurementGate(num_qubits=3, key='a', invert_mask=(True, False, True)).on(a, b, c)
    with pytest.raises(ValueError):
        _ = cirq.MeasurementGate(num_qubits=1, key='a', invert_mask=(True, False)).on(a)
    with pytest.raises(ValueError):
        _ = cirq.MeasurementGate(num_qubits=3, key='a', invert_mask=(True, False, True)).on(a, b)


def test_consistent_protocols():
    for n in range(1, 5):
        gate = cirq.MeasurementGate(num_qubits=n, key='a')
        cirq.testing.assert_implements_consistent_protocols(gate)

        gate = cirq.MeasurementGate(num_qubits=n, key='a', qid_shape=(3,) * n)
        cirq.testing.assert_implements_consistent_protocols(gate)


def test_op_repr():
    a, b = cirq.LineQubit.range(2)
    assert repr(cirq.measure(a)) == 'cirq.measure(cirq.LineQubit(0))'
    assert repr(cirq.measure(a, b)) == ('cirq.measure(cirq.LineQubit(0), cirq.LineQubit(1))')
    assert repr(cirq.measure(a, b, key='out', invert_mask=(False, True))) == (
        "cirq.measure(cirq.LineQubit(0), cirq.LineQubit(1), "
        "key=cirq.MeasurementKey(name='out'), "
        "invert_mask=(False, True))"
    )
    assert repr(
        cirq.measure(
            a,
            b,
            key='out',
            invert_mask=(False, True),
            confusion_map={(0,): np.array([[0, 1], [1, 0]], dtype=np.dtype('int64'))},
        )
    ) == (
        "cirq.measure(cirq.LineQubit(0), cirq.LineQubit(1), "
        "key=cirq.MeasurementKey(name='out'), "
        "invert_mask=(False, True), "
        "confusion_map={(0,): np.array([[0, 1], [1, 0]], dtype=np.dtype('int64'))})"
    )


def test_repr():
    gate = cirq.MeasurementGate(
        3,
        'a',
        (True, False),
        (1, 2, 3),
        {(2,): np.array([[0, 1], [1, 0]], dtype=np.dtype('int64'))},
    )
    assert repr(gate) == (
        "cirq.MeasurementGate(3, cirq.MeasurementKey(name='a'), (True, False), "
        "qid_shape=(1, 2, 3), "
        "confusion_map={(2,): np.array([[0, 1], [1, 0]], dtype=np.dtype('int64'))})"
    )


def test_act_on_state_vector():
    a, b = [cirq.LineQubit(3), cirq.LineQubit(1)]
    m = cirq.measure(
        a, b, key='out', invert_mask=(True,), confusion_map={(1,): np.array([[0, 1], [1, 0]])}
    )

    args = cirq.StateVectorSimulationState(
        available_buffer=np.empty(shape=(2, 2, 2, 2, 2)),
        qubits=cirq.LineQubit.range(5),
        prng=np.random.RandomState(),
        initial_state=cirq.one_hot(shape=(2, 2, 2, 2, 2), dtype=np.complex64),
        dtype=np.complex64,
    )
    cirq.act_on(m, args)
    assert args.log_of_measurement_results == {'out': [1, 1]}

    args = cirq.StateVectorSimulationState(
        available_buffer=np.empty(shape=(2, 2, 2, 2, 2)),
        qubits=cirq.LineQubit.range(5),
        prng=np.random.RandomState(),
        initial_state=cirq.one_hot(
            index=(0, 1, 0, 0, 0), shape=(2, 2, 2, 2, 2), dtype=np.complex64
        ),
        dtype=np.complex64,
    )
    cirq.act_on(m, args)
    assert args.log_of_measurement_results == {'out': [1, 0]}

    args = cirq.StateVectorSimulationState(
        available_buffer=np.empty(shape=(2, 2, 2, 2, 2)),
        qubits=cirq.LineQubit.range(5),
        prng=np.random.RandomState(),
        initial_state=cirq.one_hot(
            index=(0, 1, 0, 1, 0), shape=(2, 2, 2, 2, 2), dtype=np.complex64
        ),
        dtype=np.complex64,
    )
    cirq.act_on(m, args)
    datastore = cast(cirq.ClassicalDataDictionaryStore, args.classical_data)
    out = cirq.MeasurementKey('out')
    assert args.log_of_measurement_results == {'out': [0, 0]}
    assert datastore.records[out] == [(0, 0)]
    cirq.act_on(m, args)
    assert args.log_of_measurement_results == {'out': [0, 0]}
    assert datastore.records[out] == [(0, 0), (0, 0)]


def test_act_on_clifford_tableau():
    a, b = [cirq.LineQubit(3), cirq.LineQubit(1)]
    m = cirq.measure(
        a, b, key='out', invert_mask=(True,), confusion_map={(1,): np.array([[0, 1], [1, 0]])}
    )
    # The below assertion does not fail since it ignores non-unitary operations
    cirq.testing.assert_all_implemented_act_on_effects_match_unitary(m)

    args = cirq.CliffordTableauSimulationState(
        tableau=cirq.CliffordTableau(num_qubits=5, initial_state=0),
        qubits=cirq.LineQubit.range(5),
        prng=np.random.RandomState(),
    )
    cirq.act_on(m, args)
    assert args.log_of_measurement_results == {'out': [1, 1]}

    args = cirq.CliffordTableauSimulationState(
        tableau=cirq.CliffordTableau(num_qubits=5, initial_state=8),
        qubits=cirq.LineQubit.range(5),
        prng=np.random.RandomState(),
    )

    cirq.act_on(m, args)
    assert args.log_of_measurement_results == {'out': [1, 0]}

    args = cirq.CliffordTableauSimulationState(
        tableau=cirq.CliffordTableau(num_qubits=5, initial_state=10),
        qubits=cirq.LineQubit.range(5),
        prng=np.random.RandomState(),
    )
    cirq.act_on(m, args)
    datastore = cast(cirq.ClassicalDataDictionaryStore, args.classical_data)
    out = cirq.MeasurementKey('out')
    assert args.log_of_measurement_results == {'out': [0, 0]}
    assert datastore.records[out] == [(0, 0)]
    cirq.act_on(m, args)
    assert args.log_of_measurement_results == {'out': [0, 0]}
    assert datastore.records[out] == [(0, 0), (0, 0)]


def test_act_on_stabilizer_ch_form():
    a, b = [cirq.LineQubit(3), cirq.LineQubit(1)]
    m = cirq.measure(
        a, b, key='out', invert_mask=(True,), confusion_map={(1,): np.array([[0, 1], [1, 0]])}
    )
    # The below assertion does not fail since it ignores non-unitary operations
    cirq.testing.assert_all_implemented_act_on_effects_match_unitary(m)

    args = cirq.StabilizerChFormSimulationState(
        qubits=cirq.LineQubit.range(5), prng=np.random.RandomState(), initial_state=0
    )
    cirq.act_on(m, args)
    assert args.log_of_measurement_results == {'out': [1, 1]}

    args = cirq.StabilizerChFormSimulationState(
        qubits=cirq.LineQubit.range(5), prng=np.random.RandomState(), initial_state=8
    )

    cirq.act_on(m, args)
    assert args.log_of_measurement_results == {'out': [1, 0]}

    args = cirq.StabilizerChFormSimulationState(
        qubits=cirq.LineQubit.range(5), prng=np.random.RandomState(), initial_state=10
    )
    cirq.act_on(m, args)
    datastore = cast(cirq.ClassicalDataDictionaryStore, args.classical_data)
    out = cirq.MeasurementKey('out')
    assert args.log_of_measurement_results == {'out': [0, 0]}
    assert datastore.records[out] == [(0, 0)]
    cirq.act_on(m, args)
    assert args.log_of_measurement_results == {'out': [0, 0]}
    assert datastore.records[out] == [(0, 0), (0, 0)]


def test_act_on_qutrit():
    a, b = [cirq.LineQid(3, dimension=3), cirq.LineQid(1, dimension=3)]
    m = cirq.measure(
        a,
        b,
        key='out',
        invert_mask=(True,),
        confusion_map={(1,): np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])},
    )

    args = cirq.StateVectorSimulationState(
        available_buffer=np.empty(shape=(3, 3, 3, 3, 3)),
        qubits=cirq.LineQid.range(5, dimension=3),
        prng=np.random.RandomState(),
        initial_state=cirq.one_hot(
            index=(0, 2, 0, 2, 0), shape=(3, 3, 3, 3, 3), dtype=np.complex64
        ),
        dtype=np.complex64,
    )
    cirq.act_on(m, args)
    assert args.log_of_measurement_results == {'out': [2, 0]}

    args = cirq.StateVectorSimulationState(
        available_buffer=np.empty(shape=(3, 3, 3, 3, 3)),
        qubits=cirq.LineQid.range(5, dimension=3),
        prng=np.random.RandomState(),
        initial_state=cirq.one_hot(
            index=(0, 1, 0, 2, 0), shape=(3, 3, 3, 3, 3), dtype=np.complex64
        ),
        dtype=np.complex64,
    )
    cirq.act_on(m, args)
    assert args.log_of_measurement_results == {'out': [2, 2]}

    args = cirq.StateVectorSimulationState(
        available_buffer=np.empty(shape=(3, 3, 3, 3, 3)),
        qubits=cirq.LineQid.range(5, dimension=3),
        prng=np.random.RandomState(),
        initial_state=cirq.one_hot(
            index=(0, 2, 0, 1, 0), shape=(3, 3, 3, 3, 3), dtype=np.complex64
        ),
        dtype=np.complex64,
    )
    cirq.act_on(m, args)
    assert args.log_of_measurement_results == {'out': [0, 0]}
