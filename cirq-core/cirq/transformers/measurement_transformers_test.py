# Copyright 2022 The Cirq Developers
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
from sympy.parsing import sympy_parser

import cirq
from cirq.transformers.measurement_transformers import _ConfusionChannel, _MeasurementQid, _mod_add


def assert_equivalent_to_deferred(circuit: cirq.Circuit):
    qubits = list(circuit.all_qubits())
    sim = cirq.Simulator()
    num_qubits = len(qubits)
    dimensions = [q.dimension for q in qubits]
    for i in range(np.prod(dimensions)):
        bits = cirq.big_endian_int_to_digits(i, base=dimensions)
        modified = cirq.Circuit()
        for j in range(num_qubits):
            modified.append(cirq.XPowGate(dimension=qubits[j].dimension)(qubits[j]) ** bits[j])
        modified.append(circuit)
        deferred = cirq.defer_measurements(modified)
        result = sim.simulate(modified)
        result1 = sim.simulate(deferred)
        np.testing.assert_equal(result.measurements, result1.measurements)


def test_basic():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a'),
        cirq.X(q1).with_classical_controls('a'),
        cirq.measure(q1, key='b'),
    )
    assert_equivalent_to_deferred(circuit)
    deferred = cirq.defer_measurements(circuit)
    q_ma = _MeasurementQid('a', q0)
    cirq.testing.assert_same_circuits(
        deferred,
        cirq.Circuit(
            cirq.CX(q0, q_ma),
            cirq.CX(q_ma, q1),
            cirq.measure(q_ma, key='a'),
            cirq.measure(q1, key='b'),
        ),
    )


def test_qudits():
    q0, q1 = cirq.LineQid.range(2, dimension=3)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a'),
        cirq.XPowGate(dimension=3).on(q1).with_classical_controls('a'),
        cirq.measure(q1, key='b'),
    )
    assert_equivalent_to_deferred(circuit)
    deferred = cirq.defer_measurements(circuit)
    q_ma = _MeasurementQid('a', q0)
    cirq.testing.assert_same_circuits(
        deferred,
        cirq.Circuit(
            _mod_add(q0, q_ma),
            cirq.XPowGate(dimension=3).on(q1).controlled_by(q_ma, control_values=[[1, 2]]),
            cirq.measure(q_ma, key='a'),
            cirq.measure(q1, key='b'),
        ),
    )


def test_sympy_control():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a'),
        cirq.X(q1).with_classical_controls(sympy.Symbol('a')),
        cirq.measure(q1, key='b'),
    )
    assert_equivalent_to_deferred(circuit)
    deferred = cirq.defer_measurements(circuit)
    q_ma = _MeasurementQid('a', q0)
    cirq.testing.assert_same_circuits(
        deferred,
        cirq.Circuit(
            cirq.CX(q0, q_ma),
            cirq.CX(q_ma, q1),
            cirq.measure(q_ma, key='a'),
            cirq.measure(q1, key='b'),
        ),
    )


def test_sympy_qudits():
    q0, q1 = cirq.LineQid.range(2, dimension=3)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a'),
        cirq.XPowGate(dimension=3).on(q1).with_classical_controls(sympy.Symbol('a')),
        cirq.measure(q1, key='b'),
    )
    assert_equivalent_to_deferred(circuit)
    deferred = cirq.defer_measurements(circuit)
    q_ma = _MeasurementQid('a', q0)
    cirq.testing.assert_same_circuits(
        deferred,
        cirq.Circuit(
            _mod_add(q0, q_ma),
            cirq.XPowGate(dimension=3).on(q1).controlled_by(q_ma, control_values=[[1, 2]]),
            cirq.measure(q_ma, key='a'),
            cirq.measure(q1, key='b'),
        ),
    )


def test_sympy_control_complex():
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a'),
        cirq.measure(q1, key='b'),
        cirq.X(q2).with_classical_controls(sympy_parser.parse_expr('a >= b')),
        cirq.measure(q2, key='c'),
    )
    assert_equivalent_to_deferred(circuit)
    deferred = cirq.defer_measurements(circuit)
    q_ma = _MeasurementQid('a', q0)
    q_mb = _MeasurementQid('b', q1)
    cirq.testing.assert_same_circuits(
        deferred,
        cirq.Circuit(
            cirq.CX(q0, q_ma),
            cirq.CX(q1, q_mb),
            cirq.ControlledOperation(
                [q_ma, q_mb], cirq.X(q2), cirq.SumOfProducts([[0, 0], [1, 0], [1, 1]])
            ),
            cirq.measure(q_ma, key='a'),
            cirq.measure(q_mb, key='b'),
            cirq.measure(q2, key='c'),
        ),
    )


def test_sympy_control_complex_qudit():
    q0, q1, q2 = cirq.LineQid.for_qid_shape((4, 2, 2))
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a'),
        cirq.measure(q1, key='b'),
        cirq.X(q2).with_classical_controls(sympy_parser.parse_expr('a > b')),
        cirq.measure(q2, key='c'),
    )
    assert_equivalent_to_deferred(circuit)
    deferred = cirq.defer_measurements(circuit)
    q_ma = _MeasurementQid('a', q0)
    q_mb = _MeasurementQid('b', q1)
    cirq.testing.assert_same_circuits(
        deferred,
        cirq.Circuit(
            _mod_add(q0, q_ma),
            cirq.CX(q1, q_mb),
            cirq.ControlledOperation(
                [q_ma, q_mb],
                cirq.X(q2),
                cirq.SumOfProducts([[1, 0], [2, 0], [3, 0], [2, 1], [3, 1]]),
            ),
            cirq.measure(q_ma, key='a'),
            cirq.measure(q_mb, key='b'),
            cirq.measure(q2, key='c'),
        ),
    )


def test_multiple_sympy_control_complex():
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a'),
        cirq.measure(q1, key='b'),
        cirq.X(q2)
        .with_classical_controls(sympy_parser.parse_expr('a >= b'))
        .with_classical_controls(sympy_parser.parse_expr('a <= b')),
        cirq.measure(q2, key='c'),
    )
    assert_equivalent_to_deferred(circuit)
    deferred = cirq.defer_measurements(circuit)
    q_ma = _MeasurementQid('a', q0)
    q_mb = _MeasurementQid('b', q1)
    cirq.testing.assert_same_circuits(
        deferred,
        cirq.Circuit(
            cirq.CX(q0, q_ma),
            cirq.CX(q1, q_mb),
            cirq.ControlledOperation(
                [q_ma, q_mb], cirq.X(q2), cirq.SumOfProducts([[0, 0], [1, 1]])
            ),
            cirq.measure(q_ma, key='a'),
            cirq.measure(q_mb, key='b'),
            cirq.measure(q2, key='c'),
        ),
    )


def test_sympy_and_key_control():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a'),
        cirq.X(q1).with_classical_controls(sympy.Symbol('a')).with_classical_controls('a'),
        cirq.measure(q1, key='b'),
    )
    assert_equivalent_to_deferred(circuit)
    deferred = cirq.defer_measurements(circuit)
    q_ma = _MeasurementQid('a', q0)
    cirq.testing.assert_same_circuits(
        deferred,
        cirq.Circuit(
            cirq.CX(q0, q_ma),
            cirq.CX(q_ma, q1),
            cirq.measure(q_ma, key='a'),
            cirq.measure(q1, key='b'),
        ),
    )


def test_sympy_control_multiqubit():
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.measure(q0, q1, key='a'),
        cirq.X(q2).with_classical_controls(sympy_parser.parse_expr('a >= 2')),
        cirq.measure(q2, key='c'),
    )
    assert_equivalent_to_deferred(circuit)
    deferred = cirq.defer_measurements(circuit)
    q_ma0 = _MeasurementQid('a', q0)
    q_ma1 = _MeasurementQid('a', q1)
    cirq.testing.assert_same_circuits(
        deferred,
        cirq.Circuit(
            cirq.CX(q0, q_ma0),
            cirq.CX(q1, q_ma1),
            cirq.ControlledOperation(
                [q_ma0, q_ma1], cirq.X(q2), cirq.SumOfProducts([[1, 0], [1, 1]])
            ),
            cirq.measure(q_ma0, q_ma1, key='a'),
            cirq.measure(q2, key='c'),
        ),
    )


def test_nocompile_context():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a').with_tags('nocompile'),
        cirq.X(q1).with_classical_controls('a').with_tags('nocompile'),
        cirq.measure(q1, key='b'),
    )
    deferred = cirq.defer_measurements(
        circuit, context=cirq.TransformerContext(tags_to_ignore=('nocompile',))
    )
    cirq.testing.assert_same_circuits(deferred, circuit)


def test_nocompile_context_leaves_invalid_circuit():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a').with_tags('nocompile'),
        cirq.X(q1).with_classical_controls('a'),
        cirq.measure(q1, key='b'),
    )
    with pytest.raises(ValueError, match='Deferred measurement for key=a not found'):
        _ = cirq.defer_measurements(
            circuit, context=cirq.TransformerContext(tags_to_ignore=('nocompile',))
        )


def test_pauli():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.PauliMeasurementGate(cirq.DensePauliString('Y'), key='a').on(q0),
        cirq.X(q1).with_classical_controls('a'),
        cirq.measure(q1, key='b'),
    )
    deferred = cirq.defer_measurements(circuit)
    q_ma = _MeasurementQid('a', q0)
    cirq.testing.assert_same_circuits(
        cirq.unroll_circuit_op(deferred),
        cirq.Circuit(
            cirq.SingleQubitCliffordGate.X_sqrt(q0),
            cirq.CX(q0, q_ma),
            (cirq.SingleQubitCliffordGate.X_sqrt(q0) ** -1),
            cirq.Moment(cirq.CX(q_ma, q1)),
            cirq.measure(q_ma, key='a'),
            cirq.measure(q1, key='b'),
        ),
    )


def test_extra_measurements():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a'),
        cirq.measure(q0, key='b'),
        cirq.X(q1).with_classical_controls('a'),
        cirq.measure(q1, key='c'),
    )
    assert_equivalent_to_deferred(circuit)
    deferred = cirq.defer_measurements(circuit)
    q_ma = _MeasurementQid('a', q0)
    cirq.testing.assert_same_circuits(
        deferred,
        cirq.Circuit(
            cirq.CX(q0, q_ma),
            cirq.CX(q_ma, q1),
            cirq.measure(q_ma, key='a'),
            cirq.measure(q0, key='b'),
            cirq.measure(q1, key='c'),
        ),
    )


def test_extra_controlled_bits():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a'),
        cirq.CX(q0, q1).with_classical_controls('a'),
        cirq.measure(q1, key='b'),
    )
    assert_equivalent_to_deferred(circuit)
    deferred = cirq.defer_measurements(circuit)
    q_ma = _MeasurementQid('a', q0)
    cirq.testing.assert_same_circuits(
        deferred,
        cirq.Circuit(
            cirq.CX(q0, q_ma),
            cirq.CCX(q_ma, q0, q1),
            cirq.measure(q_ma, key='a'),
            cirq.measure(q1, key='b'),
        ),
    )


def test_extra_control_bits():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a'),
        cirq.measure(q0, key='b'),
        cirq.X(q1).with_classical_controls('a', 'b'),
        cirq.measure(q1, key='c'),
    )
    assert_equivalent_to_deferred(circuit)
    deferred = cirq.defer_measurements(circuit)
    q_ma = _MeasurementQid('a', q0)
    q_mb = _MeasurementQid('b', q0)
    cirq.testing.assert_same_circuits(
        deferred,
        cirq.Circuit(
            cirq.CX(q0, q_ma),
            cirq.CX(q0, q_mb),
            cirq.CCX(q_ma, q_mb, q1),
            cirq.measure(q_ma, key='a'),
            cirq.measure(q_mb, key='b'),
            cirq.measure(q1, key='c'),
        ),
    )


def test_subcircuit():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.CircuitOperation(
            cirq.FrozenCircuit(
                cirq.measure(q0, key='a'),
                cirq.X(q1).with_classical_controls('a'),
                cirq.measure(q1, key='b'),
            )
        )
    )
    assert_equivalent_to_deferred(circuit)
    deferred = cirq.defer_measurements(circuit)
    q_m = _MeasurementQid('a', q0)
    cirq.testing.assert_same_circuits(
        deferred,
        cirq.Circuit(
            cirq.CX(q0, q_m),
            cirq.CX(q_m, q1),
            cirq.measure(q_m, key='a'),
            cirq.measure(q1, key='b'),
        ),
    )


def test_multi_qubit_measurements():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, q1, key='a'),
        cirq.X(q0),
        cirq.measure(q0, key='b'),
        cirq.measure(q1, key='c'),
    )
    assert_equivalent_to_deferred(circuit)
    deferred = cirq.defer_measurements(circuit)
    q_ma0 = _MeasurementQid('a', q0)
    q_ma1 = _MeasurementQid('a', q1)
    cirq.testing.assert_same_circuits(
        deferred,
        cirq.Circuit(
            cirq.CX(q0, q_ma0),
            cirq.CX(q1, q_ma1),
            cirq.X(q0),
            cirq.measure(q_ma0, q_ma1, key='a'),
            cirq.measure(q0, key='b'),
            cirq.measure(q1, key='c'),
        ),
    )


def test_multi_qubit_control():
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.measure(q0, q1, key='a'),
        cirq.X(q2).with_classical_controls('a'),
        cirq.measure(q2, key='b'),
    )
    assert_equivalent_to_deferred(circuit)
    deferred = cirq.defer_measurements(circuit)
    q_ma0 = _MeasurementQid('a', q0)
    q_ma1 = _MeasurementQid('a', q1)
    cirq.testing.assert_same_circuits(
        deferred,
        cirq.Circuit(
            cirq.CX(q0, q_ma0),
            cirq.CX(q1, q_ma1),
            cirq.X(q2).controlled_by(
                q_ma0, q_ma1, control_values=cirq.SumOfProducts(((0, 1), (1, 0), (1, 1)))
            ),
            cirq.measure(q_ma0, q_ma1, key='a'),
            cirq.measure(q2, key='b'),
        ),
    )


@pytest.mark.parametrize('index', [-3, -2, -1, 0, 1, 2])
def test_repeated(index: int):
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a'),  # The control measurement when `index` is 0 or -2
        cirq.X(q0),
        cirq.measure(q0, key='a'),  # The control measurement when `index` is 1 or -1
        cirq.X(q1).with_classical_controls(cirq.KeyCondition(cirq.MeasurementKey('a'), index)),
        cirq.measure(q1, key='b'),
    )
    if index in [-3, 2]:
        with pytest.raises(ValueError, match='Invalid index'):
            _ = cirq.defer_measurements(circuit)
        return
    assert_equivalent_to_deferred(circuit)
    deferred = cirq.defer_measurements(circuit)
    q_ma = _MeasurementQid('a', q0)  # The ancilla qubit created for the first `a` measurement
    q_ma1 = _MeasurementQid('a', q0, 1)  # The ancilla qubit created for the second `a` measurement
    # The ancilla used for control should match the measurement used for control above.
    q_expected_control = q_ma if index in [0, -2] else q_ma1
    cirq.testing.assert_same_circuits(
        deferred,
        cirq.Circuit(
            cirq.CX(q0, q_ma),
            cirq.X(q0),
            cirq.CX(q0, q_ma1),
            cirq.Moment(cirq.CX(q_expected_control, q1)),
            cirq.measure(q_ma, key='a'),
            cirq.measure(q_ma1, key='a'),
            cirq.measure(q1, key='b'),
        ),
    )


def test_diagram():
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(
        cirq.measure(q0, q2, key='a'),
        cirq.measure(q1, q3, key='b'),
        cirq.X(q0),
        cirq.measure(q0, q1, q2, q3, key='c'),
    )
    deferred = cirq.defer_measurements(circuit)
    cirq.testing.assert_has_diagram(
        deferred,
        """
                      ┌────┐
0: ────────────────────@───────X────────M('c')───
                       │                │
1: ────────────────────┼─@──────────────M────────
                       │ │              │
2: ────────────────────┼@┼──────────────M────────
                       │││              │
3: ────────────────────┼┼┼@─────────────M────────
                       ││││
M('a[0]', q=q(0)): ────X┼┼┼────M('a')────────────
                        │││    │
M('a[0]', q=q(2)): ─────X┼┼────M─────────────────
                         ││
M('b[0]', q=q(1)): ──────X┼────M('b')────────────
                          │    │
M('b[0]', q=q(3)): ───────X────M─────────────────
                      └────┘
""",
        use_unicode_characters=True,
    )


def test_repr():
    def test_repr(qid: _MeasurementQid):
        cirq.testing.assert_equivalent_repr(qid, global_vals={'_MeasurementQid': _MeasurementQid})

    test_repr(_MeasurementQid('a', cirq.LineQubit(0)))
    test_repr(_MeasurementQid('a', cirq.NamedQubit('x')))
    test_repr(_MeasurementQid('a', cirq.NamedQid('x', 4)))
    test_repr(_MeasurementQid('a', cirq.GridQubit(2, 3)))
    test_repr(_MeasurementQid('0:1:a', cirq.LineQid(9, 4)))


def test_confusion_map():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(q0),
        cirq.measure(q0, key='a', confusion_map={(0,): np.array([[0.8, 0.2], [0.1, 0.9]])}),
        cirq.X(q1).with_classical_controls('a'),
        cirq.measure(q1, key='b'),
    )
    deferred = cirq.defer_measurements(circuit)

    # We use DM simulator because the deferred circuit has channels
    sim = cirq.DensityMatrixSimulator()

    # 10K samples would take a long time if we had not deferred the measurements, as we'd have to
    # run 10K simulations. Here with DM simulator it's 100ms.
    result = sim.sample(deferred, repetitions=10_000)

    # This should be 5_000 due to the H, then 1_000 more due to 0's flipping to 1's with p=0.2, and
    # then 500 less due to 1's flipping to 0's with p=0.1, so 5_500.
    assert 5_100 <= np.sum(result['a']) <= 5_900
    assert np.all(result['a'] == result['b'])


def test_confusion_map_density_matrix():
    q0, q1 = cirq.LineQubit.range(2)
    p_q0 = 0.3  # probability to measure 1 for q0
    confusion = np.array([[0.8, 0.2], [0.1, 0.9]])
    circuit = cirq.Circuit(
        # Rotate q0 such that the probability to measure 1 is p_q0
        cirq.X(q0) ** (np.arcsin(np.sqrt(p_q0)) * 2 / np.pi),
        cirq.measure(q0, key='a', confusion_map={(0,): confusion}),
        cirq.X(q1).with_classical_controls('a'),
    )
    deferred = cirq.defer_measurements(circuit)
    q_order = (q0, q1, _MeasurementQid('a', q0))
    rho = cirq.final_density_matrix(deferred, qubit_order=q_order).reshape((2,) * 6)

    # q0 density matrix should be a diagonal with the probabilities [1-p, p].
    q0_probs = [1 - p_q0, p_q0]
    assert np.allclose(cirq.partial_trace(rho, [0]), np.diag(q0_probs))

    # q1 and the ancilla should both be the q1 probs matmul the confusion matrix.
    expected = np.diag(q0_probs @ confusion)
    assert np.allclose(cirq.partial_trace(rho, [1]), expected)
    assert np.allclose(cirq.partial_trace(rho, [2]), expected)


def test_confusion_map_invert_mask_ordering():
    q0 = cirq.LineQubit(0)
    # Confusion map sets the measurement to zero, and the invert mask changes it to one.
    # If these are run out of order then the result would be zero.
    circuit = cirq.Circuit(
        cirq.measure(
            q0, key='a', confusion_map={(0,): np.array([[1, 0], [1, 0]])}, invert_mask=(1,)
        ),
        cirq.I(q0),
    )
    assert_equivalent_to_deferred(circuit)


def test_confusion_map_qudits():
    q0 = cirq.LineQid(0, dimension=3)
    # First op takes q0 to superposed state, then confusion map measures 2 regardless.
    circuit = cirq.Circuit(
        cirq.XPowGate(dimension=3).on(q0) ** 1.3,
        cirq.measure(
            q0, key='a', confusion_map={(0,): np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])}
        ),
        cirq.IdentityGate(qid_shape=(3,)).on(q0),
    )
    assert_equivalent_to_deferred(circuit)


def test_multi_qubit_confusion_map():
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.measure(
            q0,
            q1,
            key='a',
            confusion_map={
                (0, 1): np.array(
                    [
                        [0.7, 0.1, 0.1, 0.1],
                        [0.1, 0.6, 0.1, 0.2],
                        [0.2, 0.2, 0.5, 0.1],
                        [0.0, 0.0, 1.0, 0.0],
                    ]
                )
            },
        ),
        cirq.X(q2).with_classical_controls('a'),
        cirq.measure(q2, key='b'),
    )
    deferred = cirq.defer_measurements(circuit)
    sim = cirq.DensityMatrixSimulator()
    result = sim.sample(deferred, repetitions=10_000)

    # The initial state is zero, so the first measurement will confuse by the first line in the
    # map, giving 7000 0's, 1000 1's, 1000 2's, and 1000 3's, for a sum of 6000 on average.
    assert 5_600 <= np.sum(result['a']) <= 6_400

    # The measurement will be non-zero 3000 times on average.
    assert 2_600 <= np.sum(result['b']) <= 3_400

    # Try a deterministic one: initial state is 3, which the confusion map sends to 2 with p=1.
    deferred.insert(0, cirq.X.on_each(q0, q1))
    result = sim.sample(deferred, repetitions=100)
    assert np.sum(result['a']) == 200
    assert np.sum(result['b']) == 100


def test_confusion_map_errors():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a', confusion_map={(0,): np.array([1])}),
        cirq.X(q1).with_classical_controls('a'),
    )
    with pytest.raises(ValueError, match='map must be 2D'):
        _ = cirq.defer_measurements(circuit)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a', confusion_map={(0,): np.array([[0.7, 0.3]])}),
        cirq.X(q1).with_classical_controls('a'),
    )
    with pytest.raises(ValueError, match='map must be square'):
        _ = cirq.defer_measurements(circuit)
    circuit = cirq.Circuit(
        cirq.measure(
            q0,
            key='a',
            confusion_map={(0,): np.array([[0.7, 0.1, 0.2], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6]])},
        ),
        cirq.X(q1).with_classical_controls('a'),
    )
    with pytest.raises(ValueError, match='size does not match'):
        _ = cirq.defer_measurements(circuit)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a', confusion_map={(0,): np.array([[-1, 2], [0, 1]])}),
        cirq.X(q1).with_classical_controls('a'),
    )
    with pytest.raises(ValueError, match='negative probabilities'):
        _ = cirq.defer_measurements(circuit)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a', confusion_map={(0,): np.array([[0.3, 0.3], [0.3, 0.3]])}),
        cirq.X(q1).with_classical_controls('a'),
    )
    with pytest.raises(ValueError, match='invalid probabilities'):
        _ = cirq.defer_measurements(circuit)


def test_dephase():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.CircuitOperation(
            cirq.FrozenCircuit(
                cirq.CX(q1, q0),
                cirq.measure(q0, key='a'),
                cirq.CX(q0, q1),
                cirq.measure(q1, key='b'),
            )
        )
    )
    dephased = cirq.dephase_measurements(circuit)
    cirq.testing.assert_same_circuits(
        dephased,
        cirq.Circuit(
            cirq.CircuitOperation(
                cirq.FrozenCircuit(
                    cirq.CX(q1, q0),
                    cirq.KrausChannel.from_channel(cirq.phase_damp(1), key='a')(q0),
                    cirq.CX(q0, q1),
                    cirq.KrausChannel.from_channel(cirq.phase_damp(1), key='b')(q1),
                )
            )
        ),
    )


def test_dephase_classical_conditions():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a'),
        cirq.X(q1).with_classical_controls('a'),
        cirq.measure(q1, key='b'),
    )
    with pytest.raises(ValueError, match='defer_measurements first to remove classical controls'):
        _ = cirq.dephase_measurements(circuit)


def test_dephase_nocompile_context():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.CircuitOperation(
            cirq.FrozenCircuit(
                cirq.CX(q1, q0),
                cirq.measure(q0, key='a').with_tags('nocompile'),
                cirq.CX(q0, q1),
                cirq.measure(q1, key='b'),
            )
        )
    )
    dephased = cirq.dephase_measurements(
        circuit, context=cirq.TransformerContext(deep=True, tags_to_ignore=('nocompile',))
    )
    cirq.testing.assert_same_circuits(
        dephased,
        cirq.Circuit(
            cirq.CircuitOperation(
                cirq.FrozenCircuit(
                    cirq.CX(q1, q0),
                    cirq.measure(q0, key='a').with_tags('nocompile'),
                    cirq.CX(q0, q1),
                    cirq.KrausChannel.from_channel(cirq.phase_damp(1), key='b')(q1),
                )
            )
        ),
    )


def test_drop_terminal():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.CircuitOperation(
            cirq.FrozenCircuit(cirq.CX(q0, q1), cirq.measure(q0, q1, key='a~b', invert_mask=[0, 1]))
        )
    )
    dropped = cirq.drop_terminal_measurements(circuit)
    cirq.testing.assert_same_circuits(
        dropped,
        cirq.Circuit(
            cirq.CircuitOperation(cirq.FrozenCircuit(cirq.CX(q0, q1), cirq.I(q0), cirq.X(q1)))
        ),
    )


def test_drop_terminal_nonterminal_error():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.CircuitOperation(
            cirq.FrozenCircuit(cirq.measure(q0, q1, key='a~b', invert_mask=[0, 1]), cirq.CX(q0, q1))
        )
    )
    with pytest.raises(ValueError, match='Circuit contains a non-terminal measurement'):
        _ = cirq.drop_terminal_measurements(circuit)

    with pytest.raises(ValueError, match='Context has `deep=False`'):
        _ = cirq.drop_terminal_measurements(circuit, context=cirq.TransformerContext(deep=False))

    with pytest.raises(ValueError, match='Context has `deep=False`'):
        _ = cirq.drop_terminal_measurements(circuit, context=None)


def test_confusion_channel_consistency():
    two_d_chan = _ConfusionChannel(np.array([[0.5, 0.5], [0.4, 0.6]]), shape=(2,))
    cirq.testing.assert_has_consistent_apply_channel(two_d_chan)
    three_d_chan = _ConfusionChannel(
        np.array([[0.5, 0.3, 0.2], [0.4, 0.5, 0.1], [0, 0, 1]]), shape=(3,)
    )
    cirq.testing.assert_has_consistent_apply_channel(three_d_chan)
    two_q_chan = _ConfusionChannel(
        np.array([[0.5, 0.3, 0.1, 0.1], [0.4, 0.5, 0.1, 0], [0, 0, 1, 0], [0, 0, 0.5, 0.5]]),
        shape=(2, 2),
    )
    cirq.testing.assert_has_consistent_apply_channel(two_q_chan)
