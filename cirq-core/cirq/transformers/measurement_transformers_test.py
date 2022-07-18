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

import cirq
from cirq.transformers.measurement_transformers import _MeasurementQid


def assert_equivalent_to_deferred(circuit: cirq.Circuit):
    qubits = list(circuit.all_qubits())
    sim = cirq.Simulator()
    num_qubits = len(qubits)
    for i in range(2**num_qubits):
        bits = cirq.big_endian_int_to_bits(i, bit_count=num_qubits)
        modified = cirq.Circuit()
        for j in range(num_qubits):
            if bits[j]:
                modified.append(cirq.X(qubits[j]))
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
0: ─────────────────@───────X────────M('c')───
                    │                │
1: ─────────────────┼─@──────────────M────────
                    │ │              │
2: ─────────────────┼@┼──────────────M────────
                    │││              │
3: ─────────────────┼┼┼@─────────────M────────
                    ││││
M('a', q=q(0)): ────X┼┼┼────M('a')────────────
                     │││    │
M('a', q=q(2)): ─────X┼┼────M─────────────────
                      ││
M('b', q=q(1)): ──────X┼────M('b')────────────
                       │    │
M('b', q=q(3)): ───────X────M─────────────────
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


def test_sympy_control():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, q1, key='a'), cirq.X(q1).with_classical_controls(sympy.Symbol('a'))
    )
    with pytest.raises(ValueError, match='Only KeyConditions are allowed'):
        _ = cirq.defer_measurements(circuit)


def test_confusion_map():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, q1, key='a', confusion_map={(0,): np.array([[0.9, 0.1], [0.1, 0.9]])}),
        cirq.X(q1).with_classical_controls('a'),
    )
    with pytest.raises(
        NotImplementedError, match='Deferring confused measurement is not implemented'
    ):
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
