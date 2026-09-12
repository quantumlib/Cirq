# Copyright 2024 The Cirq Developers
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

"""Tests for the PyZX-based ZX transformer."""

from __future__ import annotations

import math
from collections.abc import Callable

import cirq
import numpy as np
import pyzx as zx
import pytest
from pyzx.circuit.gates import ConditionalGate
from pyzx.circuit.gates import Measurement as PyzxMeasurement
from pyzx.circuit.gates import Reset as PyzxReset

from cirq.contrib.zxtransformer.zxtransformer import ZXTransformer, zx_transformer

_ATOL = 1e-8


def _gate_count(circuit: cirq.AbstractCircuit) -> int:
    return sum(1 for _ in circuit.all_operations())


def _assert_unitarily_equivalent(
    original: cirq.AbstractCircuit, transformed: cirq.AbstractCircuit
) -> None:
    qubit_map = {qid: qid for qid in original.all_qubits()}
    cirq.testing.assert_circuits_have_same_unitary_given_final_permutation(
        original, transformed, qubit_map
    )


def _assert_channel_equivalent(
    original: cirq.AbstractCircuit,
    transformed: cirq.AbstractCircuit,
) -> None:
    qubit_order = sorted(original.all_qubits())
    original_dm = cirq.final_density_matrix(
        original, qubit_order=qubit_order, ignore_measurement_results=True
    )
    transformed_dm = cirq.final_density_matrix(
        transformed, qubit_order=qubit_order, ignore_measurement_results=True
    )
    np.testing.assert_allclose(transformed_dm, original_dm, atol=_ATOL)


def _run_zx_transformer(
    circuit: cirq.Circuit,
    optimize: Callable[[zx.Circuit], zx.Circuit] | None = None,
) -> cirq.Circuit:
    transformed = ZXTransformer(optimize)(circuit)
    _assert_unitarily_equivalent(circuit, transformed)
    return transformed


def test_empty_circuit() -> None:
    circuit = cirq.Circuit()
    result = zx_transformer(circuit)
    assert len(result) == 0


def test_single_gate_circuit() -> None:
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q))
    result = _run_zx_transformer(circuit)
    assert _gate_count(result) >= 1


def test_gate_count_reduction_on_redundant_rotations() -> None:
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q), cirq.H(q), cirq.Z(q), cirq.Z(q))
    result = ZXTransformer()(circuit)
    assert _gate_count(result) <= _gate_count(circuit)
    if len(result) == 0:
        np.testing.assert_allclose(circuit.unitary(qubit_order=(q,)), np.eye(2), atol=_ATOL)
    else:
        _assert_unitarily_equivalent(circuit, result)


def test_random_unitary_circuit() -> None:
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.testing.random_circuit(
        qubits, n_moments=8, op_density=0.8, random_state=np.random.RandomState(6585)
    )
    _run_zx_transformer(circuit)


def test_basic_circuit() -> None:
    """Regression circuit from PyZX benchmarks (mod5_4_before)."""
    q = cirq.LineQubit.range(5)
    circuit = cirq.Circuit(
        cirq.X(q[4]),
        cirq.H(q[4]),
        cirq.CCZ(q[0], q[3], q[4]),
        cirq.CCZ(q[2], q[3], q[4]),
        cirq.H(q[4]),
        cirq.CX(q[3], q[4]),
        cirq.H(q[4]),
        cirq.CCZ(q[1], q[2], q[4]),
        cirq.H(q[4]),
        cirq.CX(q[2], q[4]),
        cirq.H(q[4]),
        cirq.CCZ(q[0], q[1], q[4]),
        cirq.H(q[4]),
        cirq.CX(q[1], q[4]),
        cirq.CX(q[0], q[4]),
    )
    _run_zx_transformer(circuit)


def test_fractional_gates() -> None:
    q = cirq.NamedQubit('q')
    circuit = cirq.Circuit(cirq.ry(0.5)(q), cirq.rz(0.5)(q))
    _run_zx_transformer(circuit)


def test_rotation_gates() -> None:
    q = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.XPowGate(exponent=0.5)(q[0]),
        cirq.YPowGate(exponent=0.25)(q[1]),
        cirq.ZPowGate(exponent=0.5)(q[0]),
        cirq.Rx(rads=math.pi / 4)(q[1]),
        cirq.Ry(rads=math.pi / 2)(q[0]),
        cirq.Rz(rads=math.pi / 4)(q[1]),
        cirq.CX(q[0], q[1]),
    )
    _run_zx_transformer(circuit)


def test_custom_optimize() -> None:
    q = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(
        cirq.H(q[0]),
        cirq.H(q[1]),
        cirq.H(q[2]),
        cirq.H(q[3]),
        cirq.CX(q[0], q[1]),
        cirq.CX(q[1], q[2]),
        cirq.CX(q[2], q[3]),
        cirq.CX(q[3], q[0]),
    )

    def optimize(circ: zx.Circuit) -> zx.Circuit:
        return circ.to_basic_gates()

    _run_zx_transformer(circuit, optimize)


def test_measurement_converted_to_pyzx() -> None:
    q = cirq.NamedQubit('q')
    circuit = cirq.Circuit(cirq.H(q), cirq.measure(q, key='c'), cirq.H(q))
    transformer = ZXTransformer()
    circuits_and_ops = transformer._cirq_to_circuits_and_ops(circuit)
    assert len(circuits_and_ops) == 1
    assert isinstance(circuits_and_ops[0], zx.Circuit)
    assert transformer.measurement_keys == ['c']
    assert any(isinstance(g, PyzxMeasurement) for g in circuits_and_ops[0].gates)


def test_multi_qubit_measurement_converted_to_pyzx() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1), cirq.measure(q0, q1, key='m'))
    transformer = ZXTransformer()
    circuits_and_ops = transformer._cirq_to_circuits_and_ops(circuit)
    assert len(circuits_and_ops) == 1
    assert transformer.measurement_keys == ['m', 'm']
    assert sum(isinstance(g, PyzxMeasurement) for g in circuits_and_ops[0].gates) == 2

    result = transformer(circuit)
    measurement_ops = [op for moment in result for op in moment if cirq.is_measurement(op)]
    assert len(measurement_ops) == 1
    assert cirq.measurement_key_name(measurement_ops[0]) == 'm'
    assert measurement_ops[0].qubits == (q0, q1)
    _assert_channel_equivalent(circuit, result)


def test_measurement_with_invert_mask_preserved() -> None:
    q = cirq.NamedQubit('q')
    circuit = cirq.Circuit(cirq.X(q), cirq.measure(q, key='m', invert_mask=(True,)))
    result = ZXTransformer()(circuit)
    measurement_ops = [op for moment in result for op in moment if cirq.is_measurement(op)]
    assert len(measurement_ops) == 1
    gate = measurement_ops[0].gate
    assert isinstance(gate, cirq.MeasurementGate)
    assert gate.invert_mask == (True,)
    _assert_channel_equivalent(circuit, result)


def test_reset_converted_to_pyzx() -> None:
    q = cirq.NamedQubit('q')
    circuit = cirq.Circuit(cirq.H(q), cirq.ResetChannel()(q), cirq.H(q))
    transformer = ZXTransformer()
    circuits_and_ops = transformer._cirq_to_circuits_and_ops(circuit)
    assert len(circuits_and_ops) == 1
    assert any(isinstance(g, PyzxReset) for g in circuits_and_ops[0].gates)


def test_reset_preserved_by_transformer() -> None:
    q = cirq.NamedQubit('q')
    circuit = cirq.Circuit(cirq.X(q), cirq.ResetChannel()(q), cirq.X(q))
    result = ZXTransformer()(circuit)
    reset_ops = [
        op for moment in result for op in moment if isinstance(op.gate, cirq.ResetChannel)
    ]
    assert len(reset_ops) == 1
    _assert_channel_equivalent(circuit, result)


def test_conditional_gate_unsupported_passthrough() -> None:
    q = cirq.NamedQubit('q')
    circuit = cirq.Circuit(cirq.X(q), cirq.H(q).with_classical_controls('c'), cirq.X(q))
    transformer = ZXTransformer()
    circuits_and_ops = transformer._cirq_to_circuits_and_ops(circuit)
    assert len(circuits_and_ops) == 3
    assert isinstance(circuits_and_ops[0], zx.Circuit)
    assert isinstance(circuits_and_ops[1], cirq.ClassicallyControlledOperation)
    assert isinstance(circuits_and_ops[2], zx.Circuit)


def test_conditional_gate_x_converted() -> None:
    q = cirq.NamedQubit('q')
    circuit = cirq.Circuit(cirq.X(q).with_classical_controls('c'))
    transformer = ZXTransformer()
    circuits_and_ops = transformer._cirq_to_circuits_and_ops(circuit)
    assert len(circuits_and_ops) == 1
    assert any(isinstance(g, ConditionalGate) for g in circuits_and_ops[0].gates)


def test_conditional_gate_z_rotation_converted() -> None:
    q = cirq.NamedQubit('q')
    circuit = cirq.Circuit(cirq.S(q).with_classical_controls('c'))
    transformer = ZXTransformer()
    circuits_and_ops = transformer._cirq_to_circuits_and_ops(circuit)
    cond_gates = [g for g in circuits_and_ops[0].gates if isinstance(g, ConditionalGate)]
    assert len(cond_gates) == 1
    assert cond_gates[0].condition_register == 'c'


def test_conditional_gate_preserved_by_transformer() -> None:
    q = cirq.NamedQubit('q')
    circuit = cirq.Circuit(cirq.measure(q, key='c'), cirq.X(q).with_classical_controls('c'))
    result = ZXTransformer()(circuit)
    assert any(cirq.is_measurement(op) for moment in result for op in moment)
    assert any(
        isinstance(op, cirq.ClassicallyControlledOperation) for moment in result for op in moment
    )


def test_ccx_and_cswap_gates() -> None:
    q = cirq.LineQubit.range(3)
    _run_zx_transformer(cirq.Circuit(cirq.CCX(q[0], q[1], q[2])))
    _run_zx_transformer(cirq.Circuit(cirq.CSWAP(q[0], q[1], q[2])))


def test_xx_zz_gates() -> None:
    q = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.XXPowGate(exponent=0.5)(q[0], q[1]),
        cirq.ZZPowGate(exponent=0.25)(q[0], q[1]),
    )
    _run_zx_transformer(circuit)


def test_hybrid_optimization_segments() -> None:
    q = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(q[0]),
        cirq.H(q[0]),
        cirq.measure(q[0], key='m'),
        cirq.H(q[1]),
        cirq.H(q[1]),
    )
    result = ZXTransformer()(circuit)
    assert any(cirq.is_measurement(op) for moment in result for op in moment)
    _assert_channel_equivalent(circuit, result)


def test_multiple_measurements() -> None:
    q = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(q[0]),
        cirq.H(q[1]),
        cirq.measure(q[0], key='m0'),
        cirq.measure(q[1], key='m1'),
    )
    result = ZXTransformer()(circuit)
    keys = {cirq.measurement_key_name(op) for moment in result for op in moment if cirq.is_measurement(op)}
    assert keys == {'m0', 'm1'}
    _assert_channel_equivalent(circuit, result)


@pytest.mark.parametrize(
    'circuit',
    [
        cirq.Circuit(cirq.X(cirq.LineQubit(0))),
        cirq.Circuit(cirq.H(cirq.LineQubit(0)), cirq.T(cirq.LineQubit(0))),
        cirq.Circuit(
            cirq.H(cirq.LineQubit(0)),
            cirq.CNOT(cirq.LineQubit(0), cirq.LineQubit(1)),
            cirq.H(cirq.LineQubit(1)),
        ),
    ],
)
def test_representative_circuits_remain_unitary(circuit: cirq.Circuit) -> None:
    _run_zx_transformer(circuit)
