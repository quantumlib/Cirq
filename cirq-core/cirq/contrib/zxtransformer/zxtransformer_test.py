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

"""Tests for Cirq ZX transformer."""

import math
from typing import Optional, Callable

import cirq
import numpy as np
import pyzx as zx
from pyzx.circuit.gates import Measurement as PyzxMeasurement
from pyzx.circuit.gates import Reset as PyzxReset
from pyzx.circuit.gates import ConditionalGate

from cirq.contrib.zxtransformer.zxtransformer import zx_transformer, _cirq_to_circuits_and_ops


def _run_zxtransformer(
    qc: cirq.Circuit, optimizer: Optional[Callable[[zx.Circuit], zx.Circuit]] = None
) -> None:
    zx_qc = zx_transformer(qc) if optimizer is None else zx_transformer(qc, optimizer=optimizer)
    qubit_map = {qid: qid for qid in qc.all_qubits()}
    cirq.testing.assert_circuits_have_same_unitary_given_final_permutation(qc, zx_qc, qubit_map)


def _run_zxtransformer_nonunitary(
    qc: cirq.Circuit, optimizer: Optional[Callable[[zx.Circuit], zx.Circuit]] = None
) -> None:
    """Assert channel equivalence for circuits with measurements or resets.

    Uses density matrix comparison with measurements treated as dephasing. Not suitable for
    circuits with classically controlled operations (the classical bits are discarded).
    """
    zx_qc = zx_transformer(qc) if optimizer is None else zx_transformer(qc, optimizer=optimizer)
    qubit_order = sorted(qc.all_qubits())
    original_dm = cirq.final_density_matrix(
        qc, qubit_order=qubit_order, ignore_measurement_results=True
    )
    transformed_dm = cirq.final_density_matrix(
        zx_qc, qubit_order=qubit_order, ignore_measurement_results=True
    )
    np.testing.assert_allclose(transformed_dm, original_dm, atol=1e-8)


def test_basic_circuit() -> None:
    """Test a basic circuit.

    Taken from https://github.com/Quantomatic/pyzx/blob/master/circuits/Fast/mod5_4_before
    """
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

    _run_zxtransformer(circuit)


def test_fractional_gates() -> None:
    """Test a circuit with gates which have a fractional phase."""
    q = cirq.NamedQubit("q")
    circuit = cirq.Circuit(cirq.ry(0.5)(q), cirq.rz(0.5)(q))
    _run_zxtransformer(circuit)


def test_custom_optimize() -> None:
    """Test custom optimize method."""
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
        # Any function that takes a zx.Circuit and returns a zx.Circuit will do.
        return circ.to_basic_gates()

    _run_zxtransformer(circuit, optimize)


def test_measurement_converted_to_pyzx() -> None:
    """Test that a measurement is converted to a pyzx gate rather than treated as opaque."""
    q = cirq.NamedQubit("q")
    circuit = cirq.Circuit(cirq.H(q), cirq.measure(q, key='c'), cirq.H(q))
    circuits_and_ops, keys, _ = _cirq_to_circuits_and_ops(circuit, [*circuit.all_qubits()])
    # All ops (H, measure, H) should be in a single pyzx circuit.
    assert len(circuits_and_ops) == 1
    assert isinstance(circuits_and_ops[0], zx.Circuit)
    assert keys == ['c']
    pyzx_circuit = circuits_and_ops[0]
    assert any(isinstance(g, PyzxMeasurement) for g in pyzx_circuit.gates)


def test_multi_qubit_measurement_converted_to_pyzx() -> None:
    """Test multi-qubit measurement conversion and key tracking."""
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1), cirq.measure(q0, q1, key='m'))
    circuits_and_ops, keys, _ = _cirq_to_circuits_and_ops(circuit, [*circuit.all_qubits()])
    assert len(circuits_and_ops) == 1
    assert isinstance(circuits_and_ops[0], zx.Circuit)
    assert keys == ['m', 'm']
    pyzx_circuit = circuits_and_ops[0]
    assert sum(isinstance(g, PyzxMeasurement) for g in pyzx_circuit.gates) == 2

    result = zx_transformer(circuit)
    measurement_ops = [op for moment in result for op in moment if cirq.is_measurement(op)]
    assert len(measurement_ops) == 1
    assert cirq.measurement_key_name(measurement_ops[0]) == 'm'
    assert measurement_ops[0].qubits == (q0, q1)
    _run_zxtransformer_nonunitary(circuit)


def test_measurement_with_invert_mask_preserved_by_transformer() -> None:
    """Test that measurement invert_mask is preserved through transformation."""
    q = cirq.NamedQubit("q")
    circuit = cirq.Circuit(cirq.X(q), cirq.measure(q, key='m', invert_mask=(True,)))
    result = zx_transformer(circuit)
    measurement_ops = [op for moment in result for op in moment if cirq.is_measurement(op)]
    assert len(measurement_ops) == 1
    assert cirq.measurement_key_name(measurement_ops[0]) == 'm'
    gate = measurement_ops[0].gate
    assert isinstance(gate, cirq.MeasurementGate)
    assert gate.invert_mask == (True,)
    _run_zxtransformer_nonunitary(circuit)


def test_rotation_gates() -> None:
    """Test rotation gates with non-integer exponents."""
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

    _run_zxtransformer(circuit)


def test_measurement_preserved_by_transformer() -> None:
    """Test that measurements are preserved through the transformer round-trip."""
    q = cirq.NamedQubit("q")
    circuit = cirq.Circuit(cirq.H(q), cirq.measure(q, key='result'))
    result = zx_transformer(circuit)
    # The result should contain a measurement with the same key.
    measurement_ops = [op for moment in result for op in moment if cirq.is_measurement(op)]
    assert len(measurement_ops) == 1
    assert cirq.measurement_key_name(measurement_ops[0]) == 'result'
    _run_zxtransformer_nonunitary(circuit)


def test_reset_converted_to_pyzx() -> None:
    """Test that a reset is converted to a pyzx gate."""
    q = cirq.NamedQubit("q")
    circuit = cirq.Circuit(cirq.H(q), cirq.ResetChannel()(q), cirq.H(q))
    circuits_and_ops, _, _ = _cirq_to_circuits_and_ops(circuit, [*circuit.all_qubits()])
    assert len(circuits_and_ops) == 1
    assert isinstance(circuits_and_ops[0], zx.Circuit)
    pyzx_circuit = circuits_and_ops[0]
    assert any(isinstance(g, PyzxReset) for g in pyzx_circuit.gates)


def test_reset_preserved_by_transformer() -> None:
    """Test that resets are preserved through the transformer round-trip."""
    q = cirq.NamedQubit("q")
    circuit = cirq.Circuit(cirq.X(q), cirq.ResetChannel()(q), cirq.X(q))
    result = zx_transformer(circuit)
    reset_ops = [op for moment in result for op in moment if isinstance(op.gate, cirq.ResetChannel)]
    assert len(reset_ops) == 1
    _run_zxtransformer_nonunitary(circuit)


def test_conditional_gate_unsupported_passthrough() -> None:
    """Test that an unsupported conditional gate (H) is left as a cirq.Operation."""
    q = cirq.NamedQubit("q")
    circuit = cirq.Circuit(cirq.X(q), cirq.H(q).with_classical_controls('c'), cirq.X(q))
    circuits_and_ops, _, _ = _cirq_to_circuits_and_ops(circuit, [*circuit.all_qubits()])
    # H is not an X or Z rotation, so it cannot be converted to ConditionalGate.
    # Expected: pyzx Circuit(X), opaque ClassicallyControlledOperation, pyzx Circuit(X).
    assert len(circuits_and_ops) == 3
    assert isinstance(circuits_and_ops[0], zx.Circuit)
    assert isinstance(circuits_and_ops[1], cirq.ClassicallyControlledOperation)
    assert isinstance(circuits_and_ops[2], zx.Circuit)


def test_conditional_gate_x_converted() -> None:
    """Test that a classically controlled X gate is converted to a pyzx ConditionalGate."""
    q = cirq.NamedQubit("q")
    circuit = cirq.Circuit(cirq.X(q).with_classical_controls('c'))
    circuits_and_ops, _, _ = _cirq_to_circuits_and_ops(circuit, [*circuit.all_qubits()])
    assert len(circuits_and_ops) == 1
    assert isinstance(circuits_and_ops[0], zx.Circuit)
    pyzx_circuit = circuits_and_ops[0]
    assert any(isinstance(g, ConditionalGate) for g in pyzx_circuit.gates)


def test_conditional_gate_z_rotation_converted() -> None:
    """Test that a classically controlled Z rotation is converted to a pyzx ConditionalGate."""
    q = cirq.NamedQubit("q")
    circuit = cirq.Circuit(cirq.S(q).with_classical_controls('c'))
    circuits_and_ops, _, _ = _cirq_to_circuits_and_ops(circuit, [*circuit.all_qubits()])
    assert len(circuits_and_ops) == 1
    pyzx_circuit = circuits_and_ops[0]
    assert isinstance(pyzx_circuit, zx.Circuit)
    cond_gates = [g for g in pyzx_circuit.gates if isinstance(g, ConditionalGate)]
    assert len(cond_gates) == 1
    assert cond_gates[0].condition_register == 'c'


def test_conditional_gate_preserved_by_transformer() -> None:
    """Test that a conditional gate round-trips through the transformer."""
    q = cirq.NamedQubit("q")
    circuit = cirq.Circuit(cirq.measure(q, key='c'), cirq.X(q).with_classical_controls('c'))
    result = zx_transformer(circuit)
    # The result should still contain a measurement and a classically controlled operation.
    measurement_ops = [op for moment in result for op in moment if cirq.is_measurement(op)]
    assert len(measurement_ops) == 1
    cond_ops = [
        op
        for moment in result
        for op in moment
        if isinstance(op, cirq.ClassicallyControlledOperation)
    ]
    assert len(cond_ops) == 1


def test_ccx_gate() -> None:
    """Test that CCX (Toffoli) gates are handled."""
    q = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.CCX(q[0], q[1], q[2]))
    _run_zxtransformer(circuit)


def test_cswap_gate() -> None:
    """Test that CSWAP (Fredkin) gates are handled."""
    q = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.CSWAP(q[0], q[1], q[2]))
    _run_zxtransformer(circuit)


def test_xx_zz_gates() -> None:
    """Test that XX and ZZ gates are handled."""
    q = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.XXPowGate(exponent=0.5)(q[0], q[1]), cirq.ZZPowGate(exponent=0.25)(q[0], q[1])
    )
    _run_zxtransformer(circuit)


def test_mid_circuit_measurement_teleportation() -> None:
    """Test a teleportation-like pattern with mid-circuit measurement and conditional gates."""
    q = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(q[0]),
        cirq.CX(q[0], q[1]),
        cirq.measure(q[0], key='m0'),
        cirq.X(q[1]).with_classical_controls('m0'),
    )
    result = zx_transformer(circuit)
    # Verify measurement and conditional gate are preserved.
    measurement_ops = [op for moment in result for op in moment if cirq.is_measurement(op)]
    assert len(measurement_ops) == 1
    assert cirq.measurement_key_name(measurement_ops[0]) == 'm0'
    cond_ops = [
        op
        for moment in result
        for op in moment
        if isinstance(op, cirq.ClassicallyControlledOperation)
    ]
    assert len(cond_ops) == 1


def test_hybrid_optimization_segments() -> None:
    """Test that unitary segments around a measurement are optimized independently."""
    q = cirq.LineQubit.range(2)
    # Build a circuit with redundant gates around a measurement.
    # H H = I, so these should cancel in each segment.
    circuit = cirq.Circuit(
        cirq.H(q[0]), cirq.H(q[0]), cirq.measure(q[0], key='m'), cirq.H(q[1]), cirq.H(q[1])
    )
    result = zx_transformer(circuit)
    # The measurement should still be present.
    measurement_ops = [op for moment in result for op in moment if cirq.is_measurement(op)]
    assert len(measurement_ops) == 1
    _run_zxtransformer_nonunitary(circuit)


def test_multiple_measurements() -> None:
    """Test a circuit with multiple measurements on different qubits."""
    q = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(q[0]), cirq.H(q[1]), cirq.measure(q[0], key='m0'), cirq.measure(q[1], key='m1')
    )
    result = zx_transformer(circuit)
    measurement_ops = [op for moment in result for op in moment if cirq.is_measurement(op)]
    assert len(measurement_ops) == 2
    keys = {cirq.measurement_key_name(op) for op in measurement_ops}
    assert keys == {'m0', 'm1'}
    _run_zxtransformer_nonunitary(circuit)


def test_empty_circuit() -> None:
    """Test that an empty circuit is handled gracefully."""
    circuit = cirq.Circuit()
    result = zx_transformer(circuit)
    assert len(result) == 0
