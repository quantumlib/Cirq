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

from typing import Optional, Callable

import cirq
import pyzx as zx

from cirq.contrib.zxtransformer.zxtransformer import zx_transformer, _cirq_to_circuits_and_ops


def _run_zxtransformer(
    qc: cirq.Circuit, optimizer: Optional[Callable[[zx.Circuit], zx.Circuit]] = None
) -> None:
    zx_qc = zx_transformer(qc) if optimizer is None else zx_transformer(qc, optimizer=optimizer)
    qubit_map = {qid: qid for qid in qc.all_qubits()}
    cirq.testing.assert_circuits_have_same_unitary_given_final_permutation(qc, zx_qc, qubit_map)


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
    circuit = cirq.Circuit(cirq.ry(0.5)(q),
                           cirq.rz(0.5)(q))
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


def test_measurement() -> None:
    """Test a circuit with a measurement."""
    q = cirq.NamedQubit("q")
    circuit = cirq.Circuit(cirq.H(q), cirq.measure(q, key='c'), cirq.H(q))
    circuits_and_ops = _cirq_to_circuits_and_ops(circuit, [*circuit.all_qubits()])
    assert len(circuits_and_ops) == 3
    assert circuits_and_ops[1] == cirq.measure(q, key='c')


def test_conditional_gate() -> None:
    """Test a circuit with a conditional gate."""
    q = cirq.NamedQubit("q")
    circuit = cirq.Circuit(cirq.X(q), cirq.H(q).with_classical_controls('c'), cirq.X(q))
    circuits_and_ops = _cirq_to_circuits_and_ops(circuit, [*circuit.all_qubits()])
    assert len(circuits_and_ops) == 3
