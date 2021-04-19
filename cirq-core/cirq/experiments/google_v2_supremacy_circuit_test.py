# Copyright 2019 The Cirq Developers
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

from cirq import GridQubit
from cirq import ops
import cirq.experiments.google_v2_supremacy_circuit as supremacy_v2


def test_google_v2_supremacy_circuit():
    circuit = supremacy_v2.generate_boixo_2018_supremacy_circuits_v2_grid(
        n_rows=4, n_cols=5, cz_depth=9, seed=0
    )
    # We check that is exactly circuit inst_4x5_10_0
    # in github.com/sboixo/GRCS cz_v2
    assert len(circuit) == 11
    assert len(list(circuit.findall_operations_with_gate_type(ops.CZPowGate))) == 35
    assert len(list(circuit.findall_operations_with_gate_type(ops.XPowGate))) == 15
    assert len(list(circuit.findall_operations_with_gate_type(ops.YPowGate))) == 23
    assert len(list(circuit.findall_operations_with_gate_type(ops.ZPowGate))) == 32
    assert len(list(circuit.findall_operations_with_gate_type(ops.HPowGate))) == 40
    qubits = [GridQubit(i, j) for i in range(4) for j in range(5)]
    assert isinstance(circuit.operation_at(qubits[0], 2).gate, ops.YPowGate)
    assert isinstance(circuit.operation_at(qubits[1], 2).gate, ops.YPowGate)
    assert isinstance(circuit.operation_at(qubits[8], 2).gate, ops.XPowGate)
    assert circuit.operation_at(qubits[0], 1).gate == ops.CZ
    assert circuit.operation_at(qubits[5], 2).gate == ops.CZ
    assert circuit.operation_at(qubits[8], 3).gate == ops.CZ
    assert circuit.operation_at(qubits[13], 4).gate == ops.CZ
    assert circuit.operation_at(qubits[12], 5).gate == ops.CZ
    assert circuit.operation_at(qubits[13], 6).gate == ops.CZ
    assert circuit.operation_at(qubits[14], 7).gate == ops.CZ


def test_google_v2_supremacy_bristlecone():
    pytest.importorskip("cirq_google")
    # Check instance consistency
    c = supremacy_v2.generate_boixo_2018_supremacy_circuits_v2_bristlecone(
        n_rows=11, cz_depth=8, seed=0
    )
    assert len(c) == 10
    assert len(c.all_qubits()) == 70
    assert len(list(c.findall_operations_with_gate_type(ops.CZPowGate))) == 119
    assert len(list(c.findall_operations_with_gate_type(ops.XPowGate))) == 43
    assert len(list(c.findall_operations_with_gate_type(ops.YPowGate))) == 69
    assert isinstance(c.operation_at(GridQubit(2, 5), 2).gate, ops.YPowGate)
    assert isinstance(c.operation_at(GridQubit(3, 2), 2).gate, ops.XPowGate)
    assert isinstance(c.operation_at(GridQubit(1, 6), 3).gate, ops.XPowGate)
    # test smaller subgraph
    c = supremacy_v2.generate_boixo_2018_supremacy_circuits_v2_bristlecone(
        n_rows=9, cz_depth=8, seed=0
    )
    qubits = list(c.all_qubits())
    qubits.sort()
    assert len(qubits) == 48
    assert isinstance(c.operation_at(qubits[5], 2).gate, ops.YPowGate)
    assert isinstance(c.operation_at(qubits[7], 3).gate, ops.YPowGate)
    assert len(list(c.findall_operations_with_gate_type(ops.CZPowGate))) == 79
    assert len(list(c.findall_operations_with_gate_type(ops.XPowGate))) == 32


def test_n_rows_less_than_2():
    pytest.importorskip("cirq_google")
    with pytest.raises(AssertionError):
        supremacy_v2.generate_boixo_2018_supremacy_circuits_v2_bristlecone(
            n_rows=1, cz_depth=0, seed=0
        )
