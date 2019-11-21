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

import cirq
import cirq.testing as ct
from cirq.testing import consistent_qasm as cq
from cirq.contrib.qasm_import import circuit_from_qasm


def test_consistency_with_qasm_output_and_qiskit():
    qubits = [cirq.NamedQubit('q_{}'.format(i)) for i in range(4)]
    a, b, c, d = qubits
    circuit1 = cirq.Circuit(
        cirq.rx(np.pi / 2).on(a),
        cirq.ry(np.pi / 2).on(b),
        cirq.rz(np.pi / 2).on(b),
        cirq.X.on(a),
        cirq.Y.on(b),
        cirq.Z.on(c),
        cirq.H.on(d),
        cirq.S.on(a),
        cirq.T.on(b),
        cirq.S.on(c)**-1,
        cirq.T.on(d)**-1,
        cirq.X.on(d)**0.125,
        cirq.TOFFOLI.on(a, b, c),
        cirq.CSWAP.on(d, a, b),
        cirq.SWAP.on(c, d),
        cirq.CX.on(a, b),
        cirq.ControlledGate(cirq.Y).on(c, d),
        cirq.CZ.on(a, b),
        cirq.ControlledGate(cirq.H).on(b, c),
        cirq.IdentityGate(1).on(c),
        cirq.circuits.qasm_output.QasmUGate(1.0, 2.0, 3.0).on(d),
    )

    qasm = cirq.qasm(circuit1)

    circuit2 = circuit_from_qasm(qasm)

    cirq_unitary = cirq.unitary(circuit2)
    ct.assert_allclose_up_to_global_phase(cirq_unitary,
                                          cirq.unitary(circuit1),
                                          atol=1e-8)

    cq.assert_qiskit_parsed_qasm_consistent_with_unitary(qasm, cirq_unitary)
