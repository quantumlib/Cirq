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


# TODO as we grow the language, we'll add more complex examples here
def test_consistency_with_qasm_output():
    a, b, c, d = [cirq.NamedQubit('q_{}'.format(i)) for i in range(4)]
    circuit1 = cirq.Circuit.from_ops(
        cirq.Rx(np.pi / 2).on(a),
        cirq.Ry(np.pi / 2).on(b),
        cirq.Rz(np.pi / 2).on(b),
        cirq.IdentityGate(1).on(c),
        cirq.circuits.qasm_output.QasmUGate(1.0, 2.0, 3.0).on(d),
    )

    qasm1 = cirq.qasm(circuit1)

    circuit2 = cirq.contrib.qasm_import.qasm.QasmCircuitParser().parse(qasm1)
    ct.assert_same_circuits(circuit1, circuit2)
