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

import cirq


def test_nonoptimal_toffoli_circuit():
    q0, q1, q2 = cirq.LineQubit.range(3)
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.testing.nonoptimal_toffoli_circuit(q0, q1, q2).to_unitary_matrix(),
        cirq.unitary(cirq.TOFFOLI(q0, q1, q2)),
        atol=1e-7)
