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

from cirq.contrib.paulistring import (
    converted_gate_set,
    optimized_circuit,
)


def test_optimize():
    q0, q1, q2 = cirq.LineQubit.range(3)
    c_orig = cirq.Circuit.from_ops(
        cirq.X(q0) ** 0.5,
        cirq.X(q1),
        cirq.CZ(q1, q2),
        cirq.X(q2) ** 0.125,
        cirq.Z(q1) ** 0.5,
        cirq.Y(q1) ** 0.5,
        cirq.CZ(q0, q1),
        cirq.Z(q1) ** 0.5,
        cirq.CZ(q1, q2),
        cirq.Z(q1) ** 0.5,
        cirq.X(q2) ** 0.0625,
        cirq.CZ(q1, q2),
        cirq.X(q2) ** 0.125,
    )
    assert c_orig.to_text_diagram() == """
0: ───X^0.5─────────────────────────@────────────────────────────────────
                                    │
1: ───X───────@───S─────────Y^0.5───@───S───@───S──────────@─────────────
              │                             │              │
2: ───────────@───X^0.125───────────────────@───X^0.0625───@───X^0.125───
""".strip()

    c_opt = optimized_circuit(c_orig)

    cirq.testing.assert_allclose_up_to_global_phase(
        c_orig.to_unitary_matrix(),
        c_opt.to_unitary_matrix(),
        atol=1e-7,
    )

    assert c_opt.to_text_diagram() == """
0: ───X^0.5────────────@─────────────────────────────────────────
                       │
1: ───@───────X^-0.5───@───@─────────────────@───Z^-0.5──────────
      │                    │                 │
2: ───@────────────────────@───[X]^-0.0625───@───[X]^-0.25───Z───
""".strip()


def test_optimize_large_circuit():
    q0, q1, q2 = cirq.LineQubit.range(3)
    c_orig = cirq.testing.nonoptimal_toffoli_circuit(q0, q1, q2)

    c_opt = optimized_circuit(c_orig)

    cirq.testing.assert_allclose_up_to_global_phase(
        c_orig.to_unitary_matrix(),
        c_opt.to_unitary_matrix(),
        atol=1e-7,
    )

    assert sum(1 for op in c_opt.all_operations()
                 if isinstance(op, cirq.GateOperation)
                    and isinstance(op.gate, cirq.Rot11Gate)) == 10
