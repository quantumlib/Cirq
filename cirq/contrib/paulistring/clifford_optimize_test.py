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
    clifford_optimized_circuit,
)


def test_optimize():
    q0, q1 = cirq.LineQubit.range(2)
    c_orig = cirq.Circuit(
        cirq.X(q1)**0.5,
        cirq.CZ(q0, q1),
        cirq.Z(q0)**0.25,
        cirq.X(q1)**0.25,
        cirq.CZ(q0, q1),
        cirq.X(q1)**-0.5,
    )
    c_expected = converted_gate_set(
        cirq.Circuit(
            cirq.CZ(q0, q1),
            cirq.Z(q0)**0.25,
            cirq.X(q1)**0.25,
            cirq.CZ(q0, q1),
        ))

    c_opt = clifford_optimized_circuit(c_orig)

    cirq.testing.assert_allclose_up_to_global_phase(
        c_orig.unitary(),
        c_opt.unitary(),
        atol=1e-7,
    )

    assert c_opt == c_expected

    cirq.testing.assert_has_diagram(c_opt, """
0: ───@───[Z]^0.25───@───
      │              │
1: ───@───[X]^0.25───@───
""")


def test_remove_czs():
    q0, q1 = cirq.LineQubit.range(2)
    c_orig = cirq.Circuit(
        cirq.CZ(q0, q1),
        cirq.Z(q0)**0.5,
        cirq.CZ(q0, q1),
    )
    c_expected = converted_gate_set(cirq.Circuit(cirq.Z(q0)**0.5,))

    c_opt = clifford_optimized_circuit(c_orig)

    cirq.testing.assert_allclose_up_to_global_phase(
        c_orig.unitary(),
        c_opt.unitary(qubits_that_should_be_present=(q0, q1)),
        atol=1e-7,
    )

    assert c_opt == c_expected

    cirq.testing.assert_has_diagram(c_opt, """
0: ───Z^0.5───
""")


def test_remove_staggered_czs():
    q0, q1, q2 = cirq.LineQubit.range(3)
    c_orig = cirq.Circuit(
        cirq.CZ(q0, q1),
        cirq.CZ(q1, q2),
        cirq.CZ(q0, q1),
    )
    c_expected = converted_gate_set(cirq.Circuit(cirq.CZ(q1, q2),))

    c_opt = clifford_optimized_circuit(c_orig)

    cirq.testing.assert_allclose_up_to_global_phase(
        c_orig.unitary(),
        c_opt.unitary(qubits_that_should_be_present=(q0, q1, q2)),
        atol=1e-7,
    )

    assert c_opt == c_expected

    cirq.testing.assert_has_diagram(c_opt, """
1: ───@───
      │
2: ───@───
""")


def test_with_measurements():
    q0, q1 = cirq.LineQubit.range(2)
    c_orig = cirq.Circuit(
        cirq.X(q0),
        cirq.CZ(q0, q1),
        cirq.measure(q0, q1, key='m'),
    )
    c_expected = converted_gate_set(
        cirq.Circuit(
            cirq.CZ(q0, q1),
            cirq.X(q0),
            cirq.Z(q1),
            cirq.measure(q0, q1, key='m'),
        ))

    c_opt = clifford_optimized_circuit(c_orig)

    cirq.testing.assert_allclose_up_to_global_phase(
        c_orig.unitary(),
        c_opt.unitary(),
        atol=1e-7,
    )

    assert c_opt == c_expected

    cirq.testing.assert_has_diagram(c_opt, """
0: ───@───X───M('m')───
      │       │
1: ───@───Z───M────────
""")


def test_optimize_large_circuit():
    q0, q1, q2 = cirq.LineQubit.range(3)
    c_orig = cirq.testing.nonoptimal_toffoli_circuit(q0, q1, q2)

    c_opt = clifford_optimized_circuit(c_orig)

    cirq.testing.assert_allclose_up_to_global_phase(
        c_orig.unitary(),
        c_opt.unitary(),
        atol=1e-7,
    )
