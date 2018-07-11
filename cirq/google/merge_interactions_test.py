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
from cirq.google import ExpZGate, MergeInteractions, MergeRotations
from cirq.value import Symbol


def assert_optimizes(before, after):
    opt = MergeInteractions()
    opt.optimize_circuit(before)

    # Ignore differences that would be caught by follow-up optimizations.
    followup_optimizations = [
        MergeRotations(),
        cirq.DropNegligible(),
        cirq.DropEmptyMoments()
    ]
    for post in followup_optimizations:
        post.optimize_circuit(before)
        post.optimize_circuit(after)

    if before != after:
        # coverage: ignore
        print('ACTUAL')
        print(before)
        print('EXPECTED')
        print(after)
    assert before == after


def assert_optimization_not_broken(circuit):
    """Check that the unitary matrix for the input circuit is the same (up to
    global phase and rounding error) as the unitary matrix of the optimized
    circuit."""
    u_before = circuit.to_unitary_matrix()
    MergeInteractions().optimize_circuit(circuit)
    u_after = circuit.to_unitary_matrix()

    cirq.testing.assert_allclose_up_to_global_phase(
        u_before, u_after, atol=1e-8)


def test_clears_paired_cnot():
    q0 = cirq.QubitId()
    q1 = cirq.QubitId()
    assert_optimizes(
        before=cirq.Circuit([
            [cirq.CNOT(q0, q1)],
            [cirq.CNOT(q0, q1)],
        ]),
        after=cirq.Circuit())


def test_ignores_czs_separated_by_parameterized():
    q0 = cirq.QubitId()
    q1 = cirq.QubitId()
    assert_optimizes(
        before=cirq.Circuit([
            [cirq.CZ(q0, q1)],
            [ExpZGate(half_turns=Symbol('boo'))(q0)],
            [cirq.CZ(q0, q1)],
        ]),
        after=cirq.Circuit([
            [cirq.CZ(q0, q1)],
            [ExpZGate(half_turns=Symbol('boo'))(q0)],
            [cirq.CZ(q0, q1)],
        ]))


def test_ignores_czs_separated_by_outer_cz():
    q00 = cirq.QubitId()
    q01 = cirq.QubitId()
    q10 = cirq.QubitId()
    assert_optimizes(
        before=cirq.Circuit([
            [cirq.CZ(q00, q01)],
            [cirq.CZ(q00, q10)],
            [cirq.CZ(q00, q01)],
        ]),
        after=cirq.Circuit([
            [cirq.CZ(q00, q01)],
            [cirq.CZ(q00, q10)],
            [cirq.CZ(q00, q01)],
        ]))


def test_cnots_separated_by_single_gates_correct():
    q0 = cirq.QubitId()
    q1 = cirq.QubitId()
    assert_optimization_not_broken(
        cirq.Circuit.from_ops(
            cirq.CNOT(q0, q1),
            cirq.H(q1),
            cirq.CNOT(q0, q1),
        ))


def test_czs_separated_by_single_gates_correct():
    q0 = cirq.QubitId()
    q1 = cirq.QubitId()
    assert_optimization_not_broken(
        cirq.Circuit.from_ops(
            cirq.CZ(q0, q1),
            cirq.X(q1),
            cirq.X(q1),
            cirq.X(q1),
            cirq.CZ(q0, q1),
        ))


def test_inefficient_circuit_correct():
    t = 0.1
    v = 0.11
    q0 = cirq.QubitId()
    q1 = cirq.QubitId()
    assert_optimization_not_broken(
        cirq.Circuit.from_ops(
            cirq.H(q1),
            cirq.CNOT(q0, q1),
            cirq.H(q1),
            cirq.CNOT(q0, q1),
            cirq.CNOT(q1, q0),
            cirq.H(q0),
            cirq.CNOT(q0, q1),
            cirq.Z(q0)**t, cirq.Z(q1)**-t,
            cirq.CNOT(q0, q1),
            cirq.H(q0), cirq.Z(q1)**v,
            cirq.CNOT(q0, q1),
            cirq.Z(q0)**-v, cirq.Z(q1)**-v,
        ))
