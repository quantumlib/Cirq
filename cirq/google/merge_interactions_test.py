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

from cirq import testing
from cirq import circuits
from cirq import ops
from cirq.google import ExpZGate, MergeInteractions, MergeRotations
from cirq.value import Symbol


def assert_optimizes(before, after):
    opt = MergeInteractions()
    opt.optimize_circuit(before)

    # Ignore differences that would be caught by follow-up optimizations.
    followup_optimizations = [
        MergeRotations(),
        circuits.DropNegligible(),
        circuits.DropEmptyMoments()
    ]
    for post in followup_optimizations:
        post.optimize_circuit(before)
        post.optimize_circuit(after)

    if before != after:
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

    testing.assert_allclose_up_to_global_phase(u_before, u_after, atol=1e-8)


def test_clears_paired_cnot():
    q0 = ops.QubitId()
    q1 = ops.QubitId()
    assert_optimizes(
        before=circuits.Circuit([
            circuits.Moment([ops.CNOT(q0, q1)]),
            circuits.Moment([ops.CNOT(q0, q1)]),
        ]),
        after=circuits.Circuit())


def test_ignores_czs_separated_by_parameterized():
    q0 = ops.QubitId()
    q1 = ops.QubitId()
    assert_optimizes(
        before=circuits.Circuit([
            circuits.Moment([ops.CZ(q0, q1)]),
            circuits.Moment([ExpZGate(
                half_turns=Symbol('boo'))(q0)]),
            circuits.Moment([ops.CZ(q0, q1)]),
        ]),
        after=circuits.Circuit([
            circuits.Moment([ops.CZ(q0, q1)]),
            circuits.Moment([ExpZGate(
                half_turns=Symbol('boo'))(q0)]),
            circuits.Moment([ops.CZ(q0, q1)]),
        ]))


def test_ignores_czs_separated_by_outer_cz():
    q00 = ops.QubitId()
    q01 = ops.QubitId()
    q10 = ops.QubitId()
    assert_optimizes(
        before=circuits.Circuit([
            circuits.Moment([ops.CZ(q00, q01)]),
            circuits.Moment([ops.CZ(q00, q10)]),
            circuits.Moment([ops.CZ(q00, q01)]),
        ]),
        after=circuits.Circuit([
            circuits.Moment([ops.CZ(q00, q01)]),
            circuits.Moment([ops.CZ(q00, q10)]),
            circuits.Moment([ops.CZ(q00, q01)]),
        ]))


def test_cnots_separated_by_single_gates_correct():
    q0 = ops.QubitId()
    q1 = ops.QubitId()
    assert_optimization_not_broken(
        circuits.Circuit.from_ops(
            ops.CNOT(q0, q1),
            ops.H(q1),
            ops.CNOT(q0, q1),
        ))


def test_czs_separated_by_single_gates_correct():
    q0 = ops.QubitId()
    q1 = ops.QubitId()
    assert_optimization_not_broken(
        circuits.Circuit.from_ops(
            ops.CZ(q0, q1),
            ops.X(q1),
            ops.X(q1),
            ops.X(q1),
            ops.CZ(q0, q1),
        ))


def test_inefficient_circuit_correct():
    t = 0.1
    v = 0.11
    q0 = ops.QubitId()
    q1 = ops.QubitId()
    assert_optimization_not_broken(
        circuits.Circuit.from_ops(
            ops.H(q1),
            ops.CNOT(q0, q1),
            ops.H(q1),
            ops.CNOT(q0, q1),
            ops.CNOT(q1, q0),
            ops.H(q0),
            ops.CNOT(q0, q1),
            ops.Z(q0)**t, ops.Z(q1)**-t,
            ops.CNOT(q0, q1),
            ops.H(q0), ops.Z(q1)**v,
            ops.CNOT(q0, q1),
            ops.Z(q0)**-v, ops.Z(q1)**-v,
        ))
