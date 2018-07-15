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
import cirq.google as cg


def assert_optimizes(before: cirq.Circuit, expected: cirq.Circuit):
    actual = cirq.Circuit(before)
    opt = cg.MergeInteractions()
    opt.optimize_circuit(actual)

    # Ignore differences that would be caught by follow-up optimizations.
    followup_optimizations = [
        cg.MergeRotations(),
        cirq.DropNegligible(),
        cirq.DropEmptyMoments()
    ]
    for post in followup_optimizations:
        post.optimize_circuit(actual)
        post.optimize_circuit(expected)

    if actual != expected:
        # coverage: ignore
        print('ACTUAL')
        print(actual)
        print('EXPECTED')
        print(expected)
    assert actual == expected


def assert_optimization_not_broken(circuit):
    """Check that the unitary matrix for the input circuit is the same (up to
    global phase and rounding error) as the unitary matrix of the optimized
    circuit."""
    u_before = circuit.to_unitary_matrix()
    cg.MergeInteractions().optimize_circuit(circuit)
    u_after = circuit.to_unitary_matrix()

    cirq.testing.assert_allclose_up_to_global_phase(
        u_before, u_after, atol=1e-8)


# def test_clears_paired_cnot():
#     q0 = cirq.LineQubit(0)
#     q1 = cirq.LineQubit(1)
#     assert_optimizes(
#         before=cirq.Circuit([
#             cirq.Moment([cirq.CNOT(q0, q1)]),
#             cirq.Moment([cirq.CNOT(q0, q1)]),
#         ]),
#         expected=cirq.Circuit())
#
#
# def test_ignores_czs_separated_by_parameterized():
#     q0 = cirq.LineQubit(0)
#     q1 = cirq.LineQubit(1)
#     assert_optimizes(
#         before=cirq.Circuit([
#             cirq.Moment([cirq.CZ(q0, q1)]),
#             cirq.Moment([cg.ExpZGate(
#                 half_turns=cirq.Symbol('boo'))(q0)]),
#             cirq.Moment([cirq.CZ(q0, q1)]),
#         ]),
#         expected=cirq.Circuit([
#             cirq.Moment([cirq.CZ(q0, q1)]),
#             cirq.Moment([cg.ExpZGate(
#                 half_turns=cirq.Symbol('boo'))(q0)]),
#             cirq.Moment([cirq.CZ(q0, q1)]),
#         ]))
#
#
# def test_ignores_czs_separated_by_outer_cz():
#     q00 = cirq.GridQubit(0, 0)
#     q01 = cirq.GridQubit(0, 1)
#     q10 = cirq.GridQubit(1, 0)
#     assert_optimizes(
#         before=cirq.Circuit([
#             cirq.Moment([cirq.CZ(q00, q01)]),
#             cirq.Moment([cirq.CZ(q00, q10)]),
#             cirq.Moment([cirq.CZ(q00, q01)]),
#         ]),
#         expected=cirq.Circuit([
#             cirq.Moment([cirq.CZ(q00, q01)]),
#             cirq.Moment([cirq.CZ(q00, q10)]),
#             cirq.Moment([cirq.CZ(q00, q01)]),
#         ]))
#
#
# def test_cnots_separated_by_single_gates_correct():
#     q0 = cirq.LineQubit(0)
#     q1 = cirq.LineQubit(1)
#     assert_optimization_not_broken(
#         cirq.Circuit.from_ops(
#             cirq.CNOT(q0, q1),
#             cirq.H(q1),
#             cirq.CNOT(q0, q1),
#         ))
#
#
# def test_czs_separated_by_single_gates_correct():
#     q0 = cirq.LineQubit(0)
#     q1 = cirq.LineQubit(1)
#     assert_optimization_not_broken(
#         cirq.Circuit.from_ops(
#             cirq.CZ(q0, q1),
#             cirq.X(q1),
#             cirq.X(q1),
#             cirq.X(q1),
#             cirq.CZ(q0, q1),
#         ))
#
#
# def test_inefficient_circuit_correct():
#     t = 0.1
#     v = 0.11
#     q0 = cirq.LineQubit(0)
#     q1 = cirq.LineQubit(1)
#     assert_optimization_not_broken(
#         cirq.Circuit.from_ops(
#             cirq.H(q1),
#             cirq.CNOT(q0, q1),
#             cirq.H(q1),
#             cirq.CNOT(q0, q1),
#             cirq.CNOT(q1, q0),
#             cirq.H(q0),
#             cirq.CNOT(q0, q1),
#             cirq.Z(q0)**t, cirq.Z(q1)**-t,
#             cirq.CNOT(q0, q1),
#             cirq.H(q0), cirq.Z(q1)**v,
#             cirq.CNOT(q0, q1),
#             cirq.Z(q0)**-v, cirq.Z(q1)**-v,
#         ))


def test_swap_field():
    circuit = cirq.Circuit.from_ops(
        cirq.ISWAP(cirq.LineQubit(j), cirq.LineQubit(j + 1))
        for i in range(4)
        for j in range(i % 2, 9, 2)
    )

    print("BEFORE")
    print(circuit)

    cirq.google.MergeInteractions().optimize_circuit(circuit)
    # print("AFTER")
    # print(repr(circuit))
    # print(circuit)
    # cirq.google.MergeRotations().optimize_circuit(circuit)
    # cirq.google.EjectZ().optimize_circuit(circuit)
    # cirq.DropNegligible().optimize_circuit(circuit)
    # circuit = circuit.from_ops(circuit.all_operations(),
    #                            strategy=cirq.InsertStrategy.EARLIEST)
    #
    #
    #
    assert False
