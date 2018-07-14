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
import cirq.google as cg
from cirq.google.eject_z import _try_get_known_z_half_turns


def assert_optimizes(before, after,
                     pre_opts=(cg.ConvertToXmonGates(ignore_failures=True),),
                     post_opts=(
                             cg.ConvertToXmonGates(ignore_failures=True),
                        cirq.DropEmptyMoments(),
                     )):
    opt = cg.EjectZ()

    for pre in pre_opts:
        pre.optimize_circuit(before)
    opt.optimize_circuit(before)
    for post in post_opts:
        post.optimize_circuit(before)
        post.optimize_circuit(after)

    if before != after:
        # coverage: ignore
        print(before)
        print(after)
    assert before == after

    # And it should be idempotent.
    opt.optimize_circuit(before)
    assert before == after


def _cancel_qubit_phase(m: np.ndarray, k: int) -> None:
    n = m.shape[0]
    b = 1 << k

    for t in [False, True]:
        best_pair = max([m[i, j]
                         for i in range(n)
                         for j in range(n)
                         if t == bool(i & b)],
                        key=abs)
        counter_phase = np.conj(best_pair) / abs(best_pair)
        for i in range(n):
            if t == bool(i & b):
                m[i, :] *= counter_phase


def canonicalize_up_to_measurement_phase(circuit: cirq.Circuit) -> np.ndarray:
    matrix = circuit.to_unitary_matrix()
    ordered_qubits = cirq.QubitOrder.DEFAULT.order_for(circuit.all_qubits())
    for moment in circuit:
        for op in moment.operations:
            if cirq.MeasurementGate.is_measurement(op):
                for q in op.qubits:
                    _cancel_qubit_phase(matrix, ordered_qubits.index(q))
    return matrix


def assert_removes_all_z_gates(circuit: cirq.Circuit):
    opt = cg.EjectZ()
    optimized = circuit.copy()
    opt.optimize_circuit(optimized)
    has_z = any(_try_get_known_z_half_turns(op) is not None
                for moment in optimized
                for op in moment.operations)
    m1 = canonicalize_up_to_measurement_phase(circuit)
    m2 = canonicalize_up_to_measurement_phase(optimized)
    similar = cirq.allclose_up_to_global_phase(m1, m2)

    if has_z or not similar:
        # coverage: ignore
        print("CIRCUIT")
        print(circuit)
        print("OPTIMIZED CIRCUIT")
        print(optimized)

    if not similar:
        # coverage: ignore
        print("CANONICALIZED CIRCUIT MATRIX")
        print(m1)
        print("CANONICALIZED OPTIMIZED CIRCUIT MATRIX")
        print(m2)

    assert similar and not has_z


def test_single_z_stays():
    q = cirq.QubitId()
    assert_optimizes(
        before=cirq.Circuit([
            cirq.Moment([cirq.Z(q)**0.5]),
        ]),
        after=cirq.Circuit([
            cirq.Moment([cirq.Z(q)**0.5]),
        ]))


def test_ignores_xz_and_cz():
    q1 = cirq.QubitId()
    q2 = cirq.QubitId()
    assert_optimizes(
        before=cirq.Circuit([
            cirq.Moment([cirq.X(q1)**0.5]),
            cirq.Moment([cirq.Y(q2)**0.5]),
            cirq.Moment([cirq.CZ(q1, q2)**0.25]),
            cirq.Moment([cirq.Y(q1)**0.5]),
            cirq.Moment([cirq.X(q2)**0.5]),
        ]),
        after=cirq.Circuit([
            cirq.Moment([cirq.X(q1)**0.5]),
            cirq.Moment([cirq.Y(q2)**0.5]),
            cirq.Moment([cirq.CZ(q1, q2)**0.25]),
            cirq.Moment([cirq.Y(q1)**0.5]),
            cirq.Moment([cirq.X(q2)**0.5]),
        ]))


def test_early_z_pushed_to_end():
    q = cirq.QubitId()
    assert_optimizes(
        before=cirq.Circuit([
            cirq.Moment([cirq.Z(q)**0.5]),
            cirq.Moment(),
            cirq.Moment(),
        ]),
        after=cirq.Circuit([
            cirq.Moment(),
            cirq.Moment(),
            cirq.Moment([cirq.Z(q)**0.5]),
        ]),
        pre_opts=[cg.ConvertToXmonGates(ignore_failures=True)],
        post_opts=[cg.ConvertToXmonGates(ignore_failures=True)])


def test_multi_z_merges():
    q = cirq.QubitId()
    assert_optimizes(
        before=cirq.Circuit([
            cirq.Moment([cirq.Z(q)**0.5]),
            cirq.Moment([cirq.Z(q)**0.25]),
        ]),
        after=cirq.Circuit([
            cirq.Moment(),
            cirq.Moment([cirq.Z(q)**0.75]),
        ]))


def test_z_pushes_past_xy_and_phases_it():
    q = cirq.QubitId()
    assert_optimizes(
        before=cirq.Circuit([
            cirq.Moment([cirq.Z(q)**0.5]),
            cirq.Moment([cirq.Y(q)**0.25]),
        ]),
        after=cirq.Circuit([
            cirq.Moment(),
            cirq.Moment([cirq.X(q)**0.25]),
            cirq.Moment([cirq.Z(q)**0.5]),
        ]))


def test_z_pushes_past_cz():
    q1 = cirq.QubitId()
    q2 = cirq.QubitId()
    assert_optimizes(
        before=cirq.Circuit([
            cirq.Moment([cirq.Z(q1)**0.5]),
            cirq.Moment([cirq.CZ(q1, q2)**0.25]),
        ]),
        after=cirq.Circuit([
            cirq.Moment(),
            cirq.Moment([cirq.CZ(q1, q2)**0.25]),
            cirq.Moment([cirq.Z(q1)**0.5]),
        ]))


def test_measurement_consumes_zs():
    q = cirq.QubitId()
    assert_optimizes(
        before=cirq.Circuit([
            cirq.Moment([cirq.Z(q)**0.5]),
            cirq.Moment([cirq.Z(q)**0.25]),
            cirq.Moment([cirq.measure(q)]),
        ]),
        after=cirq.Circuit([
            cirq.Moment(),
            cirq.Moment(),
            cirq.Moment([cirq.measure(q)]),
        ]))


def test_unphaseable_causes_earlier_merge_without_size_increase():
    class UnknownGate(cirq.Gate):
        pass

    u = UnknownGate()

    # pylint: disable=not-callable
    q = cirq.QubitId()
    assert_optimizes(
        before=cirq.Circuit([
            cirq.Moment([cirq.Z(q)]),
            cirq.Moment([u(q)]),
            cirq.Moment([cirq.Z(q)**0.5]),
            cirq.Moment([cirq.X(q)]),
            cirq.Moment([cirq.Z(q)**0.25]),
            cirq.Moment([cirq.X(q)]),
            cirq.Moment([u(q)]),
        ]),
        after=cirq.Circuit([
            cirq.Moment([cirq.Z(q)]),
            cirq.Moment([u(q)]),
            cirq.Moment(),
            cirq.Moment([cirq.Y(q)]),
            cirq.Moment([cirq.Z(q)**0.75]),
            cirq.Moment([cirq.X(q)]),  # Note: wasn't phased.
            cirq.Moment([u(q)]),
        ]))


def test_symbols_block():
    q = cirq.QubitId()
    assert_optimizes(
        before=cirq.Circuit([
            cirq.Moment([cg.ExpZGate(half_turns=1)(q)]),
            cirq.Moment([cg.ExpZGate(
                half_turns=cirq.Symbol('a'))(q)]),
            cirq.Moment([cg.ExpZGate(half_turns=0.25)(q)]),
        ]),
        after=cirq.Circuit([
            cirq.Moment([cg.ExpZGate(half_turns=1)(q)]),
            cirq.Moment([cg.ExpZGate(
                half_turns=cirq.Symbol('a'))(q)]),
            cirq.Moment([cg.ExpZGate(half_turns=0.25)(q)]),
        ]))


def test_removes_zs():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    assert_removes_all_z_gates(cirq.Circuit.from_ops(
        cirq.Z(a),
        cirq.measure(a)))

    assert_removes_all_z_gates(cirq.Circuit.from_ops(
        cirq.Z(a),
        cirq.measure(a, b)))

    assert_removes_all_z_gates(cirq.Circuit.from_ops(
        cirq.Z(a),
        cirq.Z(a),
        cirq.measure(a)))

    assert_removes_all_z_gates(cirq.Circuit.from_ops(
        cirq.Z(a),
        cirq.google.XmonMeasurementGate('k').on(a)))

    assert_removes_all_z_gates(cirq.Circuit.from_ops(
        cirq.google.ExpZGate().on(a),
        cirq.measure(a)))

    assert_removes_all_z_gates(cirq.Circuit.from_ops(
        cirq.Z(a),
        cirq.google.ExpZGate().on(a),
        cirq.measure(a)))

    assert_removes_all_z_gates(cirq.Circuit.from_ops(
        cirq.Z(a),
        cirq.google.ExpWGate().on(a),
        cirq.measure(a)))

    assert_removes_all_z_gates(cirq.Circuit.from_ops(
        cirq.Z(a),
        cirq.google.ExpWGate().on(a),
        cirq.google.ExpWGate().on(a),
        cirq.measure(a)))

    assert_removes_all_z_gates(cirq.Circuit.from_ops(
        cirq.Z(a),
        cirq.Z(b),
        cirq.CZ(a, b),
        cirq.google.Exp11Gate().on(a, b),
        cirq.measure(a, b)))


def test_unknown_operation_blocks():
    q = cirq.NamedQubit('q')

    class UnknownOp(cirq.Operation):
        @property
        def qubits(self):
            return [q]

        def with_qubits(self, *new_qubits):
            raise NotImplementedError()

    u = UnknownOp()

    assert_optimizes(
        before=cirq.Circuit([
            cirq.Moment([cg.ExpZGate(half_turns=1)(q)]),
            cirq.Moment([u]),
        ]),
        after=cirq.Circuit([
            cirq.Moment([cg.ExpZGate(half_turns=1)(q)]),
            cirq.Moment([u]),
        ]))
