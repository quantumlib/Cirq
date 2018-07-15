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
from typing import Iterable, Tuple, Optional

import numpy as np

import cirq
import cirq.google as cg
from cirq.google.eject_z import _try_get_known_z_half_turns


def assert_optimizes(before: cirq.Circuit,
                     expected: cirq.Circuit,
                     pre_opts: Iterable[cirq.OptimizationPass] = (
                             cg.ConvertToXmonGates(ignore_failures=True),),
                     post_opts: Iterable[cirq.OptimizationPass] = (
                             cg.ConvertToXmonGates(ignore_failures=True),
                             cirq.DropEmptyMoments())):
    opt = cg.EjectZ()

    circuit = before.copy()
    for pre in pre_opts:
        pre.optimize_circuit(circuit)
    opt.optimize_circuit(circuit)
    for post in post_opts:
        post.optimize_circuit(circuit)
        post.optimize_circuit(expected)

    if circuit != expected:
        # coverage: ignore
        print("BEFORE")
        print(before)
        print("AFTER")
        print(circuit)
        print("EXPECTED")
        print(expected)
    assert circuit == expected

    # And it should be idempotent.
    opt.optimize_circuit(circuit)
    assert circuit == expected


def _cancel_qubit_phase(m1: np.ndarray, m2: np.ndarray, k: int) -> None:
    n = m1.shape[0]
    b = 1 << k

    for t in [False, True]:
        best_loc = max([(i, j)
                        for i in range(n)
                        for j in range(n)
                        if t == bool(i & b)],
                       key=lambda e: max(abs(m1[e]), abs(m2[e])))
        best_val_1 = m1[best_loc]
        best_val_2 = m2[best_loc]
        counter_phase_1 = np.conj(best_val_1) / abs(best_val_1)
        counter_phase_2 = np.conj(best_val_2) / abs(best_val_2)
        for i in range(n):
            if t == bool(i & b):
                m1[i, :] *= counter_phase_1
                m2[i, :] *= counter_phase_2


def canonicalize_up_to_measurement_phase(
        circuit1: cirq.Circuit,
        circuit2: cirq.Circuit) -> Tuple[Optional[np.ndarray],
                                         Optional[np.ndarray]]:
    ordered_qubits = cirq.QubitOrder.DEFAULT.order_for(circuit1.all_qubits())
    ordered_qubits_2 = cirq.QubitOrder.DEFAULT.order_for(circuit2.all_qubits())
    assert ordered_qubits == ordered_qubits_2
    assert circuit1.are_all_measurements_terminal()
    assert circuit2.are_all_measurements_terminal()

    terminal_1 = {q
                  for op in circuit1.all_operations()
                  if cirq.MeasurementGate.is_measurement(op)
                  for q in op.qubits}
    terminal_2 = {q
                  for op in circuit1.all_operations()
                  if cirq.MeasurementGate.is_measurement(op)
                  for q in op.qubits}
    assert terminal_1 == terminal_2

    matrix1, matrix2 = None, None
    try:
        matrix1 = circuit1.to_unitary_matrix()
    except TypeError:
        pass
    try:
        matrix2 = circuit2.to_unitary_matrix()
    except TypeError:
        pass
    assert (matrix1 is None) == (matrix2 is None)
    if matrix1 is None or matrix2 is None:
        return np.eye(1), np.eye(1)
    for q in terminal_1:
        _cancel_qubit_phase(matrix1, matrix2, ordered_qubits.index(q))
    return matrix1, matrix2


def is_same_circuit_up_to_measurement_phase(circuit1: cirq.Circuit,
                                            circuit2: cirq.Circuit,
                                            atol: float):
    m1, m2 = canonicalize_up_to_measurement_phase(circuit1, circuit2)
    if m1 is not None and m2 is not None:
        return cirq.allclose_up_to_global_phase(m1, m2, atol=atol)


def assert_equivalent_circuit_up_to_measurement_phase(actual: cirq.Circuit,
                                                      expected: cirq.Circuit,
                                                      atol: float):
    similar = is_same_circuit_up_to_measurement_phase(actual,
                                                      expected,
                                                      atol)
    if not similar:
        # coverage: ignore
        print("ACTUAL")
        print(actual)
        print("EXPECTED")
        print(expected)
    assert similar


def assert_removes_all_z_gates(circuit: cirq.Circuit):
    opt = cg.EjectZ()
    optimized = circuit.copy()
    opt.optimize_circuit(optimized)
    has_z = any(_try_get_known_z_half_turns(op) is not None
                for moment in optimized
                for op in moment.operations)
    similar = is_same_circuit_up_to_measurement_phase(circuit,
                                                      optimized,
                                                      atol=1e-8)

    if has_z or not similar:
        # coverage: ignore
        print("CIRCUIT")
        print(circuit)
        print("OPTIMIZED CIRCUIT")
        print(optimized)

    assert similar and not has_z


def test_single_z_stays():
    q = cirq.NamedQubit('q')
    assert_optimizes(
        before=cirq.Circuit([
            cirq.Moment([cirq.Z(q)**0.5]),
        ]),
        expected=cirq.Circuit([
            cirq.Moment([cirq.Z(q)**0.5]),
        ]))


def test_ignores_xz_and_cz():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    assert_optimizes(
        before=cirq.Circuit([
            cirq.Moment([cirq.X(a)**0.5]),
            cirq.Moment([cirq.Y(b)**0.5]),
            cirq.Moment([cirq.CZ(a, b)**0.25]),
            cirq.Moment([cirq.Y(a)**0.5]),
            cirq.Moment([cirq.X(b)**0.5]),
        ]),
        expected=cirq.Circuit([
            cirq.Moment([cirq.X(a)**0.5]),
            cirq.Moment([cirq.Y(b)**0.5]),
            cirq.Moment([cirq.CZ(a, b)**0.25]),
            cirq.Moment([cirq.Y(a)**0.5]),
            cirq.Moment([cirq.X(b)**0.5]),
        ]))


def test_early_z():
    q = cirq.NamedQubit('q')
    assert_optimizes(
        before=cirq.Circuit([
            cirq.Moment([cirq.Z(q)**0.5]),
            cirq.Moment(),
            cirq.Moment(),
        ]),
        expected=cirq.Circuit([
            cirq.Moment([cirq.Z(q)**0.5]),
            cirq.Moment(),
            cirq.Moment(),
        ]),
        pre_opts=[cg.ConvertToXmonGates(ignore_failures=True)],
        post_opts=[cg.ConvertToXmonGates(ignore_failures=True)])


def test_multi_z_merges():
    q = cirq.NamedQubit('q')
    assert_optimizes(
        before=cirq.Circuit([
            cirq.Moment([cirq.Z(q)**0.5]),
            cirq.Moment([cirq.Z(q)**0.25]),
        ]),
        expected=cirq.Circuit([
            cirq.Moment(),
            cirq.Moment([cirq.Z(q)**0.75]),
        ]))


def test_z_pushes_past_xy_and_phases_it():
    q = cirq.NamedQubit('q')
    assert_optimizes(
        before=cirq.Circuit([
            cirq.Moment([cirq.Z(q)**0.5]),
            cirq.Moment([cirq.Y(q)**0.25]),
        ]),
        expected=cirq.Circuit([
            cirq.Moment(),
            cirq.Moment([cirq.X(q)**0.25]),
            cirq.Moment([cirq.Z(q)**0.5]),
        ]))


def test_z_pushes_past_cz():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    assert_optimizes(
        before=cirq.Circuit([
            cirq.Moment([cirq.Z(a)**0.5]),
            cirq.Moment([cirq.CZ(a, b)**0.25]),
        ]),
        expected=cirq.Circuit([
            cirq.Moment(),
            cirq.Moment([cirq.CZ(a, b)**0.25]),
            cirq.Moment([cirq.Z(a)**0.5]),
        ]))


def test_measurement_consumes_zs():
    q = cirq.NamedQubit('q')
    assert_optimizes(
        before=cirq.Circuit([
            cirq.Moment([cirq.Z(q)**0.5]),
            cirq.Moment([cirq.Z(q)**0.25]),
            cirq.Moment([cirq.measure(q)]),
        ]),
        expected=cirq.Circuit([
            cirq.Moment(),
            cirq.Moment(),
            cirq.Moment([cirq.measure(q)]),
        ]))


def test_unphaseable_causes_earlier_merge_without_size_increase():
    class UnknownGate(cirq.Gate):
        pass

    u = UnknownGate()

    # pylint: disable=not-callable
    q = cirq.NamedQubit('q')
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
        expected=cirq.Circuit([
            cirq.Moment([cirq.Z(q)]),
            cirq.Moment([u(q)]),
            cirq.Moment(),
            cirq.Moment([cirq.Y(q)]),
            cirq.Moment([cg.ExpWGate(axis_half_turns=0.25).on(q)]),
            cirq.Moment([cirq.Z(q)**0.75]),
            cirq.Moment([u(q)]),
        ]))


def test_symbols_block():
    q = cirq.NamedQubit('q')
    assert_optimizes(
        before=cirq.Circuit([
            cirq.Moment([cg.ExpZGate(half_turns=1)(q)]),
            cirq.Moment([cg.ExpZGate(
                half_turns=cirq.Symbol('a'))(q)]),
            cirq.Moment([cg.ExpZGate(half_turns=0.25)(q)]),
        ]),
        expected=cirq.Circuit([
            cirq.Moment([cg.ExpZGate(
                half_turns=cirq.Symbol('a'))(q)]),
            cirq.Moment([cg.ExpZGate(half_turns=1.25)(q)]),
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
        expected=cirq.Circuit([
            cirq.Moment([cg.ExpZGate(half_turns=1)(q)]),
            cirq.Moment([u]),
        ]))
