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

from typing import Iterable

import cirq
from cirq.optimizers.eject_z import _try_get_known_z_half_turns


def assert_optimizes(before: cirq.Circuit,
                     expected: cirq.Circuit,
                     post_opts: Iterable[cirq.OptimizationPass] = (
                             cirq.DropEmptyMoments(),
                     )):
    opt = cirq.EjectZ()

    if cirq.has_unitary(before):
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
            before, expected, atol=1e-8)

    circuit = before.copy()
    opt.optimize_circuit(circuit)
    for post in post_opts:
        post.optimize_circuit(circuit)
        post.optimize_circuit(expected)

    cirq.testing.assert_same_circuits(circuit, expected)

    # And it should be idempotent.
    opt.optimize_circuit(circuit)
    cirq.testing.assert_same_circuits(circuit, expected)


def assert_removes_all_z_gates(circuit: cirq.Circuit):
    opt = cirq.EjectZ()
    optimized = circuit.copy()
    opt.optimize_circuit(optimized)
    has_z = any(_try_get_known_z_half_turns(op) is not None
                for moment in optimized
                for op in moment.operations)
    assert not has_z

    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        circuit,
        optimized,
        atol=1e-8)


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
        ]))


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
            cirq.Moment([cirq.PhasedXPowGate(phase_exponent=-0.75).on(q)]),
            cirq.Moment([cirq.Z(q)**0.75]),
            cirq.Moment([u(q)]),
        ]))


def test_symbols_block():
    q = cirq.NamedQubit('q')
    assert_optimizes(
        before=cirq.Circuit([
            cirq.Moment([cirq.Z(q)]),
            cirq.Moment([cirq.Z(q)**cirq.Symbol('a')]),
            cirq.Moment([cirq.Z(q)**0.25]),
        ]),
        expected=cirq.Circuit([
            cirq.Moment([cirq.Z(q)**cirq.Symbol('a')]),
            cirq.Moment([cirq.Z(q)**1.25]),
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
        cirq.measure(a, key='k')))

    assert_removes_all_z_gates(cirq.Circuit.from_ops(
        cirq.Z(a),
        cirq.X(a),
        cirq.measure(a)))

    assert_removes_all_z_gates(cirq.Circuit.from_ops(
        cirq.Z(a),
        cirq.X(a),
        cirq.X(a),
        cirq.measure(a)))

    assert_removes_all_z_gates(cirq.Circuit.from_ops(
        cirq.Z(a),
        cirq.Z(b),
        cirq.CZ(a, b),
        cirq.CZ(a, b),
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
            cirq.Moment([cirq.Z(q)]),
            cirq.Moment([u]),
        ]),
        expected=cirq.Circuit([
            cirq.Moment([cirq.Z(q)]),
            cirq.Moment([u]),
        ]))
