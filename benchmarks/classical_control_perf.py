# Copyright 2026 The Cirq Developers
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

"""Benchmarks for classically controlled operations.

These benchmarks keep the qubit register size small and constant while scaling
the number of sequential measure-and-control rounds. This avoids exponential
state-vector growth during simulation and ensures that moment-placement key
tracking, classical datastore updates, condition resolution, and measurement
deferral dominate the measured runtime.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import sympy

import cirq

ROUNDS = (8, 32, 128, pytest.param(1000, marks=pytest.mark.slow))


def _make_key_condition_circuit(
    num_rounds: int, *, reuse_key: bool = False, alternate_qubits: bool = False
) -> cirq.Circuit:
    """Builds `num_rounds` sequential measure-and-control rounds.

    When `reuse_key=True` and `alternate_qubits=True`, alternating the measured
    qubit between `q0` and `q1` while controlling `q2` ensures that every round
    after the first exercises all three classical key-conflict checks in
    `_PlacementCache` (`mkey_indices` and `ckey_indices` for measurement keys,
    and `mkey_indices` for control keys) without being masked by same-qubit
    conflicts. When `alternate_qubits=False`, both `reuse_key=False` and
    `reuse_key=True` produce identical 2-qubit circuit layouts on `q0` and `q2`.
    """
    q0, q1, q2 = cirq.LineQubit.range(3)
    ops: list[cirq.Operation] = []
    for i in range(num_rounds):
        q_meas = q0 if (not alternate_qubits or i % 2 == 0) else q1
        key = 'a' if reuse_key else f'a{i}'
        ops.extend(
            (cirq.H(q_meas), cirq.measure(q_meas, key=key), cirq.X(q2).with_classical_controls(key))
        )
    ops.append(cirq.measure(q2, key='target'))
    return cirq.Circuit(ops)


def _make_bitmask_condition_circuit(num_controls: int = 4) -> cirq.Circuit:
    """Builds a bitmask-conditioned circuit over `num_controls` control qubits."""
    controls = cirq.LineQubit.range(num_controls)
    target = cirq.LineQubit(num_controls)
    key = cirq.MeasurementKey('c')
    cond = cirq.BitMaskKeyCondition.create_equal_mask(key, bitmask=(1 << num_controls) - 1)
    return cirq.Circuit(
        cirq.H.on_each(controls),
        cirq.measure(*controls, key=key),
        cirq.X(target).with_classical_controls(cond),
        cirq.measure(target, key='target'),
    )


def _make_sympy_fields(num_rounds: int) -> list[sympy.Basic]:
    """Builds SymPy condition expressions with both `Symbol` and `Indexed` free symbols."""
    expressions: list[sympy.Basic] = []
    for i in range(num_rounds):
        b = sympy.IndexedBase(f'a{i}')
        expressions.append(sympy.Xor(b[0], b[1]))
    return expressions


def _make_sympy_condition_circuit(
    num_rounds: int, expressions: Sequence[sympy.Basic] | None = None
) -> cirq.Circuit:
    """Builds `num_rounds` sequential SymPy-conditioned rounds on 3 qubits."""
    q0, q1, q2 = cirq.LineQubit.range(3)
    if expressions is None:
        expressions = _make_sympy_fields(num_rounds)
    ops: list[cirq.Operation] = []
    for i, expr in enumerate(expressions):
        ops.extend(
            (
                cirq.H(q0 if i % 2 == 0 else q1),
                cirq.measure(q0, q1, key=f'a{i}'),
                cirq.X(q2).with_classical_controls(expr),
            )
        )
    ops.append(cirq.measure(q2, key='target'))
    return cirq.Circuit(ops)


def _make_repeat_until_circuit(num_rounds: int) -> cirq.Circuit:
    """Builds `num_rounds` sequential `repeat_until` subcircuit loops on 2 qubits."""
    q0, q1 = cirq.LineQubit.range(2)
    key = cirq.MeasurementKey('a')
    subcircuit = cirq.FrozenCircuit(
        cirq.H(q0), cirq.measure(q0, key=key), cirq.X(q1).with_classical_controls(key)
    )
    loop_op = cirq.CircuitOperation(subcircuit, repeat_until=cirq.KeyCondition(key))
    return cirq.Circuit([loop_op for _ in range(num_rounds)], cirq.measure(q1, key='target'))


def _append_controls_uncached(
    base_circuit: cirq.Circuit, appended_ops: Sequence[cirq.Operation]
) -> cirq.Circuit:
    """Appends classically controlled operations onto a circuit without `_PlacementCache`."""
    circuit = base_circuit.copy()
    circuit.append(appended_ops)
    return circuit


def _unroll_repetition_ids(
    q0: cirq.Qid, q1: cirq.Qid, subcircuit: cirq.FrozenCircuit, num_repetitions: int
) -> cirq.Circuit:
    """Wraps a repeated subcircuit in a `CircuitOperation` and unrolls it deeply."""
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a'),
        cirq.CircuitOperation(subcircuit, repetitions=num_repetitions, use_repetition_ids=True),
        cirq.X(q1).with_classical_controls('a'),
    )
    return cirq.CircuitOperation(cirq.FrozenCircuit(circuit)).mapped_circuit(deep=True)


class TestCircuitConstruction:
    """Benchmarks for building classically controlled circuits and scheduling moments."""

    group = "classical_control_construction"

    @pytest.mark.parametrize("num_rounds", ROUNDS)
    @pytest.mark.benchmark(group=group)
    def test_construct_key_condition(self, benchmark, num_rounds: int) -> None:
        circuit = benchmark(
            _make_key_condition_circuit, num_rounds, reuse_key=True, alternate_qubits=True
        )
        assert len(circuit) == 2 * num_rounds + 2

    @pytest.mark.parametrize("num_rounds", ROUNDS)
    @pytest.mark.benchmark(group=group)
    def test_construct_sympy_condition(self, benchmark, num_rounds: int) -> None:
        # Pre-build SymPy expressions so the benchmark measures Cirq's condition
        # wrapping and moment placement rather than SymPy symbol parsing.
        expressions = _make_sympy_fields(num_rounds)
        circuit = benchmark(_make_sympy_condition_circuit, num_rounds, expressions)
        assert len(circuit) == 2 * num_rounds + 2

    @pytest.mark.parametrize("num_rounds", ROUNDS)
    @pytest.mark.benchmark(group=group)
    def test_earliest_available_moment_key_condition(self, benchmark, num_rounds: int) -> None:
        q0, q1, q2 = cirq.LineQubit.range(3)
        base_circuit = cirq.Circuit(
            (cirq.H(q0), cirq.measure(q0, key=f'a{i}')) for i in range(num_rounds)
        )
        # Alternating (control on q1, measure on q2) and (measure on q1, control
        # on q2) ensures `earliest_available_moment` resolves every placement via
        # its three classical key-conflict checks rather than qubit conflicts.
        appended_ops: list[cirq.Operation] = []
        for i in range(num_rounds):
            key = f'a{i}'
            if i % 2 == 0:
                appended_ops.extend(
                    (cirq.X(q1).with_classical_controls(key), cirq.measure(q2, key=key))
                )
            else:
                appended_ops.extend(
                    (cirq.measure(q1, key=key), cirq.X(q2).with_classical_controls(key))
                )
        circuit = benchmark(_append_controls_uncached, base_circuit, appended_ops)
        assert len(circuit) == 2 * num_rounds + 2


class TestSimulation:
    """Benchmarks for simulating classically controlled circuits."""

    group = "classical_control_simulation"
    repetitions = 20

    @pytest.mark.parametrize("num_rounds", ROUNDS)
    @pytest.mark.benchmark(group=group)
    def test_run_key_condition(self, benchmark, num_rounds: int) -> None:
        circuit = _make_key_condition_circuit(num_rounds, reuse_key=False)
        simulator = cirq.Simulator(seed=1234, split_untangled_states=False)
        result = benchmark(simulator.run, circuit, repetitions=self.repetitions)
        assert result.measurements['target'].shape == (self.repetitions, 1)

    @pytest.mark.parametrize("num_rounds", ROUNDS)
    @pytest.mark.benchmark(group=group)
    def test_run_repeated_key_condition(self, benchmark, num_rounds: int) -> None:
        circuit = _make_key_condition_circuit(num_rounds, reuse_key=True)
        simulator = cirq.Simulator(seed=1234, split_untangled_states=False)
        result = benchmark(simulator.run, circuit, repetitions=self.repetitions)
        assert result.records['a'].shape == (self.repetitions, num_rounds, 1)

    @pytest.mark.parametrize("num_rounds", ROUNDS)
    @pytest.mark.benchmark(group=group)
    def test_run_sympy_condition(self, benchmark, num_rounds: int) -> None:
        circuit = _make_sympy_condition_circuit(num_rounds)
        simulator = cirq.Simulator(seed=1234, split_untangled_states=False)
        result = benchmark(simulator.run, circuit, repetitions=self.repetitions)
        assert result.measurements['target'].shape == (self.repetitions, 1)

    @pytest.mark.parametrize("num_rounds", ROUNDS)
    @pytest.mark.benchmark(group=group)
    def test_run_repeat_until(self, benchmark, num_rounds: int) -> None:
        circuit = _make_repeat_until_circuit(num_rounds)
        simulator = cirq.Simulator(seed=1234, split_untangled_states=False)
        result = benchmark(simulator.run, circuit, repetitions=self.repetitions)
        assert result.records['a'].shape[0] == self.repetitions
        assert result.records['a'].shape[1] >= num_rounds


class TestTransformations:
    """Benchmarks for transforming classically controlled circuits."""

    group = "classical_control_transform"

    @pytest.mark.parametrize("num_rounds", ROUNDS)
    @pytest.mark.benchmark(group=group)
    def test_defer_measurements_rounds(self, benchmark, num_rounds: int) -> None:
        circuit = _make_key_condition_circuit(num_rounds)
        deferred = benchmark(cirq.defer_measurements, circuit)
        assert len(deferred.all_qubits()) == num_rounds + 2

    @pytest.mark.parametrize("num_controls", [4, 8, pytest.param(12, marks=pytest.mark.slow)])
    @pytest.mark.benchmark(group=group)
    def test_defer_measurements_bitmask(self, benchmark, num_controls: int) -> None:
        circuit = _make_bitmask_condition_circuit(num_controls=num_controls)
        deferred = benchmark(cirq.defer_measurements, circuit)
        assert len(deferred.all_qubits()) == 2 * num_controls + 1

    @pytest.mark.parametrize("num_repetitions", ROUNDS)
    @pytest.mark.benchmark(group=group)
    def test_unroll_repetition_ids(self, benchmark, num_repetitions: int) -> None:
        q0, q1 = cirq.LineQubit.range(2)
        subcircuit = cirq.FrozenCircuit(
            cirq.measure(q0, key='a'),
            cirq.X(q1).with_classical_controls('a'),
            cirq.Z(q1).with_classical_controls(sympy.Eq(sympy.Symbol('a'), 1)),
        )
        unrolled = benchmark(_unroll_repetition_ids, q0, q1, subcircuit, num_repetitions)
        measurements = list(unrolled.findall_operations_with_gate_type(cirq.MeasurementGate))
        assert len(measurements) == num_repetitions + 1
