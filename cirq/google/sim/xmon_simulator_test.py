# Copyright 2018 Google LLC
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

"""Tests for xmon_simulator."""

import cmath
import itertools
import math
from typing import Sequence

import numpy as np
import pytest


from cirq.circuits import Circuit
from cirq.devices import UnconstrainedDevice
from cirq.google import (
    ExpWGate, ExpZGate, Exp11Gate, XmonMeasurementGate, XmonQubit,
)
from cirq.google.sim import xmon_simulator
from cirq.ops import op_tree
from cirq.ops import raw_types
from cirq.ops.common_gates import CNOT, H, X, Z
from cirq.ops.gate_features import CompositeGate, SingleQubitGate
from cirq.schedules import moment_by_moment_schedule
from cirq.study.resolver import ParamResolver
from cirq.study.sweeps import Linspace
from cirq.value import Symbol

Q1 = XmonQubit(0, 0)
Q2 = XmonQubit(1, 0)
Q3 = XmonQubit(2, 0)


def basic_circuit():
    sqrt_x = ExpWGate(half_turns=0.5, axis_half_turns=0.0)
    z = ExpZGate()
    cz = Exp11Gate()
    circuit = Circuit()
    circuit.append(
        [sqrt_x(Q1), sqrt_x(Q2),
         cz(Q1, Q2),
         sqrt_x(Q1), sqrt_x(Q2),
         z(Q1)])
    return circuit


def large_circuit():
    np.random.seed(0)
    qubits = [XmonQubit(i, 0) for i in range(10)]
    sqrt_x = ExpWGate(half_turns=0.5, axis_half_turns=0.0)
    cz = Exp11Gate()
    circuit = Circuit()
    for _ in range(11):
        circuit.append(
            [sqrt_x(qubit) for qubit in qubits if np.random.random() < 0.5])
        circuit.append([cz(qubits[i], qubits[i + 1]) for i in range(9)])
    circuit.append([XmonMeasurementGate(key='meas')(*qubits)])
    return circuit


def test_xmon_options_negative_num_shards():
    with pytest.raises(AssertionError):
        xmon_simulator.Options(num_shards=-1)


def test_xmon_options_negative_min_qubits_before_shard():
    with pytest.raises(AssertionError):
        xmon_simulator.Options(min_qubits_before_shard=-1)


def test_xmon_options():
    options = xmon_simulator.Options(num_shards=3, min_qubits_before_shard=0)
    assert options.num_prefix_qubits == 1
    assert options.min_qubits_before_shard == 0


def run(simulator, circuit, scheduler, **kw):
    if scheduler is None:
        program = circuit
    else:
        program = scheduler(UnconstrainedDevice, circuit)
    return simulator.run(program, **kw)


SCHEDULERS = [None, moment_by_moment_schedule]


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_run_no_results(scheduler):
    simulator = xmon_simulator.Simulator()
    result = run(simulator, basic_circuit(), scheduler)
    assert len(result.measurements) == 0


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_run(scheduler):
    np.random.seed(0)
    circuit = basic_circuit()
    circuit.append(
        [XmonMeasurementGate(key='a')(Q1),
         XmonMeasurementGate(key='b')(Q2),])

    simulator = xmon_simulator.Simulator()
    result = run(simulator, circuit, scheduler)
    assert result.measurements == {'a': [False], 'b': [False]}


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_run_state(scheduler):
    simulator = xmon_simulator.Simulator()
    result = run(simulator, basic_circuit(), scheduler, qubits=[Q1, Q2])
    np.testing.assert_almost_equal(result.final_states[0],
                                   np.array([-0.5j, 0.5, -0.5, 0.5j]))


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_run_state_different_order_of_qubits(scheduler):
    simulator = xmon_simulator.Simulator()
    result = run(simulator, basic_circuit(), scheduler, qubits=[Q2, Q1])
    np.testing.assert_almost_equal(result.final_states[0],
                                   np.array([-0.5j, -0.5, 0.5, 0.5j]))


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_consistent_seeded_run_sharded(scheduler):
    circuit = large_circuit()

    simulator = xmon_simulator.Simulator()
    result = run(simulator, circuit, scheduler)
    np.testing.assert_equal(
        result.measurements['meas'],
        [[True, False, False, True, False, False, True, False, False, False]])


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_consistent_seeded_run_no_sharding(scheduler):
    circuit = large_circuit()

    simulator = xmon_simulator.Simulator()
    result = run(simulator,
                 circuit,
                 scheduler,
                 options=xmon_simulator.Options(num_shards=1))
    np.testing.assert_equal(
        result.measurements['meas'],
        [[True, False, False, True, False, False, True, False, False, False]])

@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_run_no_sharing_few_qubits(scheduler):
    np.random.seed(0)
    circuit = basic_circuit()
    circuit.append(
        [XmonMeasurementGate(key='a')(Q1),
         XmonMeasurementGate(key='b')(Q2),])

    simulator = xmon_simulator.Simulator()
    options = xmon_simulator.Options(min_qubits_before_shard=0)
    result = run(simulator, circuit, scheduler, options=options)
    np.testing.assert_equal(result.measurements['a'], [[False]])
    np.testing.assert_equal(result.measurements['b'], [[False]])


def test_moment_steps_no_results():
    simulator = xmon_simulator.Simulator()
    for step in simulator.moment_steps(basic_circuit()):
        assert len(step.measurements) == 0


def test_moment_steps():
    np.random.seed(0)
    circuit = basic_circuit()
    circuit.append(
        [XmonMeasurementGate(key='a')(Q1),
         XmonMeasurementGate(key='b')(Q2),])

    simulator = xmon_simulator.Simulator()
    results = []
    for step in simulator.moment_steps(circuit):
        results.append(step)
    expected = [{}, {}, {}, {}, {'a': [False], 'b': [False]}]
    assert len(results) == len(expected)
    assert all(a.measurements == b for a, b in zip(results, expected))


def test_moment_steps_state():
    np.random.seed(0)
    circuit = basic_circuit()

    simulator = xmon_simulator.Simulator()
    results = []
    for step in simulator.moment_steps(circuit, qubits=[Q1, Q2]):
        results.append(step.state())
    np.testing.assert_almost_equal(results,
                                   np.array([[0.5, 0.5j, 0.5j, -0.5],
                                             [0.5, 0.5j, 0.5j, 0.5],
                                             [-0.5, 0.5j, 0.5j, -0.5],
                                             [-0.5j, 0.5, -0.5, 0.5j]]))


def test_moment_steps_set_state():
    np.random.seed(0)
    circuit = basic_circuit()

    simulator = xmon_simulator.Simulator()
    step = simulator.moment_steps(circuit, qubits=[Q1, Q2])

    result = next(step)
    result.set_state(0)
    np.testing.assert_almost_equal(result.state(), np.array([1, 0, 0, 0]))


def test_moment_steps_set_state_2():
    np.random.seed(0)
    circuit = basic_circuit()

    simulator = xmon_simulator.Simulator()
    step = simulator.moment_steps(circuit, qubits=[Q1, Q2])

    result = next(step)
    result.set_state(np.array([1j, 0, 0, 0], dtype=np.complex64))
    np.testing.assert_almost_equal(result.state(),
                                   np.array([1j, 0, 0, 0], dtype=np.complex64))


def compute_gate(circuit, resolver, num_qubits=1):
    simulator = xmon_simulator.Simulator()
    gate = []
    for initial_state in range(1 << num_qubits):
        result = simulator.run(circuit,
                               initial_state=initial_state,
                               param_resolver=resolver)
        gate.append(result.final_states[0])
    return np.array(gate).transpose()


def test_param_resolver_exp_w_half_turns():
    exp_w = ExpWGate(
        half_turns=Symbol('a'),
        axis_half_turns=0.0)
    circuit = Circuit()
    circuit.append(exp_w(Q1))
    resolver = ParamResolver({'a': 0.5})
    result = compute_gate(circuit, resolver)
    amp = 1.0 / math.sqrt(2)
    np.testing.assert_almost_equal(result,
                                   np.array([[amp, amp * 1j],
                                             [amp * 1j, amp]]))


def test_param_resolver_exp_w_axis_half_turns():
    exp_w = ExpWGate(
        half_turns=1.0, axis_half_turns=Symbol('a'))
    circuit = Circuit()
    circuit.append(exp_w(Q1))
    resolver = ParamResolver({'a': 0.5})
    result = compute_gate(circuit, resolver)
    np.testing.assert_almost_equal(result,
                                   np.array([[0, 1],
                                             [-1, 0]]))


def test_param_resolver_exp_w_multiple_params():
    exp_w = ExpWGate(
        half_turns=Symbol('a'),
        axis_half_turns=Symbol('b'))
    circuit = Circuit()
    circuit.append(exp_w(Q1))
    resolver = ParamResolver({'a': 0.5, 'b': 0.5})
    result = compute_gate(circuit, resolver)
    amp = 1.0 / math.sqrt(2)
    np.testing.assert_almost_equal(result,
                                   np.array([[amp, amp],
                                             [-amp, amp]]))


def test_param_resolver_exp_z_half_turns():
    exp_z = ExpZGate(half_turns=Symbol('a'))
    circuit = Circuit()
    circuit.append(exp_z(Q1))
    resolver = ParamResolver({'a': 0.5})
    result = compute_gate(circuit, resolver)
    np.testing.assert_almost_equal(
        result,
        np.array([[cmath.exp(1j * math.pi * 0.25), 0],
                  [0, cmath.exp(-1j * math.pi * 0.25)]]))


def test_param_resolver_exp_11_half_turns():
    exp_11 = Exp11Gate(half_turns=Symbol('a'))
    circuit = Circuit()
    circuit.append(exp_11(Q1, Q2))
    resolver = ParamResolver({'a': 0.5})
    result = compute_gate(circuit, resolver, num_qubits=2)
    # Slight hack: doesn't depend on order of qubits.
    np.testing.assert_almost_equal(
        result,
        np.diag([1, 1, 1, cmath.exp(1j * math.pi * 0.5)]))


def test_param_resolver_param_dict():
    exp_w = ExpWGate(
        half_turns=Symbol('a'),
        axis_half_turns=0.0)
    circuit = Circuit()
    circuit.append(exp_w(Q1))
    resolver = ParamResolver({'a': 0.5})

    simulator = xmon_simulator.Simulator()
    result = simulator.run(circuit, resolver)
    assert result.params.param_dict == {'a': 0.5}


def test_run_circuit_sweep():
    circuit = Circuit.from_ops(
        ExpWGate(half_turns=Symbol('a')).on(Q1),
        XmonMeasurementGate('m').on(Q1),
    )

    sweep = Linspace('a', 0, 10, 11)
    simulator = xmon_simulator.Simulator()

    for i, result in enumerate(
                        simulator.run_sweep(circuit, sweep, repetitions=1)):
        assert result.params['a'] == i
        assert result.measurements['m'] == [i % 2 != 0]


def test_run_circuit_sweeps():
    circuit = Circuit.from_ops(
        ExpWGate(half_turns=Symbol('a')).on(Q1),
        XmonMeasurementGate('m').on(Q1),
    )

    sweep = Linspace('a', 0, 5, 6)
    sweep2 = Linspace('a', 6, 10, 5)
    simulator = xmon_simulator.Simulator()

    for i, result in enumerate(
                        simulator.run_sweep(circuit, [sweep, sweep2],
                                            repetitions=1)):
        assert result.params['a'] == i
        assert result.measurements['m'] == [i % 2 != 0]


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_composite_gates(scheduler):
    circuit = Circuit()
    circuit.append([X(Q1), CNOT(Q1, Q2)])
    m = XmonMeasurementGate('a')
    circuit.append([m(Q1, Q2)])

    simulator = xmon_simulator.Simulator()
    result = run(simulator, circuit, scheduler)
    np.testing.assert_equal(result.measurements['a'], [[True, True]])


class UnsupportedGate(SingleQubitGate):

    def matrix(self) -> np.ndarray:
        return np.ndarray([[1, 0], [0, 1j]])

    def __repr__(self):
        return "UnsupportedGate"


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_unsupported_gate(scheduler):
    circuit = Circuit()
    gate = UnsupportedGate()
    circuit.append([H(Q1), gate(Q2)])

    simulator = xmon_simulator.Simulator()
    with pytest.raises(TypeError, msg="UnsupportedGate"):
        _ = run(simulator, circuit, scheduler)


class UnsupportedCompositeGate(SingleQubitGate, CompositeGate):

    def matrix(self) -> np.ndarray:
        return np.ndarray([[1, 0], [0, -1j]])

    def __repr__(self):
        return "UnsupportedCompositeGate"

    def default_decompose(
        self, qubits: Sequence[raw_types.QubitId]) -> op_tree.OP_TREE:
        qubit = qubits[0]
        yield Z(qubit)
        yield UnsupportedGate().on(qubit)


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_unsupported_gate_composite(scheduler):
    circuit = Circuit()
    gate = UnsupportedGate()
    circuit.append([H(Q1), gate(Q2)])

    simulator = xmon_simulator.Simulator()
    with pytest.raises(TypeError, msg="UnsupportedGate"):
        _ = run(simulator, circuit, scheduler)


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_measurement_qubit_order(scheduler):
    circuit = Circuit()
    meas = XmonMeasurementGate()
    circuit.append(X(Q2))
    circuit.append(X(Q1))
    circuit.append([meas.on(Q1, Q3, Q2)])
    simulator = xmon_simulator.Simulator()
    result = run(simulator, circuit, scheduler)
    np.testing.assert_equal(result.measurements[''], [[True, False, True]])


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_inverted_measurement(scheduler):
    circuit = Circuit.from_ops(
        XmonMeasurementGate('a', invert_mask=(False,))(Q1),
        X(Q1),
        XmonMeasurementGate('b', invert_mask=(False,))(Q1),
        XmonMeasurementGate('c', invert_mask=(True,))(Q1),
        X(Q1),
        XmonMeasurementGate('d', invert_mask=(True,))(Q1))
    simulator = xmon_simulator.Simulator()
    result = run(simulator, circuit, scheduler)
    assert {'a': [[False]], 'b': [[True]], 'c': [[False]],
            'd': [[True]]} == result.measurements


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_inverted_measurement_multiple_qubits(scheduler):
    circuit = Circuit.from_ops(
        XmonMeasurementGate('a', invert_mask=(False, True))(Q1, Q2),
        XmonMeasurementGate('b', invert_mask=(True, False))(Q1, Q2),
        XmonMeasurementGate('c', invert_mask=(True, False))(Q2, Q1))
    simulator = xmon_simulator.Simulator()
    result = run(simulator, circuit, scheduler)
    np.testing.assert_equal(result.measurements['a'], [[False, True]])
    np.testing.assert_equal(result.measurements['b'], [[True, False]])
    np.testing.assert_equal(result.measurements['c'], [[True, False]])


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_measurement_multiple_measurements(scheduler):
    circuit = Circuit()
    measa = XmonMeasurementGate('a')
    measb = XmonMeasurementGate('b')
    circuit.append(X(Q1))
    circuit.append([measa.on(Q1, Q2)])
    circuit.append(X(Q1))
    circuit.append([measb.on(Q1, Q2)])
    simulator = xmon_simulator.Simulator()
    result = run(simulator, circuit, scheduler)
    np.testing.assert_equal(result.measurements['a'], [[True, False]])
    np.testing.assert_equal(result.measurements['b'], [[False, False]])


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_measurement_multiple_measurements_qubit_order(scheduler):
    circuit = Circuit()
    measa = XmonMeasurementGate('a')
    measb = XmonMeasurementGate('b')
    circuit.append(X(Q1))
    circuit.append([measa.on(Q1, Q2)])
    circuit.append([measb.on(Q2, Q1)])
    simulator = xmon_simulator.Simulator()
    result = run(simulator, circuit, scheduler)
    np.testing.assert_equal(result.measurements['a'], [[True, False]])
    np.testing.assert_equal(result.measurements['b'], [[False, True]])


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_measurement_keys_repeat(scheduler):
    circuit = Circuit()
    meas = XmonMeasurementGate('a')
    circuit.append([meas.on(Q1), X.on(Q1), X.on(Q2), meas.on(Q2)])
    simulator = xmon_simulator.Simulator()
    with pytest.raises(ValueError, message='Repeated Measurement key a'):
        run(simulator, circuit, scheduler)


def bit_flip_circuit(flip0, flip1):
    q1, q2 = XmonQubit(0, 0), XmonQubit(0, 1)
    g1, g2 = ExpWGate(half_turns=flip0)(q1), ExpWGate(half_turns=flip1)(q2)
    m1, m2 = XmonMeasurementGate('q1')(q1), XmonMeasurementGate('q2')(q2)
    circuit = Circuit()
    circuit.append([g1, g2, m1, m2])
    return circuit


def test_circuit_repetitions():
    sim = xmon_simulator.Simulator()
    circuit = bit_flip_circuit(1, 1)

    result = sim.run(circuit, repetitions=10)
    assert result.params.param_dict == {}
    assert result.repetitions == 10
    np.testing.assert_equal(result.measurements['q1'], [[True]] * 10)
    np.testing.assert_equal(result.measurements['q2'], [[True]] * 10)


def test_circuit_parameters():
    sim = xmon_simulator.Simulator()
    circuit = bit_flip_circuit(Symbol('a'), Symbol('b'))

    resolvers = [ParamResolver({'a': b1, 'b': b2})
                 for b1 in range(2) for b2 in range(2)]

    all_trials = sim.run_sweep(circuit, params=resolvers, repetitions=1)
    assert len(all_trials) == 4
    for result in all_trials:
        assert result.repetitions == 1
        expect_a = result.params['a'] == 1
        expect_b = result.params['b'] == 1
        np.testing.assert_equal(result.measurements['q1'], [[expect_a]])
        np.testing.assert_equal(result.measurements['q2'], [[expect_b]])
    # All parameters explored.
    assert (set(itertools.product([0, 1], [0, 1]))
            == {(r.params['a'], r.params['b']) for r in all_trials})


def test_circuit_param_and_reps():
    sim = xmon_simulator.Simulator()
    circuit = bit_flip_circuit(Symbol('a'), Symbol('b'))

    resolvers = [ParamResolver({'a': b1, 'b': b2})
                 for b1 in range(2) for b2 in range(2)]

    all_trials = sim.run_sweep(circuit, params=resolvers, repetitions=3)
    assert len(all_trials) == 4
    for result in all_trials:
        assert result.repetitions == 3
        expect_a = result.params['a'] == 1
        expect_b = result.params['b'] == 1
        np.testing.assert_equal(result.measurements['q1'], [[expect_a]] * 3)
        np.testing.assert_equal(result.measurements['q2'], [[expect_b]] * 3)
    # All parameters explored.
    # All parameters explored.
    assert (set(itertools.product([0, 1], [0, 1]))
            == {(r.params['a'], r.params['b']) for r in all_trials})
