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
import math

import numpy as np
import pytest

from cirq.circuits import Circuit
from cirq.devices import UnconstrainedDevice
from cirq.google import (
    ExpWGate, ExpZGate, Exp11Gate, XmonMeasurementGate, XmonQubit,
)
from cirq.google import (ParameterizedValue)
from cirq.ops.common_gates import CNOT, X
from cirq.schedules import moment_by_moment_schedule
from cirq.sim.google import xmon_simulator
from cirq.study.resolver import ParamResolver

Q1 = XmonQubit(0, 0)
Q2 = XmonQubit(1, 0)


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
    for i in range(10):
        circuit.append(
            XmonMeasurementGate(key='meas')(qubits[i]))
    return circuit, qubits


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


def assert_empty_context(context):
    assert xmon_simulator.TrialContext(param_dict={}) == context


def run(simulator, circuit, scheduler, **kw):
    if scheduler is None:
        return simulator.run(circuit, **kw)
    else:
        schedule = scheduler(UnconstrainedDevice, circuit)
        return simulator.run_schedule(schedule, **kw)


SCHEDULERS = [None, moment_by_moment_schedule]


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_run_no_results(scheduler):
    simulator = xmon_simulator.Simulator()
    context, result = run(simulator, basic_circuit(), scheduler)
    assert len(result.measurements) == 0
    assert_empty_context(context)


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_run(scheduler):
    np.random.seed(0)
    circuit = basic_circuit()
    circuit.append(
        [XmonMeasurementGate(key='a')(Q1),
         XmonMeasurementGate(key='b')(Q2),])

    simulator = xmon_simulator.Simulator()
    context, result = run(simulator, circuit, scheduler)
    assert result.measurements == {'a': [False], 'b': [False]}
    assert_empty_context(context)


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_run_state(scheduler):
    simulator = xmon_simulator.Simulator()
    context, result = run(simulator, basic_circuit(), scheduler, qubits=[Q1, Q2])
    np.testing.assert_almost_equal(result.final_state,
                                   np.array([-0.5j, 0.5, -0.5, 0.5j]))
    assert_empty_context(context)


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_run_state_different_order_of_qubits(scheduler):
    simulator = xmon_simulator.Simulator()
    context, result = run(simulator, basic_circuit(), scheduler, qubits=[Q2, Q1])
    np.testing.assert_almost_equal(result.final_state,
                                   np.array([-0.5j, -0.5, 0.5, 0.5j]))
    assert_empty_context(context)


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_run_sharded(scheduler):
    circuit, qubits = large_circuit()

    simulator = xmon_simulator.Simulator()
    context, result = run(simulator, circuit, scheduler, qubits=qubits)
    assert result.measurements == {
        'meas': [False, False, False, True, False, False, True, False, False,
                 True]}
    assert_empty_context(context)


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_run_no_sharding(scheduler):
    circuit, qubits = large_circuit()

    simulator = xmon_simulator.Simulator()
    context, result = run(simulator,
                          circuit,
                          scheduler,
                          options=xmon_simulator.Options(num_shards=1),
                          qubits=qubits)
    assert result.measurements == {
        'meas': [False, False, False, True, False, False, True, False, False,
                 True]}


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_run_no_sharing_few_qubits(scheduler):
    np.random.seed(0)
    circuit = basic_circuit()
    circuit.append(
        [XmonMeasurementGate(key='a')(Q1),
         XmonMeasurementGate(key='b')(Q2),])

    simulator = xmon_simulator.Simulator()
    options = xmon_simulator.Options(min_qubits_before_shard=0)
    context, result = run(simulator, circuit, scheduler, options=options)
    assert result.measurements == {'a': [False], 'b': [False]}
    assert_empty_context(context)


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

    result = step.__next__()
    result.set_state(0)
    np.testing.assert_almost_equal(result.state(), np.array([1, 0, 0, 0]))


def compute_gate(circuit, resolver, num_qubits=1):
    simulator = xmon_simulator.Simulator()
    gate = []
    for initial_state in range(1 << num_qubits):
        _, result = simulator.run(circuit, initial_state=initial_state,
                                       param_resolver=resolver)
        gate.append(result.final_state)
    return np.array(gate).transpose()


@pytest.mark.parametrize('offset', (0.0, 0.2))
def test_param_resolver_exp_w_half_turns(offset):
    exp_w = ExpWGate(
        half_turns=ParameterizedValue('a', offset),
        axis_half_turns=0.0)
    circuit = Circuit()
    circuit.append(exp_w(Q1))
    resolver = ParamResolver({'a': 0.5 - offset})
    result = compute_gate(circuit, resolver)
    amp = 1.0 / math.sqrt(2)
    np.testing.assert_almost_equal(result,
                                   np.array([[amp, amp * 1j],
                                             [amp * 1j, amp]]))


@pytest.mark.parametrize('offset', (0.0, 0.2))
def test_param_resolver_exp_w_axis_half_turns(offset):
    exp_w = ExpWGate(
        half_turns=1.0, axis_half_turns=ParameterizedValue('a', offset))
    circuit = Circuit()
    circuit.append(exp_w(Q1))
    resolver = ParamResolver({'a': 0.5 - offset})
    result = compute_gate(circuit, resolver)
    np.testing.assert_almost_equal(result,
                                   np.array([[0, 1],
                                             [-1, 0]]))


@pytest.mark.parametrize('offset', (0.0, 0.2))
def test_param_resolver_exp_w_multiple_params(offset):
    exp_w = ExpWGate(
        half_turns=ParameterizedValue('a', offset),
        axis_half_turns=ParameterizedValue('b', offset))
    circuit = Circuit()
    circuit.append(exp_w(Q1))
    resolver = ParamResolver({'a': 0.5 - offset, 'b': 0.5 - offset})
    result = compute_gate(circuit, resolver)
    amp = 1.0 / math.sqrt(2)
    np.testing.assert_almost_equal(result,
                                   np.array([[amp, amp],
                                             [-amp, amp]]))


@pytest.mark.parametrize('offset', (0.0, 0.2))
def test_param_resolver_exp_z_half_turns(offset):
    exp_z = ExpZGate(half_turns=ParameterizedValue('a', offset))
    circuit = Circuit()
    circuit.append(exp_z(Q1))
    resolver = ParamResolver({'a': 0.5 - offset})
    result = compute_gate(circuit, resolver)
    np.testing.assert_almost_equal(
        result,
        np.array([[cmath.exp(1j * math.pi * 0.25), 0],
                  [0, cmath.exp(-1j * math.pi * 0.25)]]))


@pytest.mark.parametrize('offset', (0.0, 0.2))
def test_param_resolver_exp_11_half_turns(offset):
    exp_11 = Exp11Gate(half_turns=ParameterizedValue('a', offset))
    circuit = Circuit()
    circuit.append(exp_11(Q1, Q2))
    resolver = ParamResolver({'a': 0.5 - offset})
    result = compute_gate(circuit, resolver, num_qubits=2)
    # Slight hack: doesn't depend on order of qubits.
    np.testing.assert_almost_equal(
        result,
        np.diag([1, 1, 1, cmath.exp(1j * math.pi * 0.5)]))


@pytest.mark.parametrize('offset', (0.0, 0.2))
def test_param_resolver_param_dict(offset):
    exp_w = ExpWGate(
        half_turns=ParameterizedValue('a', offset),
        axis_half_turns=0.0)
    circuit = Circuit()
    circuit.append(exp_w(Q1))
    resolver = ParamResolver({'a': 0.5})

    simulator = xmon_simulator.Simulator()
    context, result = simulator.run(circuit, param_resolver=resolver)
    assert context.param_dict == {'a': 0.5}


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_composite_gates(scheduler):
    circuit = Circuit()
    circuit.append([X(Q1), CNOT(Q1, Q2)])
    m = XmonMeasurementGate('a')
    circuit.append([m(Q1), m(Q2)])

    simulator = xmon_simulator.Simulator()
    context, result = run(simulator, circuit, scheduler)
    assert result.measurements['a'] == [True, True]
