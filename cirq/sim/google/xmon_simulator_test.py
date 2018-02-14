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

from cirq import circuits
from cirq import ops
from cirq.google import ExpWGate, ExpZGate, Exp11Gate, ParameterizedValue, \
    XmonMeasurementGate
from cirq.google.resolver import ParamResolver
from cirq.sim.google import xmon_simulator

Q1 = ops.QubitLoc(0, 0)
Q2 = ops.QubitLoc(1, 0)


def basic_circuit():
    sqrt_x = ExpWGate(half_turns=0.5, axis_half_turns=0.0)
    z = ExpZGate()
    cz = Exp11Gate()
    circuit = circuits.Circuit()
    circuit.append(
        [sqrt_x(Q1), sqrt_x(Q2),
         cz(Q1, Q2),
         sqrt_x(Q1), sqrt_x(Q2),
         z(Q1)])
    return circuit


def large_circuit():
    np.random.seed(0)
    qubits = [ops.QubitLoc(i, 0) for i in range(10)]
    sqrt_x = ExpWGate(half_turns=0.5, axis_half_turns=0.0)
    cz = Exp11Gate()
    circuit = circuits.Circuit()
    for _ in range(11):
        circuit.append(
            [sqrt_x(qubit) for qubit in qubits if np.random.random() < 0.5])
        circuit.append([cz(qubits[i], qubits[i + 1]) for i in range(9)])
    for i in range(10):
        circuit.append(
            XmonMeasurementGate(key='meas')(qubits[i]))
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


def test_run_no_results():
    simulator = xmon_simulator.Simulator()
    result = simulator.run(basic_circuit())
    assert len(result.measurements) == 0


def test_run():
    np.random.seed(0)
    circuit = basic_circuit()
    circuit.append(
        [XmonMeasurementGate(key='a')(Q1),
         XmonMeasurementGate(key='b')(Q2),])

    simulator = xmon_simulator.Simulator()
    result = simulator.run(circuit)
    assert result.measurements == {'a': [False], 'b': [False]}


def test_run_state():
    simulator = xmon_simulator.Simulator()
    result = simulator.run(basic_circuit(), qubits=[Q1, Q2])
    np.testing.assert_almost_equal(result.state(),
                                   np.array([-0.5j, 0.5, -0.5, 0.5j]))


def test_run_state_different_order_of_qubits():
    simulator = xmon_simulator.Simulator()
    result = simulator.run(basic_circuit(), qubits=[Q2, Q1])
    np.testing.assert_almost_equal(result.state(),
                                   np.array([-0.5j, -0.5, 0.5, 0.5j]))


def test_run_sharded():
    circuit = large_circuit()

    simulator = xmon_simulator.Simulator()
    result = simulator.run(circuit)
    assert result.measurements == {
        'meas': [False, False, False, True, False, False, True, False, False,
                 True]}


def test_run_no_sharding():
    circuit = large_circuit()

    simulator = xmon_simulator.Simulator()
    result = simulator.run(circuit, xmon_simulator.Options(num_shards=1))
    assert result.measurements == {
        'meas': [False, False, False, True, False, False, True, False, False,
                 True]}


def test_run_no_sharing_few_qubits():
    np.random.seed(0)
    circuit = basic_circuit()
    circuit.append(
        [XmonMeasurementGate(key='a')(Q1),
         XmonMeasurementGate(key='b')(Q2),])

    simulator = xmon_simulator.Simulator()
    options = xmon_simulator.Options(min_qubits_before_shard=0)
    result = simulator.run(circuit, options=options)
    assert result.measurements == {'a': [False], 'b': [False]}


def test_run_set_state_computational_basis():
    simulator = xmon_simulator.Simulator()
    result = simulator.run(basic_circuit(), qubits=[Q1, Q2])
    result.set_state(0)
    np.testing.assert_almost_equal(result.state(), np.array([1, 0, 0, 0]))


def test_run_set_state_nd_array_fail():
    simulator = xmon_simulator.Simulator()
    result = simulator.run(basic_circuit(), qubits=[Q1, Q2])
    with pytest.raises(ValueError):
        result.set_state(np.array([0.5, 0.5, 0.5, -0.5], dtype=np.float32))


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
    expected = [{}, {}, {}, {'b': [False]}, {'a': [False]}]
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
    result = []
    for initial_state in range(1 << num_qubits):
        state = simulator.run(circuit, initial_state=initial_state,
                              param_resolver=resolver).state()
        result.append(state)
    return np.array(result).transpose()


@pytest.mark.parametrize('offset', (0.0, 0.2))
def test_param_resolver_exp_w_half_turns(offset):
    exp_w = ExpWGate(
        half_turns=ParameterizedValue('a', offset),
        axis_half_turns=0.0)
    circuit = circuits.Circuit()
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
    circuit = circuits.Circuit()
    circuit.append(exp_w(Q1))
    resolver = ParamResolver({'a': 0.5 - offset})
    result = compute_gate(circuit, resolver)
    amp = 1.0 / math.sqrt(2)
    np.testing.assert_almost_equal(result,
                                   np.array([[0, 1],
                                             [-1, 0]]))


@pytest.mark.parametrize('offset', (0.0, 0.2))
def test_param_resolver_exp_w_multiple_params(offset):
    exp_w = ExpWGate(
        half_turns=ParameterizedValue('a', offset),
        axis_half_turns=ParameterizedValue('b', offset))
    circuit = circuits.Circuit()
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
    circuit = circuits.Circuit()
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
    circuit = circuits.Circuit()
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
    circuit = circuits.Circuit()
    circuit.append(exp_w(Q1))
    resolver = ParamResolver({'a': 0.5})

    simulator = xmon_simulator.Simulator()
    result = simulator.run(circuit, param_resolver=resolver)
    assert result.param_dict == {'a': 0.5}
