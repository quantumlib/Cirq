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

"""Tests for xmon_simulator."""

import cmath
import itertools
import math
import time
from typing import Optional, Callable

import numpy as np
import pytest

import cirq
import cirq.google as cg

Q1 = cirq.GridQubit(0, 0)
Q2 = cirq.GridQubit(1, 0)
Q3 = cirq.GridQubit(2, 0)


def basic_circuit():
    sqrt_x = cirq.PhasedXPowGate(exponent=-0.5, phase_exponent=0.0)
    circuit = cirq.Circuit()
    circuit.append(
        [sqrt_x(Q1), sqrt_x(Q2),
         cirq.CZ(Q1, Q2),
         sqrt_x(Q1), sqrt_x(Q2),
         cirq.Z(Q1)])
    return circuit


def large_circuit():
    np.random.seed(0)
    qubits = [cirq.GridQubit(i, 0) for i in range(10)]
    sqrt_x = cirq.PhasedXPowGate(exponent=0.5, phase_exponent=0.0)
    circuit = cirq.Circuit()
    for _ in range(11):
        circuit.append(
            [sqrt_x(qubit) for qubit in qubits if np.random.random() < 0.5])
        circuit.append([cirq.CZ(qubits[i], qubits[i + 1]) for i in range(9)])
    circuit.append([cirq.MeasurementGate(key='meas')(*qubits)])
    return circuit


def test_xmon_options_negative_num_shards():
    with pytest.raises(AssertionError):
        cg.XmonOptions(num_shards=-1)


def test_xmon_options_negative_min_qubits_before_shard():
    with pytest.raises(AssertionError):
        cg.XmonOptions(min_qubits_before_shard=-1)


def test_xmon_options():
    options = cg.XmonOptions(num_shards=3,
                             min_qubits_before_shard=0)
    assert options.num_prefix_qubits == 1
    assert options.min_qubits_before_shard == 0


def run(simulator: cg.XmonSimulator,
        circuit: cirq.Circuit,
        scheduler: Optional[Callable],
        **kw):
    if scheduler is None:
        program = circuit
    else:
        program = scheduler(cirq.UnconstrainedDevice, circuit)
    return simulator.run(program, **kw)


def simulate(simulator, circuit, scheduler, **kw):
    if scheduler is None:
        program = circuit
    else:
        program = scheduler(cirq.UnconstrainedDevice, circuit)
    return simulator.simulate(program, **kw)


SCHEDULERS = [None, cirq.moment_by_moment_schedule]


@pytest.mark.parametrize(('scheduler', 'use_processes'),
                         zip(SCHEDULERS, (True, False)))
def test_run_no_results(scheduler, use_processes):
    options = cg.XmonOptions(use_processes=use_processes)
    simulator = cg.XmonSimulator(options)
    result = run(simulator, basic_circuit(), scheduler)
    assert len(result.measurements) == 0


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_run(scheduler):
    np.random.seed(0)
    circuit = basic_circuit()
    circuit.append(
        [cirq.MeasurementGate(key='a')(Q1),
         cirq.MeasurementGate(key='b')(Q2)])

    simulator = cg.XmonSimulator()
    result = run(simulator, circuit, scheduler)
    assert result.measurements['a'].dtype == bool
    assert result.measurements['b'].dtype == bool
    np.testing.assert_equal(result.measurements,
                            {'a': [[False]], 'b': [[True]]})


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_run_empty_circuit(scheduler):
    simulator = cg.XmonSimulator()
    result = run(simulator, cirq.Circuit(), scheduler)
    assert len(result.measurements) == 0


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_initial_state_empty_circuit_qubits_specified(scheduler):
    simulator = cg.XmonSimulator()

    result = simulate(simulator,
                      cirq.Circuit(),
                      scheduler,
                      qubit_order=[Q1, Q2])
    np.testing.assert_almost_equal(result.final_state, np.array([1, 0, 0, 0]))

    result = simulate(simulator,
                      cirq.Circuit(),
                      scheduler,
                      qubit_order=[Q1, Q2],
                      initial_state=1)
    np.testing.assert_almost_equal(result.final_state, np.array([0, 1, 0, 0]))

    result = simulate(simulator,
                      cirq.Circuit(),
                      scheduler,
                      qubit_order=[Q1, Q2],
                      initial_state=np.array([0, 1, 0, 0],
                                             dtype=np.complex64))
    np.testing.assert_almost_equal(result.final_state, np.array([0, 1, 0, 0]))


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_qubit_order_to_wavefunction_order_matches_np_kron(scheduler):
    simulator = cg.XmonSimulator()
    zero = [1, 0]
    one = [0, 1]

    result = simulate(simulator,
                      cirq.Circuit.from_ops(cirq.X(Q1)),
                      scheduler,
                      qubit_order=[Q1, Q2])
    assert cirq.allclose_up_to_global_phase(
        result.final_state, np.kron(one, zero))

    result = simulate(simulator,
                      cirq.Circuit.from_ops(cirq.X(Q1)),
                      scheduler,
                      qubit_order=[Q2, Q1])
    assert cirq.allclose_up_to_global_phase(
        result.final_state, np.kron(zero, one))

    result = simulate(simulator,
                      cirq.Circuit.from_ops(cirq.X(Q1)),
                      scheduler,
                      qubit_order=cirq.QubitOrder.sorted_by(repr))
    assert cirq.allclose_up_to_global_phase(
        result.final_state, np.array(one))

    result = simulate(simulator,
                      cirq.Circuit.from_ops(cirq.X(Q1), cirq.Z(Q2)),
                      scheduler,
                      qubit_order=cirq.QubitOrder.sorted_by(repr))
    assert cirq.allclose_up_to_global_phase(
        result.final_state, np.kron(one, zero))


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_bit_flip_order_to_wavefunction_order_matches_np_kron(scheduler):
    simulator = cg.XmonSimulator()

    result = simulate(simulator,
                      cirq.Circuit.from_ops(cirq.X(Q1)),
                      scheduler,
                      qubit_order=[Q1, Q2, Q3])
    assert cirq.allclose_up_to_global_phase(
        result.final_state, np.array([0, 0, 0, 0, 1, 0, 0, 0]))

    result = simulate(simulator,
                      cirq.Circuit.from_ops(cirq.X(Q3)),
                      scheduler,
                      qubit_order=[Q1, Q2, Q3])
    assert cirq.allclose_up_to_global_phase(
        result.final_state, np.array([0, 1, 0, 0, 0, 0, 0, 0]))

    result = simulate(simulator,
                      cirq.Circuit.from_ops(cirq.X(Q3)),
                      scheduler,
                      qubit_order=[Q3, Q2, Q1])
    assert cirq.allclose_up_to_global_phase(
        result.final_state, np.array([0, 0, 0, 0, 1, 0, 0, 0]))

    result = simulate(simulator,
                      cirq.Circuit.from_ops(cirq.X(Q3)),
                      scheduler,
                      qubit_order=[Q2, Q3, Q1])
    assert cirq.allclose_up_to_global_phase(
        result.final_state, np.array([0, 0, 1, 0, 0, 0, 0, 0]))


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_invalid_initial_state_empty_circuit_qubits_specified(scheduler):
    simulator = cg.XmonSimulator()

    with pytest.raises(ValueError):
        _ = simulate(simulator,
                     cirq.Circuit(),
                     scheduler,
                     qubit_order=[Q1, Q2],
                     initial_state=-1)

    with pytest.raises(ValueError):
        _ = simulate(simulator,
                     cirq.Circuit(),
                     scheduler,
                     qubit_order=[Q1, Q2],
                     initial_state=100)

    with pytest.raises(ValueError):
        _ = simulate(simulator,
                     cirq.Circuit(),
                     scheduler,
                     qubit_order=[Q1, Q2],
                     initial_state=np.array([0.0, 1.0], dtype=np.complex64))


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_initial_state_empty_circuit_qubits_not_specified(scheduler):
    simulator = cg.XmonSimulator()

    result = simulate(simulator, cirq.Circuit(), scheduler)
    np.testing.assert_almost_equal(result.final_state, np.array([1.0]))

    result = simulate(simulator, cirq.Circuit(), scheduler, initial_state=0)
    np.testing.assert_almost_equal(result.final_state, np.array([1.0]))

    result = simulate(simulator, cirq.Circuit(), scheduler,
                      initial_state=np.array([1], dtype=np.complex64))
    np.testing.assert_almost_equal(result.final_state, np.array([1.0]))


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_invalid_initial_state_empty_circuit_qubits_not_specified(scheduler):
    simulator = cg.XmonSimulator()

    with pytest.raises(ValueError):
        _ = simulate(simulator, cirq.Circuit(), scheduler, initial_state=2)

    with pytest.raises(ValueError):
        _ = simulate(simulator, cirq.Circuit(), scheduler,
                     initial_state=np.array([2], dtype=np.complex64))

    with pytest.raises(ValueError):
        _ = simulate(simulator, cirq.Circuit(), scheduler,
                     initial_state=np.array([1, 0], dtype=np.complex64))


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_simulate_state(scheduler):
    simulator = cg.XmonSimulator()
    result = simulate(simulator, basic_circuit(), scheduler)
    np.testing.assert_almost_equal(result.final_state,
                                   np.array([0.5j, 0.5, -0.5, -0.5j]))


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_simulate_initial_state_int(scheduler):
    simulator = cg.XmonSimulator()
    result = simulate(simulator, basic_circuit(), scheduler,
                      initial_state=2)
    np.testing.assert_almost_equal(result.final_state,
                                   np.array([0.5, 0.5j, 0.5j, 0.5]))


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_initial_state_identity(scheduler):
    simulator = cg.XmonSimulator()
    result0 = simulate(simulator, cirq.Circuit(), scheduler,
                       initial_state=0, qubit_order=[Q1, Q2])
    result1 = simulate(simulator, cirq.Circuit(), scheduler,
                       initial_state=1, qubit_order=[Q1, Q2])
    result2 = simulate(simulator, cirq.Circuit(), scheduler,
                       initial_state=2, qubit_order=[Q1, Q2])
    result3 = simulate(simulator, cirq.Circuit(), scheduler,
                       initial_state=3, qubit_order=[Q1, Q2])
    np.testing.assert_almost_equal(result0.final_state,
                                   np.array([1, 0, 0, 0]))
    np.testing.assert_almost_equal(result1.final_state,
                                   np.array([0, 1, 0, 0]))
    np.testing.assert_almost_equal(result2.final_state,
                                   np.array([0, 0, 1, 0]))
    np.testing.assert_almost_equal(result3.final_state,
                                   np.array([0, 0, 0, 1]))


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_initial_state_consistency(scheduler):
    def blip(k, n):
        buf = np.zeros(n, dtype=np.complex64)
        buf[k] = 1
        return buf

    simulator = cg.XmonSimulator()
    for i in range(8):
        int_result = simulate(simulator,
                              cirq.Circuit(),
                              scheduler,
                              initial_state=i,
                              qubit_order=[Q1, Q2, Q3]).final_state

        array_result = simulate(simulator,
                                cirq.Circuit(),
                                scheduler,
                                initial_state=blip(i, 8),
                                qubit_order=[Q1, Q2, Q3]).final_state

        np.testing.assert_allclose(int_result, blip(i, 8), atol=1e-8)
        np.testing.assert_allclose(int_result, array_result, atol=1e-8)


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_simulate_initial_state_ndarray(scheduler):
    simulator = cg.XmonSimulator()
    result = simulate(simulator, basic_circuit(), scheduler,
                      initial_state=np.array([0, 0, 1, 0], dtype=np.complex64))
    np.testing.assert_almost_equal(result.final_state,
                                   np.array([0.5, 0.5j, 0.5j, 0.5]))


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_simulate_initial_state_ndarray_upconvert(scheduler):
    simulator = cg.XmonSimulator()
    result = simulate(simulator, basic_circuit(), scheduler,
                      initial_state=np.array([0, 0, 1, 0], dtype=np.float32))
    np.testing.assert_almost_equal(result.final_state,
                                   np.array([0.5, 0.5j, 0.5j, 0.5]))


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_run_initial_state_ndarray_not_upconvertible(scheduler):
    simulator = cg.XmonSimulator()

    with pytest.raises(TypeError):
        _ = run(simulator, basic_circuit(), scheduler,
                initial_state=np.array([0, 0, 1, 0],
                                       dtype=np.float128))


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_simulate_state_different_order_of_qubits(scheduler):
    simulator = cg.XmonSimulator()
    result = simulate(simulator,
                      basic_circuit(),
                      scheduler,
                      qubit_order=[Q2, Q1])
    np.testing.assert_almost_equal(result.final_state,
                                   np.array([0.5j, -0.5, 0.5, -0.5j]))


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_consistent_seeded_run_sharded(scheduler):
    circuit = large_circuit()

    simulator = cg.XmonSimulator()
    result = run(simulator, circuit, scheduler)
    np.testing.assert_equal(
            result.measurements['meas'],
            [[False, False, True, True, True, True, False, True, False, False]])


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_consistent_seeded_run_no_sharding(scheduler):
    circuit = large_circuit()

    simulator = cg.XmonSimulator(
        cg.XmonOptions(num_shards=1))
    result = run(simulator, circuit, scheduler, )
    np.testing.assert_equal(
        result.measurements['meas'],
        [[False, False, True, True, True, True, False, True, False, False]])


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_run_no_sharing_few_qubits(scheduler):
    np.random.seed(0)
    circuit = basic_circuit()
    circuit.append(
        [cirq.MeasurementGate(key='a')(Q1),
         cirq.MeasurementGate(key='b')(Q2)])

    simulator = cg.XmonSimulator(
        cg.XmonOptions(min_qubits_before_shard=0))
    result = run(simulator, circuit, scheduler)
    np.testing.assert_equal(result.measurements['a'], [[False]])
    np.testing.assert_equal(result.measurements['b'], [[True]])


def test_simulate_moment_steps_no_results():
    simulator = cg.XmonSimulator()
    for step in simulator.simulate_moment_steps(basic_circuit()):
        assert len(step.measurements) == 0


def test_simulate_moment_steps():
    np.random.seed(0)
    circuit = basic_circuit()
    circuit.append(
        [cirq.MeasurementGate(key='a')(Q1),
         cirq.MeasurementGate(key='b')(Q2)])

    simulator = cg.XmonSimulator()
    results = []
    for step in simulator.simulate_moment_steps(circuit):
        results.append(step)
    expected = [{}, {}, {}, {}, {'a': [False], 'b': [False]}]
    assert len(results) == len(expected)
    assert all(a.measurements == b for a, b in zip(results, expected))


def test_simulate_moment_steps_state():
    np.random.seed(0)
    circuit = basic_circuit()

    simulator = cg.XmonSimulator()
    results = []
    for step in simulator.simulate_moment_steps(circuit):
        results.append(step.state())
    np.testing.assert_almost_equal(results,
                                   np.array([[0.5, 0.5j, 0.5j, -0.5],
                                             [0.5, 0.5j, 0.5j, 0.5],
                                             [-0.5, 0.5j, 0.5j, -0.5],
                                             [0.5j, 0.5, -0.5, -0.5j]]))


def test_simulate_moment_steps_set_state():
    np.random.seed(0)
    circuit = basic_circuit()

    simulator = cg.XmonSimulator()
    step = simulator.simulate_moment_steps(circuit)

    result = next(step)
    result.set_state(0)
    np.testing.assert_almost_equal(result.state(), np.array([1, 0, 0, 0]))


def test_simulate_moment_steps_set_state_2():
    np.random.seed(0)
    circuit = basic_circuit()

    simulator = cg.XmonSimulator()
    step = simulator.simulate_moment_steps(circuit)

    result = next(step)
    result.set_state(np.array([1j, 0, 0, 0], dtype=np.complex64))
    np.testing.assert_almost_equal(result.state(),
                                   np.array([1j, 0, 0, 0], dtype=np.complex64))


def test_simulate_moment_steps_sample():
    np.random.seed(0)
    circuit = cirq.Circuit.from_ops(cirq.X(Q1),
                                    cirq.MeasurementGate(key='a')(Q1),
                                    cirq.MeasurementGate(key='b')(Q2))
    simulator = cg.XmonSimulator()
    for step in simulator.simulate_moment_steps(circuit, qubit_order=[Q1, Q2]):
        pass
    np.testing.assert_equal([[True]], step.sample([Q1]))
    np.testing.assert_equal([[True, False]], step.sample([Q1, Q2]))
    np.testing.assert_equal([[False]], step.sample([Q2]))

    np.testing.assert_equal([[True]] * 3, step.sample([Q1], 3))
    np.testing.assert_equal([[True, False]] * 3, step.sample([Q1, Q2], 3))
    np.testing.assert_equal([[False]] * 3, step.sample([Q2], 3))


def compute_gate(circuit, resolver, num_qubits=1):
    simulator = cg.XmonSimulator()
    gate = []
    for initial_state in range(1 << num_qubits):
        result = simulator.simulate(circuit,
                                    initial_state=initial_state,
                                    param_resolver=resolver)
        gate.append(result.final_state)
    return np.array(gate).transpose()


def test_param_resolver_exp_w_half_turns():
    exp_w = cirq.PhasedXPowGate(
        exponent=cirq.Symbol('a'),
        phase_exponent=0.0)
    circuit = cirq.Circuit()
    circuit.append(exp_w(Q1))
    resolver = cirq.ParamResolver({'a': -0.5})
    result = compute_gate(circuit, resolver)
    amp = 1.0 / math.sqrt(2)
    np.testing.assert_almost_equal(result,
                                   np.array([[amp, amp * 1j],
                                             [amp * 1j, amp]]))


def test_param_resolver_exp_w_axis_half_turns():
    exp_w = cirq.PhasedXPowGate(
        exponent=1.0, phase_exponent=cirq.Symbol('a'))
    circuit = cirq.Circuit()
    circuit.append(exp_w(Q1))
    resolver = cirq.ParamResolver({'a': 0.5})
    result = compute_gate(circuit, resolver)
    np.testing.assert_almost_equal(result,
                                   np.array([[0, -1],
                                             [1, 0]]))


def test_param_resolver_exp_w_multiple_params():
    exp_w = cirq.PhasedXPowGate(
        exponent=cirq.Symbol('a'),
        phase_exponent=cirq.Symbol('b'))
    circuit = cirq.Circuit()
    circuit.append(exp_w(Q1))
    resolver = cirq.ParamResolver({'a': -0.5, 'b': 0.5})
    result = compute_gate(circuit, resolver)
    amp = 1.0 / math.sqrt(2)
    np.testing.assert_almost_equal(result,
                                   np.array([[amp, amp],
                                             [-amp, amp]]))


def test_param_resolver_exp_z_half_turns():
    exp_z = cirq.Z**cirq.Symbol('a')
    circuit = cirq.Circuit()
    circuit.append(exp_z(Q1))
    resolver = cirq.ParamResolver({'a': -0.5})
    result = compute_gate(circuit, resolver)
    np.testing.assert_almost_equal(
        result,
        np.array([[cmath.exp(1j * math.pi * 0.25), 0],
                  [0, cmath.exp(-1j * math.pi * 0.25)]]))


def test_param_resolver_exp_11_half_turns():
    circuit = cirq.Circuit()
    circuit.append(cirq.CZ(Q1, Q2)**cirq.Symbol('a'))
    resolver = cirq.ParamResolver({'a': 0.5})
    result = compute_gate(circuit, resolver, num_qubits=2)
    # Slight hack: doesn't depend on order of qubits.
    np.testing.assert_almost_equal(
        result,
        np.diag([1, 1, 1, cmath.exp(1j * math.pi * 0.5)]))


def test_param_resolver_param_dict():
    exp_w = cirq.PhasedXPowGate(
        exponent=cirq.Symbol('a'),
        phase_exponent=0.0)
    circuit = cirq.Circuit()
    circuit.append(exp_w(Q1))
    resolver = cirq.ParamResolver({'a': 0.5})

    simulator = cg.XmonSimulator()
    result = simulator.run(circuit, resolver)
    assert result.params.param_dict == {'a': 0.5}


def test_run_circuit_sweep():
    circuit = cirq.Circuit.from_ops(
        cirq.X(Q1)**cirq.Symbol('a'),
        cirq.MeasurementGate('m').on(Q1),
    )

    sweep = cirq.Linspace('a', 0, 10, 11)
    simulator = cg.XmonSimulator()

    for i, result in enumerate(
            simulator.run_sweep(circuit, sweep, repetitions=1)):
        assert result.params['a'] == i
        np.testing.assert_equal(result.measurements['m'], [[i % 2 != 0]])


def test_run_circuit_sweeps():
    circuit = cirq.Circuit.from_ops(
        cirq.X(Q1)**cirq.Symbol('a'),
        cirq.MeasurementGate('m').on(Q1),
    )

    sweep = cirq.Linspace('a', 0, 5, 6)
    sweep2 = cirq.Linspace('a', 6, 10, 5)
    simulator = cg.XmonSimulator()

    for i, result in enumerate(
            simulator.run_sweep(circuit, [sweep, sweep2],
                                repetitions=1)):
        assert result.params['a'] == i
        np.testing.assert_equal(result.measurements['m'], [[i % 2 != 0]])


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_composite_gates(scheduler):
    circuit = cirq.Circuit()
    circuit.append([cirq.X(Q1), cirq.CNOT(Q1, Q2)])
    m = cirq.MeasurementGate('a')
    circuit.append([m(Q1, Q2)])

    simulator = cg.XmonSimulator()
    result = run(simulator, circuit, scheduler)
    np.testing.assert_equal(result.measurements['a'], [[True, True]])


class UnsupportedGate(cirq.SingleQubitGate):

    def __repr__(self):
        return "UnsupportedGate"


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_unsupported_gate(scheduler):
    circuit = cirq.Circuit()
    gate = UnsupportedGate()
    circuit.append([cirq.H(Q1), gate(Q2)])

    simulator = cg.XmonSimulator()
    with pytest.raises(TypeError, message="UnsupportedGate"):
        _ = run(simulator, circuit, scheduler)


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_unsupported_gate_composite(scheduler):
    circuit = cirq.Circuit()
    gate = UnsupportedGate()
    circuit.append([cirq.H(Q1), gate(Q2)])

    simulator = cg.XmonSimulator()
    with pytest.raises(TypeError, message="UnsupportedGate"):
        _ = run(simulator, circuit, scheduler)


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_measurement_qubit_order(scheduler):
    circuit = cirq.Circuit()
    meas = cirq.MeasurementGate(key='')
    circuit.append(cirq.X(Q2))
    circuit.append(cirq.X(Q1))
    circuit.append([meas.on(Q1, Q3, Q2)])
    simulator = cg.XmonSimulator()
    result = run(simulator, circuit, scheduler)
    np.testing.assert_equal(result.measurements[''], [[True, False, True]])


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_inverted_measurement(scheduler):
    circuit = cirq.Circuit.from_ops(
        cirq.MeasurementGate('a', invert_mask=(False,))(Q1),
        cirq.X(Q1),
        cirq.MeasurementGate('b', invert_mask=(False,))(Q1),
        cirq.MeasurementGate('c', invert_mask=(True,))(Q1),
        cirq.X(Q1),
        cirq.MeasurementGate('d', invert_mask=(True,))(Q1))
    simulator = cg.XmonSimulator()
    result = run(simulator, circuit, scheduler)
    np.testing.assert_equal(result.measurements,
                            {'a': [[False]], 'b': [[True]], 'c': [[False]],
                             'd': [[True]]})


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_inverted_measurement_multiple_qubits(scheduler):
    circuit = cirq.Circuit.from_ops(
        cirq.MeasurementGate('a', invert_mask=(False, True))(Q1, Q2),
        cirq.MeasurementGate('b', invert_mask=(True, False))(Q1, Q2),
        cirq.MeasurementGate('c', invert_mask=(True, False))(Q2, Q1))
    simulator = cg.XmonSimulator()
    result = run(simulator, circuit, scheduler)
    np.testing.assert_equal(result.measurements['a'], [[False, True]])
    np.testing.assert_equal(result.measurements['b'], [[True, False]])
    np.testing.assert_equal(result.measurements['c'], [[True, False]])


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_measurement_multiple_measurements(scheduler):
    circuit = cirq.Circuit()
    measure_a = cirq.MeasurementGate('a')
    measure_b = cirq.MeasurementGate('b')
    circuit.append(cirq.X(Q1))
    circuit.append([measure_a.on(Q1, Q2)])
    circuit.append(cirq.X(Q1))
    circuit.append([measure_b.on(Q1, Q2)])
    simulator = cg.XmonSimulator()
    result = run(simulator, circuit, scheduler)
    np.testing.assert_equal(result.measurements['a'], [[True, False]])
    np.testing.assert_equal(result.measurements['b'], [[False, False]])


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_measurement_multiple_measurements_qubit_order(scheduler):
    circuit = cirq.Circuit()
    measure_a = cirq.MeasurementGate('a')
    measure_b = cirq.MeasurementGate('b')
    circuit.append(cirq.X(Q1))
    circuit.append([measure_a.on(Q1, Q2)])
    circuit.append([measure_b.on(Q2, Q1)])
    simulator = cg.XmonSimulator()
    result = run(simulator, circuit, scheduler)
    np.testing.assert_equal(result.measurements['a'], [[True, False]])
    np.testing.assert_equal(result.measurements['b'], [[False, True]])


@pytest.mark.parametrize('scheduler', SCHEDULERS)
def test_measurement_keys_repeat(scheduler):
    circuit = cirq.Circuit()
    meas = cirq.MeasurementGate('a')
    circuit.append([meas.on(Q1), cirq.X.on(Q1), cirq.X.on(Q2), meas.on(Q2)])
    simulator = cg.XmonSimulator()
    with pytest.raises(ValueError, message='Repeated Measurement key a'):
        run(simulator, circuit, scheduler)


def test_handedness_of_xmon_exp_x_gate():
    circuit = cirq.Circuit.from_ops(cirq.X(Q1)**0.5)
    simulator = cg.XmonSimulator()
    result = list(simulator.simulate_moment_steps(circuit))[-1]
    cirq.testing.assert_allclose_up_to_global_phase(
        result.state(),
        np.array([1, -1j]) * np.sqrt(0.5),
        atol=1e-7)


def test_handedness_of_xmon_exp_y_gate():
    circuit = cirq.Circuit.from_ops(cirq.Y(Q1)**0.5)
    simulator = cg.XmonSimulator()
    result = list(simulator.simulate_moment_steps(circuit))[-1]
    cirq.testing.assert_allclose_up_to_global_phase(
        result.state(),
        np.array([1, 1]) * np.sqrt(0.5),
        atol=1e-7)


def test_handedness_of_xmon_exp_z_gate():
    circuit = cirq.Circuit.from_ops(cirq.H(Q1),
                                    cirq.Z(Q1)**0.5)
    simulator = cg.XmonSimulator()
    result = list(simulator.simulate_moment_steps(circuit))[-1]
    cirq.testing.assert_allclose_up_to_global_phase(
        result.state(),
        np.array([1, 1j]) * np.sqrt(0.5),
        atol=1e-7)


def test_handedness_of_xmon_exp_11_gate():
    circuit = cirq.Circuit.from_ops(cirq.H(Q1),
                                    cirq.H(Q2),
                                    cirq.CZ(Q1, Q2)**0.5)
    simulator = cg.XmonSimulator()
    result = list(simulator.simulate_moment_steps(circuit))[-1]
    cirq.testing.assert_allclose_up_to_global_phase(
        result.state(),
        np.array([1, 1, 1, 1j]) / 2,
        atol=1e-7)


def test_handedness_of_x_gate():
    circuit = cirq.Circuit.from_ops(cirq.X(Q1)**0.5)
    simulator = cg.XmonSimulator()
    result = list(simulator.simulate_moment_steps(circuit))[-1]
    cirq.testing.assert_allclose_up_to_global_phase(
        result.state(),
        np.array([1, -1j]) * np.sqrt(0.5),
        atol=1e-7)


def test_handedness_of_y_gate():
    circuit = cirq.Circuit.from_ops(cirq.Y(Q1)**0.5)
    simulator = cg.XmonSimulator()
    result = list(simulator.simulate_moment_steps(circuit))[-1]
    cirq.testing.assert_allclose_up_to_global_phase(
        result.state(),
        np.array([1, 1]) * np.sqrt(0.5),
        atol=1e-7)


def test_handedness_of_z_gate():
    circuit = cirq.Circuit.from_ops(cirq.H(Q1), cirq.Z(Q1)**0.5)
    simulator = cg.XmonSimulator()
    result = list(simulator.simulate_moment_steps(circuit))[-1]
    cirq.testing.assert_allclose_up_to_global_phase(
        result.state(),
        np.array([1, 1j]) * np.sqrt(0.5),
        atol=1e-7)


def test_handedness_of_cz_gate():
    circuit = cirq.Circuit.from_ops(cirq.H(Q1),
                                    cirq.H(Q2),
                                    cirq.CZ(Q1, Q2)**0.5)
    simulator = cg.XmonSimulator()
    result = list(simulator.simulate_moment_steps(circuit))[-1]
    cirq.testing.assert_allclose_up_to_global_phase(
        result.state(),
        np.array([1, 1, 1, 1j]) / 2,
        atol=1e-7)


def test_handedness_of_basic_gates():
    circuit = cirq.Circuit.from_ops(
        cirq.X(Q1)**-0.5,
        cirq.Z(Q1)**-0.5,
        cirq.Y(Q1)**0.5,
        cirq.MeasurementGate(key='').on(Q1),
    )
    result = cg.XmonSimulator().run(circuit)
    np.testing.assert_equal(result.measurements[''], [[True]])


def test_handedness_of_xmon_gates():
    circuit = cirq.Circuit.from_ops(
        cirq.X(Q1)**-0.5,
        cirq.Z(Q1)**-0.5,
        cirq.Y(Q1)**0.5,
        cirq.MeasurementGate(key='').on(Q1),
    )
    result = cg.XmonSimulator().run(circuit)
    np.testing.assert_equal(result.measurements[''], [[True]])


def bit_flip_circuit(flip0, flip1):
    q1, q2 = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)
    g1, g2 = cirq.X(q1)**flip0, cirq.X(q2)**flip1
    m1, m2 = cirq.MeasurementGate('q1')(q1), cirq.MeasurementGate('q2')(q2)
    circuit = cirq.Circuit()
    circuit.append([g1, g2, m1, m2])
    return circuit


def test_circuit_repetitions():
    sim = cg.XmonSimulator()
    circuit = bit_flip_circuit(1, 1)

    result = sim.run(circuit, repetitions=10)
    assert result.params.param_dict == {}
    assert result.repetitions == 10
    np.testing.assert_equal(result.measurements['q1'], [[True]] * 10)
    np.testing.assert_equal(result.measurements['q2'], [[True]] * 10)


def test_circuit_repetitions_optimized_regression():
    sim = cg.XmonSimulator()
    circuit = bit_flip_circuit(1, 1)

    # When not optimized this takes around 20 seconds to run, otherwise it
    # runs in less than a second.
    start = time.time()
    result = sim.run(circuit, repetitions=10000)
    assert result.repetitions == 10000
    end = time.time()
    assert end - start < 1.0


def test_circuit_parameters():
    sim = cg.XmonSimulator()
    circuit = bit_flip_circuit(cirq.Symbol('a'), cirq.Symbol('b'))

    resolvers = [cirq.ParamResolver({'a': b1, 'b': b2})
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


def test_circuit_bad_parameters():
    sim = cg.XmonSimulator()
    circuit = bit_flip_circuit(cirq.Symbol('a'), cirq.Symbol('b'))
    with pytest.raises(TypeError):
        sim.run_sweep(circuit, params=3, repetitions=1)


def test_circuit_param_and_reps():
    sim = cg.XmonSimulator()
    circuit = bit_flip_circuit(cirq.Symbol('a'), cirq.Symbol('b'))

    resolvers = [cirq.ParamResolver({'a': b1, 'b': b2})
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
    assert (set(itertools.product([0, 1], [0, 1]))
            == {(r.params['a'], r.params['b']) for r in all_trials})


def assert_simulated_states_match_circuit_matrix_by_basis(circuit):
    basis = [Q1, Q2]
    matrix = circuit.to_unitary_matrix(qubit_order=basis)
    simulator = cg.XmonSimulator()
    for i in range(matrix.shape[0]):
        col = matrix[:, i]
        result = list(simulator.simulate_moment_steps(
            circuit,
            initial_state=i,
            qubit_order=basis))[-1]
        cirq.testing.assert_allclose_up_to_global_phase(
            col,
            result.state(),
            atol=1e-5)


def test_compare_simulator_states_to_gate_matrices():
    assert_simulated_states_match_circuit_matrix_by_basis(
        cirq.Circuit.from_ops(cirq.CNOT(Q1, Q2)))

    assert_simulated_states_match_circuit_matrix_by_basis(
        cirq.Circuit.from_ops(cirq.Z(Q1)**0.5, cirq.Z(Q2)))

    assert_simulated_states_match_circuit_matrix_by_basis(
        cirq.Circuit.from_ops(cirq.X(Q1)**0.5))

    assert_simulated_states_match_circuit_matrix_by_basis(
        cirq.Circuit.from_ops(cirq.Y(Q2)**(1 / 3)))

    assert_simulated_states_match_circuit_matrix_by_basis(
        cirq.Circuit.from_ops(cirq.H(Q2)))

    assert_simulated_states_match_circuit_matrix_by_basis(
        cirq.Circuit.from_ops(cirq.CZ(Q1, Q2)**0.5))


def test_simulator_trial_result():
    a = cirq.GridQubit(0, 0)
    b = cirq.GridQubit(0, 1)
    c = cirq.GridQubit(0, 2)
    circuit = cirq.Circuit.from_ops(
        cirq.X(a),
        cirq.CNOT(a, b),
        cirq.measure(a, key='a'),
        cirq.measure(b, key='b'),
        cirq.measure(c, key='c')
    )
    result = cirq.google.XmonSimulator().run(circuit)
    assert str(result) == 'a=1\nb=1\nc=0'


def test_simulator_trial_repeated_result():
    a = cirq.GridQubit(0, 0)
    b = cirq.GridQubit(0, 1)
    c = cirq.GridQubit(0, 2)
    circuit = cirq.Circuit.from_ops(
        cirq.X(b),
        cirq.measure(a, b, key='ab'),
        cirq.measure(c, key='c')
    )
    result = cirq.google.XmonSimulator().run(circuit, repetitions=5)
    assert str(result) == 'ab=00000, 11111\nc=00000'


def test_simulator_simulate_trial_result_str():
    a = cirq.GridQubit(0, 0)
    b = cirq.GridQubit(0, 1)
    c = cirq.GridQubit(0, 2)
    circuit = cirq.Circuit.from_ops(
        cirq.X(a),
        cirq.CNOT(a, b),
        cirq.measure(a, key='a'),
        cirq.measure(b, key='b'),
        cirq.measure(c, key='c')
    )
    result = cirq.google.XmonSimulator().simulate(circuit)
    assert str(result) == "a=1 b=1 c=0"


def test_simulator_implied_measurement_key():
    q = cirq.GridQubit(0, 0)
    circuit = cirq.Circuit.from_ops(
        cirq.X(q),
        cirq.measure(q),
        cirq.measure(q, key='other'),
    )
    result = cirq.google.XmonSimulator().run(circuit, repetitions=5)
    assert str(result) == "(0, 0)=11111\nother=11111"


