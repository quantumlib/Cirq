# Copyright 2019 The Cirq Developers
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
from unittest import mock
import itertools
import random
import numpy as np
import pytest
import sympy

import cirq


class PlusGate(cirq.Gate):
    """A qudit gate that increments a qudit state mod its dimension."""

    def __init__(self, dimension, increment=1):
        self.dimension = dimension
        self.increment = increment % dimension

    def _qid_shape_(self):
        return (self.dimension,)

    def _unitary_(self):
        inc = (self.increment - 1) % self.dimension + 1
        u = np.empty((self.dimension, self.dimension))
        u[inc:] = np.eye(self.dimension)[:-inc]
        u[:inc] = np.eye(self.dimension)[-inc:]
        return u


class _TestMixture(cirq.Gate):

    def __init__(self, gate_options):
        self.gate_options = gate_options

    def _qid_shape_(self):
        return cirq.qid_shape(self.gate_options[0], ())

    def _mixture_(self):
        return [(1 / len(self.gate_options), cirq.unitary(g))
                for g in self.gate_options]


def test_invalid_dtype():
    with pytest.raises(ValueError, match='complex'):
        cirq.DensityMatrixSimulator(dtype=np.int32)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_no_measurements(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)

    circuit = cirq.Circuit(cirq.X(q0), cirq.X(q1))
    with pytest.raises(ValueError, match="no measurements"):
        simulator.run(circuit)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_no_results(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)

    circuit = cirq.Circuit(cirq.X(q0), cirq.X(q1))
    with pytest.raises(ValueError, match="no measurements"):
        simulator.run(circuit)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_empty_circuit(dtype):
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    with pytest.raises(ValueError, match="no measurements"):
        simulator.run(cirq.Circuit())


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_bit_flips(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit((cirq.X**b0)(q0), (cirq.X**b1)(q1),
                                   cirq.measure(q0), cirq.measure(q1))
            result = simulator.run(circuit)
            np.testing.assert_equal(result.measurements,
                                    {'0': [[b0]], '1': [[b1]]})


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_bit_flips_with_dephasing(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype,
                                            ignore_measurement_results=True)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit((cirq.X**b0)(q0), (cirq.X**b1)(q1),
                                   cirq.measure(q0), cirq.measure(q1))
            result = simulator.run(circuit)
            np.testing.assert_equal(result.measurements, {
                '0': [[b0]],
                '1': [[b1]]
            })


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_qudit_increments(dtype):
    q0, q1 = cirq.LineQid.for_qid_shape((3, 4))
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    for b0 in [0, 1, 2]:
        for b1 in [0, 1, 2, 3]:
            circuit = cirq.Circuit(
                [PlusGate(3, 1)(q0)] * b0,
                [PlusGate(4, 1)(q1)] * b1,
                cirq.measure(q0),
                cirq.measure(q1),
            )
            result = simulator.run(circuit)
            np.testing.assert_equal(result.measurements, {
                '0 (d=3)': [[b0]],
                '1 (d=4)': [[b1]]
            })


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_not_channel_op(dtype):
    class BadOp(cirq.Operation):
        def __init__(self, qubits):
            self._qubits = qubits

        @property
        def qubits(self):
            return self._qubits

        def with_qubits(self, *new_qubits):
            # coverage: ignore
            return BadOp(self._qubits)

    q0 = cirq.LineQubit(0)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    circuit = cirq.Circuit([BadOp([q0])])
    with pytest.raises(TypeError):
        simulator.simulate(circuit)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_mixture(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.bit_flip(0.5)(q0), cirq.measure(q0), cirq.measure(q1))
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    result = simulator.run(circuit, repetitions=100)
    np.testing.assert_equal(result.measurements['1'], [[0]] * 100)
    # Test that we get at least one of each result. Probability of this test
    # failing is 2 ** (-99).
    q0_measurements = set(x[0] for x in result.measurements['0'].tolist())
    assert q0_measurements == {0, 1}


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_qudit_mixture(dtype):
    q0, q1 = cirq.LineQid.for_qid_shape((3, 2))
    mixture = _TestMixture([PlusGate(3, 0), PlusGate(3, 1), PlusGate(3, 2)])
    circuit = cirq.Circuit(mixture(q0), cirq.measure(q0), cirq.measure(q1))
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    result = simulator.run(circuit, repetitions=100)
    np.testing.assert_equal(result.measurements['1 (d=2)'], [[0]] * 100)
    # Test that we get at least one of each result. Probability of this test
    # failing is about 3 * (2/3) ** 100.
    q0_measurements = set(x[0] for x in result.measurements['0 (d=3)'].tolist())
    assert q0_measurements == {0, 1, 2}


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_channel(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.X(q0),
                           cirq.amplitude_damp(0.5)(q0), cirq.measure(q0),
                           cirq.measure(q1))

    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    result = simulator.run(circuit, repetitions=100)
    np.testing.assert_equal(result.measurements['1'], [[0]] * 100)
    # Test that we get at least one of each result. Probability of this test
    # failing is 2 ** (-99).
    q0_measurements = set(x[0] for x in result.measurements['0'].tolist())
    assert q0_measurements == {0, 1}


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_qudit_channel(dtype):

    class TestChannel(cirq.Gate):

        def _qid_shape_(self):
            return (3,)

        def _channel_(self):
            return [
                np.array([[1, 0, 0], [0, 0.5**0.5, 0], [0, 0, 0.5**0.5]]),
                np.array([[0, 0.5**0.5, 0], [0, 0, 0], [0, 0, 0]]),
                np.array([[0, 0, 0], [0, 0, 0.5**0.5], [0, 0, 0]]),
            ]

    q0, q1 = cirq.LineQid.for_qid_shape((3, 4))
    circuit = cirq.Circuit(
        PlusGate(3, 2)(q0),
        TestChannel()(q0),
        TestChannel()(q0),
        cirq.measure(q0),
        cirq.measure(q1),
    )

    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    result = simulator.run(circuit, repetitions=100)
    np.testing.assert_equal(result.measurements['1 (d=4)'], [[0]] * 100)
    # Test that we get at least one of each result. Probability of this test
    # failing is about (3/4) ** 100.
    q0_measurements = set(x[0] for x in result.measurements['0 (d=3)'].tolist())
    assert q0_measurements == {0, 1, 2}


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_measure_at_end_no_repetitions(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    with mock.patch.object(simulator,
                           '_base_iterator',
                           wraps=simulator._base_iterator) as mock_sim:
        for b0 in [0, 1]:
            for b1 in [0, 1]:
                circuit = cirq.Circuit((cirq.X**b0)(q0), (cirq.X**b1)(q1),
                                       cirq.measure(q0), cirq.measure(q1))
                result = simulator.run(circuit, repetitions=0)
                np.testing.assert_equal(result.measurements, {
                    '0': np.empty([0, 1]),
                    '1': np.empty([0, 1])
                })
                assert result.repetitions == 0
        assert mock_sim.call_count == 4


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_repetitions_measure_at_end(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    with mock.patch.object(simulator, '_base_iterator',
                           wraps=simulator._base_iterator) as mock_sim:
        for b0 in [0, 1]:
            for b1 in [0, 1]:
                circuit = cirq.Circuit((cirq.X**b0)(q0), (cirq.X**b1)(q1),
                                       cirq.measure(q0), cirq.measure(q1))
                result = simulator.run(circuit, repetitions=3)
                np.testing.assert_equal(result.measurements,
                                        {'0': [[b0]] * 3, '1': [[b1]] * 3})
                assert result.repetitions == 3
        assert mock_sim.call_count == 4


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_qudits_repetitions_measure_at_end(dtype):
    q0, q1 = cirq.LineQid.for_qid_shape((2, 3))
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    with mock.patch.object(simulator,
                           '_base_iterator',
                           wraps=simulator._base_iterator) as mock_sim:
        for b0 in [0, 1]:
            for b1 in [0, 1, 2]:
                circuit = cirq.Circuit((cirq.X**b0)(q0),
                                       PlusGate(3, b1)(q1), cirq.measure(q0),
                                       cirq.measure(q1))
                result = simulator.run(circuit, repetitions=3)
                np.testing.assert_equal(result.measurements, {
                    '0 (d=2)': [[b0]] * 3,
                    '1 (d=3)': [[b1]] * 3
                })
                assert result.repetitions == 3
        assert mock_sim.call_count == 6


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_measurement_not_terminal_no_repetitions(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    with mock.patch.object(simulator,
                           '_base_iterator',
                           wraps=simulator._base_iterator) as mock_sim:
        for b0 in [0, 1]:
            for b1 in [0, 1]:
                circuit = cirq.Circuit((cirq.X**b0)(q0), (cirq.X**b1)(q1),
                                       cirq.measure(q0), cirq.measure(q1),
                                       cirq.H(q0), cirq.H(q1))
                result = simulator.run(circuit, repetitions=0)
                np.testing.assert_equal(result.measurements, {
                    '0': np.empty([0, 1]),
                    '1': np.empty([0, 1])
                })
                assert result.repetitions == 0
        assert mock_sim.call_count == 0


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_repetitions_measurement_not_terminal(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    with mock.patch.object(simulator, '_base_iterator',
                           wraps=simulator._base_iterator) as mock_sim:
        for b0 in [0, 1]:
            for b1 in [0, 1]:
                circuit = cirq.Circuit((cirq.X**b0)(q0), (cirq.X**b1)(q1),
                                       cirq.measure(q0), cirq.measure(q1),
                                       cirq.H(q0), cirq.H(q1))
                result = simulator.run(circuit, repetitions=3)
                np.testing.assert_equal(result.measurements,
                                        {'0': [[b0]] * 3, '1': [[b1]] * 3})
                assert result.repetitions == 3
        assert mock_sim.call_count == 12


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_qudits_repetitions_measurement_not_terminal(dtype):
    q0, q1 = cirq.LineQid.for_qid_shape((2, 3))
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    with mock.patch.object(simulator,
                           '_base_iterator',
                           wraps=simulator._base_iterator) as mock_sim:
        for b0 in [0, 1]:
            for b1 in [0, 1, 2]:
                circuit = cirq.Circuit((cirq.X**b0)(q0),
                                       PlusGate(3, b1)(q1), cirq.measure(q0),
                                       cirq.measure(q1), cirq.H(q0),
                                       PlusGate(3, -b1)(q1))
                result = simulator.run(circuit, repetitions=3)
                np.testing.assert_equal(result.measurements, {
                    '0 (d=2)': [[b0]] * 3,
                    '1 (d=3)': [[b1]] * 3
                })
                assert result.repetitions == 3
        assert mock_sim.call_count == 18


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_param_resolver(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit((cirq.X**sympy.Symbol('b0'))(q0),
                                   (cirq.X**sympy.Symbol('b1'))(q1),
                                   cirq.measure(q0), cirq.measure(q1))
            param_resolver = {'b0': b0, 'b1': b1}
            result = simulator.run(circuit, param_resolver=param_resolver)
            np.testing.assert_equal(result.measurements,
                                    {'0': [[b0]], '1': [[b1]] })
            np.testing.assert_equal(result.params,
                                    cirq.ParamResolver(param_resolver))


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_correlations(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1), cirq.measure(q0, q1))
    for _ in range(10):
        result = simulator.run(circuit)
        bits = result.measurements['0,1'][0]
        assert bits[0] == bits[1]


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_measure_multiple_qubits(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit((cirq.X**b0)(q0), (cirq.X**b1)(q1),
                                   cirq.measure(q0, q1))
            result = simulator.run(circuit, repetitions=3)
            np.testing.assert_equal(result.measurements,
                                    {'0,1': [[b0, b1]] * 3})


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_measure_multiple_qudits(dtype):
    q0, q1 = cirq.LineQid.for_qid_shape((2, 3))
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1, 2]:
            circuit = cirq.Circuit((cirq.X**b0)(q0),
                                   PlusGate(3, b1)(q1), cirq.measure(q0, q1))
            result = simulator.run(circuit, repetitions=3)
            np.testing.assert_equal(result.measurements,
                                    {'0 (d=2),1 (d=3)': [[b0, b1]] * 3})


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_run_sweeps_param_resolvers(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit((cirq.X**sympy.Symbol('b0'))(q0),
                                   (cirq.X**sympy.Symbol('b1'))(q1),
                                   cirq.measure(q0), cirq.measure(q1))
            params = [cirq.ParamResolver({'b0': b0, 'b1': b1}),
                      cirq.ParamResolver({'b0': b1, 'b1': b0})]
            results = simulator.run_sweep(circuit, params=params)

            assert len(results) == 2
            np.testing.assert_equal(results[0].measurements,
                                    {'0': [[b0]], '1': [[b1]] })
            np.testing.assert_equal(results[1].measurements,
                                    {'0': [[b1]], '1': [[b0]] })
            assert results[0].params == params[0]
            assert results[1].params == params[1]


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_no_circuit(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    circuit = cirq.Circuit()
    result = simulator.simulate(circuit, qubit_order=[q0, q1])
    expected = np.zeros((4, 4))
    expected[0, 0] = 1.0
    np.testing.assert_almost_equal(result.final_density_matrix, expected)
    assert len(result.measurements) == 0


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    circuit = cirq.Circuit(cirq.H(q0), cirq.H(q1))
    result = simulator.simulate(circuit, qubit_order=[q0, q1])
    np.testing.assert_almost_equal(result.final_density_matrix,
                                   np.ones((4, 4)) * 0.25)
    assert len(result.measurements) == 0


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_qudits(dtype):
    q0, q1 = cirq.LineQid.for_qid_shape((2, 3))
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    circuit = cirq.Circuit(cirq.H(q0), PlusGate(3, 2)(q1))
    result = simulator.simulate(circuit, qubit_order=[q1, q0])
    expected = np.zeros((6, 6))
    expected[4:, 4:] = np.ones((2, 2)) / 2
    np.testing.assert_almost_equal(result.final_density_matrix, expected)
    assert len(result.measurements) == 0


@pytest.mark.parametrize(
    'dtype,circuit',
    itertools.product([np.complex64, np.complex128], [
        cirq.testing.random_circuit(cirq.LineQubit.range(4), 5, 0.9)
        for _ in range(20)
    ]))
def test_simulate_compare_to_state_vector_simulator(dtype, circuit):
    qubits = cirq.LineQubit.range(4)
    pure_result = (cirq.Simulator(dtype=dtype).simulate(
        circuit, qubit_order=qubits).density_matrix_of())
    mixed_result = (cirq.DensityMatrixSimulator(dtype=dtype).simulate(
        circuit, qubit_order=qubits).final_density_matrix)
    assert mixed_result.shape == (16, 16)
    np.testing.assert_almost_equal(mixed_result, pure_result)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_bit_flips(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit((cirq.X**b0)(q0), (cirq.X**b1)(q1),
                                   cirq.measure(q0), cirq.measure(q1))
            result = simulator.simulate(circuit)
            np.testing.assert_equal(result.measurements, {'0': [b0], '1': [b1]})
            expected_density_matrix = np.zeros(shape=(4, 4))
            expected_density_matrix[b0 * 2 + b1, b0 * 2 + b1] = 1.0
            np.testing.assert_equal(result.final_density_matrix,
                                    expected_density_matrix)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_qudit_increments(dtype):
    q0, q1 = cirq.LineQid.for_qid_shape((2, 3))
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1, 2]:
            circuit = cirq.Circuit((cirq.X**b0)(q0), (PlusGate(3)(q1),) * b1,
                                   cirq.measure(q0), cirq.measure(q1))
            result = simulator.simulate(circuit)
            np.testing.assert_equal(result.measurements, {
                '0 (d=2)': [b0],
                '1 (d=3)': [b1]
            })
            expected_density_matrix = np.zeros(shape=(6, 6))
            expected_density_matrix[b0 * 3 + b1, b0 * 3 + b1] = 1.0
            np.testing.assert_equal(result.final_density_matrix,
                                    expected_density_matrix)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_initial_state(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit((cirq.X**b0)(q0), (cirq.X**b1)(q1))
            result = simulator.simulate(circuit, initial_state=1)
            expected_density_matrix = np.zeros(shape=(4, 4))
            expected_density_matrix[b0 * 2 + 1 - b1, b0 * 2 + 1 - b1] = 1.0
            np.testing.assert_equal(result.final_density_matrix,
                                    expected_density_matrix)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_initial_qudit_state(dtype):
    q0, q1 = cirq.LineQid.for_qid_shape((3, 4))
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    for b0 in [0, 1, 2]:
        for b1 in [0, 1, 2, 3]:
            circuit = cirq.Circuit(
                PlusGate(3, b0)(q0),
                PlusGate(4, b1)(q1),
            )
            result = simulator.simulate(circuit, initial_state=6)
            expected_density_matrix = np.zeros(shape=(12, 12))
            expected_density_matrix[(b0 + 1) % 3 * 4 + (b1 + 2) % 4,
                                    (b0 + 1) % 3 * 4 + (b1 + 2) % 4] = 1.0
            np.testing.assert_equal(result.final_density_matrix,
                                    expected_density_matrix)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_qubit_order(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit((cirq.X**b0)(q0), (cirq.X**b1)(q1))
            result = simulator.simulate(circuit, qubit_order=[q1, q0])
            expected_density_matrix = np.zeros(shape=(4, 4))
            expected_density_matrix[2 * b1 + b0, 2 * b1 + b0] = 1.0
            np.testing.assert_equal(result.final_density_matrix,
                                    expected_density_matrix)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_param_resolver(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit((cirq.X**sympy.Symbol('b0'))(q0),
                                   (cirq.X**sympy.Symbol('b1'))(q1))
            resolver = cirq.ParamResolver({'b0': b0, 'b1': b1})
            result = simulator.simulate(circuit, param_resolver=resolver)
            expected_density_matrix = np.zeros(shape=(4, 4))
            expected_density_matrix[2 * b0 + b1, 2 * b0 + b1] = 1.0
            np.testing.assert_equal(result.final_density_matrix,
                                    expected_density_matrix)
            assert result.params == resolver
            assert len(result.measurements) == 0


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_measure_multiple_qubits(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit((cirq.X**b0)(q0), (cirq.X**b1)(q1),
                                   cirq.measure(q0, q1))
            result = simulator.simulate(circuit)
            np.testing.assert_equal(result.measurements,
                                    {'0,1': [b0, b1]})


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_measure_multiple_qudits(dtype):
    q0, q1 = cirq.LineQid.for_qid_shape((2, 3))
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1, 2]:
            circuit = cirq.Circuit((cirq.X**b0)(q0),
                                   PlusGate(3, b1)(q1), cirq.measure(q0, q1))
            result = simulator.simulate(circuit)
            np.testing.assert_equal(result.measurements,
                                    {'0 (d=2),1 (d=3)': [b0, b1]})


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_sweeps_param_resolver(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit((cirq.X**sympy.Symbol('b0'))(q0),
                                   (cirq.X**sympy.Symbol('b1'))(q1))
            params = [cirq.ParamResolver({'b0': b0, 'b1': b1}),
                      cirq.ParamResolver({'b0': b1, 'b1': b0})]
            results = simulator.simulate_sweep(circuit, params=params)
            expected_density_matrix = np.zeros(shape=(4, 4))
            expected_density_matrix[2 * b0 + b1, 2 * b0 + b1] = 1.0
            np.testing.assert_equal(results[0].final_density_matrix,
                                    expected_density_matrix)

            expected_density_matrix = np.zeros(shape=(4, 4))
            expected_density_matrix[2 * b1 + b0, 2 * b1 + b0] = 1.0
            np.testing.assert_equal(results[1].final_density_matrix,
                                    expected_density_matrix)

            assert results[0].params == params[0]
            assert results[1].params == params[1]


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_moment_steps(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.H(q1), cirq.H(q0), cirq.H(q1))
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    for i, step in enumerate(simulator.simulate_moment_steps(circuit)):
        assert cirq.qid_shape(step) == (2, 2)
        if i == 0:
            np.testing.assert_almost_equal(step.density_matrix(),
                                           np.ones((4, 4)) / 4)
        else:
            np.testing.assert_almost_equal(step.density_matrix(),
                                           np.diag([1, 0, 0, 0]))


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_moment_steps_qudits(dtype):
    q0, q1 = cirq.LineQid.for_qid_shape((2, 3))
    circuit = cirq.Circuit(
        PlusGate(2, 1)(q0),
        PlusGate(3, 1)(q1),
        cirq.reset(q1),
        PlusGate(3, 1)(q1),
    )
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    for i, step in enumerate(simulator.simulate_moment_steps(circuit)):
        assert cirq.qid_shape(step) == (2, 3)
        if i == 0:
            np.testing.assert_almost_equal(step.density_matrix(),
                                           np.diag([0, 0, 0, 0, 1, 0]))
        elif i == 1:
            np.testing.assert_almost_equal(step.density_matrix(),
                                           np.diag([0, 0, 0, 1, 0, 0]))
        else:
            np.testing.assert_almost_equal(step.density_matrix(),
                                           np.diag([0, 0, 0, 0, 1, 0]))


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_moment_steps_empty_circuit(dtype):
    circuit = cirq.Circuit()
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    step = None
    for step in simulator.simulate_moment_steps(circuit):
        pass
    assert step._simulator_state() == cirq.DensityMatrixSimulatorState(
        density_matrix=np.array([[1]]), qubit_map={})


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_moment_steps_set_state(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.H(q1), cirq.H(q0), cirq.H(q1))
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    for i, step in enumerate(simulator.simulate_moment_steps(circuit)):
        np.testing.assert_almost_equal(step.density_matrix(),
                                       np.ones((4, 4)) * 0.25)
        if i == 0:
            zero_zero = np.zeros((4, 4), dtype=dtype)
            zero_zero[0, 0] = 1
            step.set_density_matrix(zero_zero)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_moment_steps_sample(dtype):
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    for i, step in enumerate(simulator.simulate_moment_steps(circuit)):
        if i == 0:
            samples = step.sample([q0, q1], repetitions=10)
            for sample in samples:
                assert (np.array_equal(sample, [True, False])
                        or np.array_equal(sample, [False, False]))
        else:
            samples = step.sample([q0, q1], repetitions=10)
            for sample in samples:
                assert (np.array_equal(sample, [True, True])
                        or np.array_equal(sample, [False, False]))


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_moment_steps_sample_qudits(dtype):

    class TestGate(cirq.Gate):
        """Swaps the 2nd qid |0> and |2> states when the 1st is |1>."""

        def _qid_shape_(self):
            return (2, 3)

        def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs):
            args.available_buffer[..., 1, 0] = args.target_tensor[..., 1, 2]
            args.target_tensor[..., 1, 2] = args.target_tensor[..., 1, 0]
            args.target_tensor[..., 1, 0] = args.available_buffer[..., 1, 0]
            return args.target_tensor

    q0, q1 = cirq.LineQid.for_qid_shape((2, 3))
    circuit = cirq.Circuit(cirq.H(q0), TestGate()(q0, q1))
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    for i, step in enumerate(simulator.simulate_moment_steps(circuit)):
        if i == 0:
            samples = step.sample([q0, q1], repetitions=10)
            for sample in samples:
                assert (np.array_equal(sample, [True, 0]) or
                        np.array_equal(sample, [False, 0]))
        else:
            samples = step.sample([q0, q1], repetitions=10)
            for sample in samples:
                assert (np.array_equal(sample, [True, 2]) or
                        np.array_equal(sample, [False, 0]))


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_simulate_moment_steps_intermediate_measurement(dtype):
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0), cirq.H(q0))
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    for i, step in enumerate(simulator.simulate_moment_steps(circuit)):
        if i == 1:
            result = int(step.measurements['0'][0])
            expected = np.zeros((2, 2))
            expected[result, result] = 1
            np.testing.assert_almost_equal(step.density_matrix(), expected)
        if i == 2:
            expected = np.array([[0.5, 0.5 * (-1) ** result],
                                 [0.5 * (-1) ** result, 0.5]])
            np.testing.assert_almost_equal(step.density_matrix(), expected)


def test_density_matrix_simulator_state_eq():
    q0, q1 = cirq.LineQubit.range(2)
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(
        cirq.DensityMatrixSimulatorState(density_matrix=np.ones((2, 2)) * 0.5,
                                         qubit_map={q0: 0}),
        cirq.DensityMatrixSimulatorState(density_matrix=np.ones((2, 2)) * 0.5,
                                         qubit_map={q0: 0}))
    eq.add_equality_group(
        cirq.DensityMatrixSimulatorState(density_matrix=np.eye(2) * 0.5,
                                         qubit_map={q0: 0}))
    eq.add_equality_group(
        cirq.DensityMatrixSimulatorState(density_matrix=np.eye(2) * 0.5,
                                         qubit_map={q0: 0, q1: 1}))


def test_density_matrix_simulator_state_qid_shape():
    q0, q1 = cirq.LineQubit.range(2)
    assert cirq.qid_shape(
        cirq.DensityMatrixSimulatorState(density_matrix=np.ones((4, 4)) / 4,
                                         qubit_map={
                                             q0: 0,
                                             q1: 1
                                         })) == (2, 2)
    q0, q1 = cirq.LineQid.for_qid_shape((3, 4))
    assert cirq.qid_shape(
        cirq.DensityMatrixSimulatorState(density_matrix=np.ones((12, 12)) / 12,
                                         qubit_map={
                                             q0: 0,
                                             q1: 1
                                         })) == (3, 4)


def test_density_matrix_simulator_state_repr():
    q0 = cirq.LineQubit(0)
    assert (repr(cirq.DensityMatrixSimulatorState(
        density_matrix=np.ones((2, 2)) * 0.5, qubit_map={q0: 0}))
            == "cirq.DensityMatrixSimulatorState(density_matrix="
               "np.array([[0.5, 0.5], [0.5, 0.5]]), "
               "qubit_map={cirq.LineQubit(0): 0})")


def test_density_matrix_trial_result_eq():
    q0 = cirq.LineQubit(0)
    final_simulator_state = cirq.DensityMatrixSimulatorState(
        density_matrix=np.ones((2, 2)) * 0.5,
        qubit_map={q0: 0})
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(
        cirq.DensityMatrixTrialResult(
            params=cirq.ParamResolver({}),
            measurements={},
            final_simulator_state=final_simulator_state),
        cirq.DensityMatrixTrialResult(
            params=cirq.ParamResolver({}),
            measurements={},
            final_simulator_state=final_simulator_state))
    eq.add_equality_group(
        cirq.DensityMatrixTrialResult(
            params=cirq.ParamResolver({'s': 1}),
            measurements={},
            final_simulator_state=final_simulator_state))
    eq.add_equality_group(
        cirq.DensityMatrixTrialResult(
            params=cirq.ParamResolver({'s': 1}),
            measurements={'m': np.array([[1]])},
            final_simulator_state=final_simulator_state))


def test_density_matrix_trial_result_qid_shape():
    q0, q1 = cirq.LineQubit.range(2)
    assert cirq.qid_shape(
        cirq.DensityMatrixTrialResult(
            params=cirq.ParamResolver({}),
            measurements={},
            final_simulator_state=cirq.DensityMatrixSimulatorState(
                density_matrix=np.ones((4, 4)) / 4, qubit_map={
                    q0: 0,
                    q1: 1
                }))) == (2, 2)
    q0, q1 = cirq.LineQid.for_qid_shape((3, 4))
    assert cirq.qid_shape(
        cirq.DensityMatrixTrialResult(
            params=cirq.ParamResolver({}),
            measurements={},
            final_simulator_state=cirq.DensityMatrixSimulatorState(
                density_matrix=np.ones((12, 12)) / 12, qubit_map={
                    q0: 0,
                    q1: 1
                }))) == (3, 4)


def test_density_matrix_trial_result_repr():
    q0 = cirq.LineQubit(0)
    final_simulator_state = cirq.DensityMatrixSimulatorState(
        density_matrix=np.ones((2, 2)) * 0.5,
        qubit_map={q0: 0})
    assert (repr(cirq.DensityMatrixTrialResult(
        params=cirq.ParamResolver({'s': 1}),
        measurements={'m': np.array([[1]])},
        final_simulator_state=final_simulator_state)) ==
            "cirq.DensityMatrixTrialResult("
            "params=cirq.ParamResolver({'s': 1}), "
            "measurements={'m': array([[1]])}, "
            "final_simulator_state=cirq.DensityMatrixSimulatorState("
                "density_matrix=np.array([[0.5, 0.5], [0.5, 0.5]]), "
                "qubit_map={cirq.LineQubit(0): 0}))""")


class XAsOp(cirq.Operation):

    def __init__(self, q):
        # coverage: ignore
        self.q = q

    @property
    def qubits(self):
        # coverage: ignore
        return self.q,

    def with_qubits(self, *new_qubits):
        # coverage: ignore
        return XAsOp(new_qubits[0])

    def _channel_(self):
        # coverage: ignore
        return cirq.channel(cirq.X)


def test_works_on_operation():

    class XAsOp(cirq.Operation):

        def __init__(self, q):
            # coverage: ignore
            self.q = q

        @property
        def qubits(self):
            # coverage: ignore
            return self.q,

        def with_qubits(self, *new_qubits):
            raise NotImplementedError()

        def _channel_(self):
            # coverage: ignore
            return cirq.channel(cirq.X)

    s = cirq.DensityMatrixSimulator()
    c = cirq.Circuit(XAsOp(cirq.LineQubit(0)))
    np.testing.assert_allclose(s.simulate(c).final_density_matrix,
                               np.diag([0, 1]),
                               atol=1e-8)


def test_works_on_operation_dephased():

    class HAsOp(cirq.Operation):

        def __init__(self, q):
            self.q = q

        @property
        def qubits(self):
            return self.q,

        def with_qubits(self, *new_qubits):
            raise NotImplementedError()

        def _channel_(self):
            return cirq.channel(cirq.H)

    s = cirq.DensityMatrixSimulator(ignore_measurement_results=True)
    c = cirq.Circuit(HAsOp(cirq.LineQubit(0)))
    np.testing.assert_allclose(s.simulate(c).final_density_matrix,
                               [[0.5 + 0.j, 0.5 + 0.j], [0.5 + 0.j, 0.5 + 0.j]],
                               atol=1e-8)


def test_works_on_pauli_string_phasor():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(np.exp(1j * np.pi * cirq.X(a) * cirq.X(b)))
    sim = cirq.DensityMatrixSimulator()
    result = sim.simulate(c).final_density_matrix
    np.testing.assert_allclose(result.reshape(4, 4),
                               np.diag([0, 0, 0, 1]),
                               atol=1e-8)

def test_works_on_pauli_string():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.X(a) * cirq.X(b))
    sim = cirq.DensityMatrixSimulator()
    result = sim.simulate(c).final_density_matrix
    np.testing.assert_allclose(result.reshape(4, 4),
                               np.diag([0, 0, 0, 1]),
                               atol=1e-8)


def test_density_matrix_trial_result_str():
    q0 = cirq.LineQubit(0)
    final_simulator_state = cirq.DensityMatrixSimulatorState(
        density_matrix=np.ones((2, 2)) * 0.5, qubit_map={q0: 0})
    result = cirq.DensityMatrixTrialResult(
        params=cirq.ParamResolver({}),
        measurements={},
        final_simulator_state=final_simulator_state)

    # numpy varies whitespace in its representation for different versions
    # Eliminate whitespace to harden tests against this variation
    result_no_whitespace = str(result).replace('\n', '').replace(' ', '')
    assert result_no_whitespace == ('measurements:(nomeasurements)'
                                    'finaldensitymatrix:'
                                    '[[0.50.5][0.50.5]]')


def test_run_sweep_parameters_not_resolved():
    a = cirq.LineQubit(0)
    simulator = cirq.DensityMatrixSimulator()
    circuit = cirq.Circuit(
        cirq.XPowGate(exponent=sympy.Symbol('a'))(a), cirq.measure(a))
    with pytest.raises(ValueError, match='symbols were not specified'):
        _ = simulator.run_sweep(circuit, cirq.ParamResolver({}))


def test_simulate_sweep_parameters_not_resolved():
    a = cirq.LineQubit(0)
    simulator = cirq.DensityMatrixSimulator()
    circuit = cirq.Circuit(
        cirq.XPowGate(exponent=sympy.Symbol('a'))(a), cirq.measure(a))
    with pytest.raises(ValueError, match='symbols were not specified'):
        _ = simulator.simulate_sweep(circuit, cirq.ParamResolver({}))


def test_random_seed():
    a = cirq.NamedQubit('a')
    circuit = cirq.Circuit(cirq.X(a)**0.5, cirq.measure(a))

    sim = cirq.DensityMatrixSimulator(seed=1234)
    result = sim.run(circuit, repetitions=10)
    assert np.all(
        result.measurements['a'] == [[False], [True], [False], [True], [True],
                                     [False], [False], [True], [True], [True]])

    sim = cirq.DensityMatrixSimulator(seed=np.random.RandomState(1234))
    result = sim.run(circuit, repetitions=10)
    assert np.all(
        result.measurements['a'] == [[False], [True], [False], [True], [True],
                                     [False], [False], [True], [True], [True]])


def test_random_seed_does_not_modify_global_state_terminal_measurements():
    a = cirq.NamedQubit('a')
    circuit = cirq.Circuit(cirq.X(a)**0.5, cirq.measure(a))

    sim = cirq.DensityMatrixSimulator(seed=1234)
    result1 = sim.run(circuit, repetitions=50)

    sim = cirq.DensityMatrixSimulator(seed=1234)
    _ = np.random.random()
    _ = random.random()
    result2 = sim.run(circuit, repetitions=50)

    assert result1 == result2


def test_random_seed_does_not_modify_global_state_non_terminal_measurements():
    a = cirq.NamedQubit('a')
    circuit = cirq.Circuit(
        cirq.X(a)**0.5, cirq.measure(a, key='a0'),
        cirq.X(a)**0.5, cirq.measure(a, key='a1'))

    sim = cirq.DensityMatrixSimulator(seed=1234)
    result1 = sim.run(circuit, repetitions=50)

    sim = cirq.DensityMatrixSimulator(seed=1234)
    _ = np.random.random()
    _ = random.random()
    result2 = sim.run(circuit, repetitions=50)

    assert result1 == result2


def test_random_seed_terminal_measurements_deterministic():
    a = cirq.NamedQubit('a')
    circuit = cirq.Circuit(cirq.X(a)**0.5, cirq.measure(a, key='a'))
    sim = cirq.DensityMatrixSimulator(seed=1234)
    result1 = sim.run(circuit, repetitions=30)
    result2 = sim.run(circuit, repetitions=30)
    assert np.all(result1.measurements['a'] ==
                  [[0], [1], [0], [1], [1], [0], [0], [1], [1], [1], [0], [1],
                   [1], [1], [0], [1], [1], [0], [1], [1], [0], [1], [0], [0],
                   [1], [1], [0], [1], [0], [1]])
    assert np.all(result2.measurements['a'] ==
                  [[1], [0], [1], [0], [1], [1], [0], [1], [0], [1], [0], [0],
                   [0], [1], [1], [1], [0], [1], [0], [1], [0], [1], [1], [0],
                   [1], [1], [1], [1], [1], [1]])


def test_random_seed_non_terminal_measurements_deterministic():
    a = cirq.NamedQubit('a')
    circuit = cirq.Circuit(
        cirq.X(a)**0.5, cirq.measure(a, key='a'),
        cirq.X(a)**0.5, cirq.measure(a, key='b'))
    sim = cirq.DensityMatrixSimulator(seed=1234)
    result = sim.run(circuit, repetitions=30)
    assert np.all(result.measurements['a'] ==
                  [[0], [0], [1], [0], [1], [0], [1], [0], [1], [1], [0], [0],
                   [1], [0], [0], [1], [1], [1], [0], [0], [0], [0], [1], [0],
                   [0], [0], [1], [1], [1], [1]])
    assert np.all(result.measurements['b'] ==
                  [[1], [1], [0], [1], [1], [1], [1], [1], [0], [1], [1], [0],
                   [1], [1], [1], [0], [0], [1], [1], [1], [0], [1], [1], [1],
                   [1], [1], [0], [1], [1], [1]])


def test_simulate_with_invert_mask():

    class PlusGate(cirq.Gate):
        """A qudit gate that increments a qudit state mod its dimension."""

        def __init__(self, dimension, increment=1):
            self.dimension = dimension
            self.increment = increment % dimension

        def _qid_shape_(self):
            return (self.dimension,)

        def _unitary_(self):
            inc = (self.increment - 1) % self.dimension + 1
            u = np.empty((self.dimension, self.dimension))
            u[inc:] = np.eye(self.dimension)[:-inc]
            u[:inc] = np.eye(self.dimension)[-inc:]
            return u

    q0, q1, q2, q3, q4 = cirq.LineQid.for_qid_shape((2, 3, 3, 3, 4))
    c = cirq.Circuit(
        PlusGate(2, 1)(q0),
        PlusGate(3, 1)(q2),
        PlusGate(3, 2)(q3),
        PlusGate(4, 3)(q4),
        cirq.measure(q0, q1, q2, q3, q4, key='a', invert_mask=(True,) * 4),
    )
    assert np.all(cirq.DensityMatrixSimulator().run(c).measurements['a'] ==
                  [[0, 1, 0, 2, 3]])


def test_simulate_noise_with_terminal_measurements():
    q = cirq.LineQubit(0)
    circuit1 = cirq.Circuit(cirq.measure(q))
    circuit2 = circuit1 + cirq.I(q)

    simulator = cirq.DensityMatrixSimulator(noise=cirq.X)
    result1 = simulator.run(circuit1, repetitions=10)
    result2 = simulator.run(circuit2, repetitions=10)

    assert result1 == result2
