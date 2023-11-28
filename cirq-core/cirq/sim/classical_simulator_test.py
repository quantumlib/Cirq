# Copyright 2023 The Cirq Developers
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
import pytest
import cirq
import sympy


def test_x_gate():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit()
    circuit.append(cirq.X(q0))
    circuit.append(cirq.X(q1))
    circuit.append(cirq.X(q1))
    circuit.append(cirq.measure((q0, q1), key='key'))
    expected_results = {'key': np.array([[[1, 0]]], dtype=np.uint8)}
    sim = cirq.ClassicalStateSimulator()
    results = sim.run(circuit, param_resolver=None, repetitions=1).records
    np.testing.assert_equal(results, expected_results)


def test_CNOT():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit()
    circuit.append(cirq.X(q0))
    circuit.append(cirq.CNOT(q0, q1))
    circuit.append(cirq.measure(q1, key='key'))
    expected_results = {'key': np.array([[[1]]], dtype=np.uint8)}
    sim = cirq.ClassicalStateSimulator()
    results = sim.run(circuit, param_resolver=None, repetitions=1).records
    np.testing.assert_equal(results, expected_results)


def test_Swap():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit()
    circuit.append(cirq.X(q0))
    circuit.append(cirq.SWAP(q0, q1))
    circuit.append(cirq.measure((q0, q1), key='key'))
    expected_results = {'key': np.array([[[0, 1]]], dtype=np.uint8)}
    sim = cirq.ClassicalStateSimulator()
    results = sim.run(circuit, param_resolver=None, repetitions=1).records
    np.testing.assert_equal(results, expected_results)


def test_CCNOT():
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit()
    circuit.append(cirq.CCNOT(q0, q1, q2))
    circuit.append(cirq.measure((q0, q1, q2), key='key'))
    circuit.append(cirq.X(q0))
    circuit.append(cirq.CCNOT(q0, q1, q2))
    circuit.append(cirq.measure((q0, q1, q2), key='key'))
    circuit.append(cirq.X(q1))
    circuit.append(cirq.X(q0))
    circuit.append(cirq.CCNOT(q0, q1, q2))
    circuit.append(cirq.measure((q0, q1, q2), key='key'))
    circuit.append(cirq.X(q0))
    circuit.append(cirq.CCNOT(q0, q1, q2))
    circuit.append(cirq.measure((q0, q1, q2), key='key'))
    expected_results = {
        'key': np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]]], dtype=np.uint8)
    }
    sim = cirq.ClassicalStateSimulator()
    results = sim.run(circuit, param_resolver=None, repetitions=1).records
    np.testing.assert_equal(results, expected_results)


def test_measurement_gate():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit()
    circuit.append(cirq.measure((q0, q1), key='key'))
    expected_results = {'key': np.array([[[0, 0]]], dtype=np.uint8)}
    sim = cirq.ClassicalStateSimulator()
    results = sim.run(circuit, param_resolver=None, repetitions=1).records
    np.testing.assert_equal(results, expected_results)


def test_qubit_order():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit()
    circuit.append(cirq.CNOT(q0, q1))
    circuit.append(cirq.X(q0))
    circuit.append(cirq.measure((q0, q1), key='key'))
    expected_results = {'key': np.array([[[1, 0]]], dtype=np.uint8)}
    sim = cirq.ClassicalStateSimulator()
    results = sim.run(circuit, param_resolver=None, repetitions=1).records
    np.testing.assert_equal(results, expected_results)


def test_same_key_instances():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit()
    circuit.append(cirq.measure((q0, q1), key='key'))
    circuit.append(cirq.X(q0))
    circuit.append(cirq.measure((q0, q1), key='key'))
    expected_results = {'key': np.array([[[0, 0], [1, 0]]], dtype=np.uint8)}
    sim = cirq.ClassicalStateSimulator()
    results = sim.run(circuit, param_resolver=None, repetitions=1).records
    np.testing.assert_equal(results, expected_results)


def test_same_key_instances_order():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit()
    circuit.append(cirq.X(q0))
    circuit.append(cirq.measure((q0, q1), key='key'))
    circuit.append(cirq.X(q0))
    circuit.append(cirq.measure((q1, q0), key='key'))
    expected_results = {'key': np.array([[[1, 0], [0, 0]]], dtype=np.uint8)}
    sim = cirq.ClassicalStateSimulator()
    results = sim.run(circuit, param_resolver=None, repetitions=1).records
    np.testing.assert_equal(results, expected_results)


def test_repetitions():
    q0 = cirq.LineQubit.range(1)
    circuit = cirq.Circuit()
    circuit.append(cirq.measure(q0, key='key'))
    expected_results = {
        'key': np.array(
            [[[0]], [[0]], [[0]], [[0]], [[0]], [[0]], [[0]], [[0]], [[0]], [[0]]], dtype=np.uint8
        )
    }
    sim = cirq.ClassicalStateSimulator()
    results = sim.run(circuit, param_resolver=None, repetitions=10).records
    np.testing.assert_equal(results, expected_results)


def test_multiple_gates():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit()
    circuit.append(cirq.X(q0))
    circuit.append(cirq.CNOT(q0, q1))
    circuit.append(cirq.CNOT(q0, q1))
    circuit.append(cirq.CNOT(q0, q1))
    circuit.append(cirq.X(q1))
    circuit.append(cirq.measure((q0, q1), key='key'))
    expected_results = {'key': np.array([[[1, 0]]], dtype=np.uint8)}
    sim = cirq.ClassicalStateSimulator()
    results = sim.run(circuit, param_resolver=None, repetitions=1).records
    np.testing.assert_equal(results, expected_results)


def test_multiple_gates_order():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit()
    circuit.append(cirq.X(q0))
    circuit.append(cirq.CNOT(q0, q1))
    circuit.append(cirq.CNOT(q1, q0))
    circuit.append(cirq.measure((q0, q1), key='key'))
    expected_results = {'key': np.array([[[0, 1]]], dtype=np.uint8)}
    sim = cirq.ClassicalStateSimulator()
    results = sim.run(circuit, param_resolver=None, repetitions=1).records
    np.testing.assert_equal(results, expected_results)


def test_param_resolver():
    gate = cirq.CNOT ** sympy.Symbol('t')
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit()
    circuit.append(cirq.X(q0))
    circuit.append(gate(q0, q1))
    circuit.append(cirq.measure((q1), key='key'))
    resolver = cirq.ParamResolver({'t': 0})
    sim = cirq.ClassicalStateSimulator()
    results_with_parameter_zero = sim.run(circuit, param_resolver=resolver, repetitions=1).records
    resolver = cirq.ParamResolver({'t': 1})
    results_with_parameter_one = sim.run(circuit, param_resolver=resolver, repetitions=1).records
    np.testing.assert_equal(results_with_parameter_zero, {'key': np.array([[[0]]], dtype=np.uint8)})
    np.testing.assert_equal(results_with_parameter_one, {'key': np.array([[[1]]], dtype=np.uint8)})


def test_unknown_gates():
    gate = cirq.Y
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(gate(q), cirq.measure((q), key='key'))
    sim = cirq.ClassicalStateSimulator()
    with pytest.raises(ValueError):
        _ = sim.run(circuit).records


def test_incompatible_measurements():
    qs = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.measure(qs, key='key'), cirq.measure(qs[0], key='key'))
    sim = cirq.ClassicalStateSimulator()
    with pytest.raises(ValueError):
        _ = sim.run(c)


def test_compatible_measurement():
    qs = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.measure(qs, key='key'), cirq.X.on_each(qs), cirq.measure(qs, key='key'))
    sim = cirq.ClassicalStateSimulator()
    res = sim.run(c, repetitions=3).records
    np.testing.assert_equal(res['key'], np.array([[[0, 0], [1, 1]]] * 3, dtype=np.uint8))
