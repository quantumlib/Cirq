import numpy as np
import pytest
import cirq
import classical_simulator
import sympy


class TestSimulator:
    def test_x_gate():
        q0 = cirq.LineQubit.range(1)
        circuit = cirq.Circuit()
        circuit = cirq.Circuit(cirq.X(q0), cirq.measure(cirq.LineQubit(q0), key='key'))
        expected_results = {'key': np.array([[[0]]], dtype=np.uint8)}
        sim = cirq.ClassicalSimulator()
        results = sim._run(circuit=circuit, param_resolver=None, repetitions=1)
        assert results == expected_results

    def test_CNOT():
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit()
        circuit.append(cirq.X(q0))
        circuit.append(cirq.CNOT(q0, q1))
        circuit.append(cirq.measure(q1, key='key'))
        expected_results = {'key': np.array([[[1]]], dtype=np.uint8)}
        sim = cirq.ClassicalSimulator()
        results = sim._run(circuit=circuit, param_resolver=None, repetitions=1)
        np.testing.assert_equal(results, expected_results)

    def test_measurement_gate():
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit()
        circuit.append(cirq.measure((q0, q1), key='key'))
        expected_results = {'key': np.array[[[0, 0]]]}
        sim = cirq.ClassicalSimulator()
        results = sim._run(circuit=circuit, param_resolver=None, repetitions=1)
        np.testing.assert_equal(results, expected_results)

    def test_qubit_order():
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit()
        circuit.append(cirq.CNOT(q0, q1))
        circuit.append(cirq.X(q0))
        circuit.append(cirq.measure((q0, q1), key='key'))
        expected_results = {'key': np.array([[[1, 0]]], dtype=np.uint8)}
        sim = cirq.ClassicalSimulator()
        results = sim._run(circuit=circuit, param_resolver=None, repetitions=1)
        np.testing.assert_equal(results, expected_results)

    def test_same_key_instances():
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit()
        circuit.append(cirq.measure((q0, q1), key='key'))
        circuit.append(cirq.X(q0))
        circuit.append(cirq.measure((q0, q1), key='key'))
        expected_results = {'key': np.array([[[0, 0], [1, 0]]], dtype=np.uint8)}
        sim = cirq.ClassicalSimulator()
        results = sim._run(circuit=circuit, param_resolver=None, repetitions=1)
        np.testing.assert_equal(results, expected_results)

    def test_same_key_instances_order():
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit()
        circuit.append(cirq.X(q0))
        circuit.append(cirq.measure((q0, q1), key='key'))
        circuit.append(cirq.X(q0))
        circuit.append(cirq.measure((q1, q0), key='key'))
        expected_results = {'key': np.array([[[1, 0], [0, 0]]], dtype=np.uint8)}
        sim = cirq.ClassicalSimulator()
        results = sim._run(circuit=circuit, param_resolver=None, repetitions=1)
        np.testing.assert_equal(results, expected_results)

    def test_repetitions():
        q0 = cirq.LineQubit.range(0)
        circuit = cirq.Circuit()
        circuit.append(cirq.measure(q0, key='key'))
        expected_results = {
            'key': np.array(
                [[[0]], [[0]], [[0]], [[0]], [[0]], [[0]], [[0]], [[0]], [[0]], [[0]]],
                dtype=np.uint8,
            )
        }
        sim = cirq.ClassicalSimulator()
        results = sim._run(circuit=circuit, param_resolver=None, repetitions=10)
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
        sim = cirq.ClassicalSimulator()
        results = sim._run(circuit=circuit, param_resolver=None, repetitions=1)
        np.testing.assert_equal(results, expected_results)

    def test_multiple_gates_order():
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit()
        circuit.append(cirq.X(q0))
        circuit.append(cirq.CNOT(q0, q1))
        circuit.append(cirq.CNOT(q1, q0))
        circuit.append(cirq.measure((q0, q1), key='key'))
        expected_results = {'key': np.array([[[0, 1]]], dtype=np.uint8)}
        sim = cirq.ClassicalSimulator()
        results = sim._run(circuit=circuit, param_resolver=None, repetitions=1)
        np.testing.assert_equal(results, expected_results)

    def test_param_resolver():
        gate = cirq.CNOT ** sympy.Symbol('t')
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit()
        circuit.append(cirq.X(q0))
        circuit.append(gate(q0, q1))
        circuit.append(cirq.measure((q1), key='key'))
        resolver = cirq.ParamResolver({'t': 0})
        sim = cirq.ClassicalSimulator()
        results_with_paramter_zero = sim._run(
            circuit=circuit, param_resolver=resolver, repetitions=1
        )
        resolver = cirq.ParamResolver({'t': 1})
        results_with_paramter_one = sim._run(
            circuit=circuit, param_resolver=resolver, repetitions=1
        )
        np.testing.assert_equal(
            results_with_paramter_zero, {'key': np.array([[[0]]], dtype=np.uint8)}
        )
        np.testing.assert_equal(
            results_with_paramter_one, {'key': np.array([[[1]]], dtype=np.uint8)}
        )

    def test_unknown_gates():
        gate = cirq.CNOT ** sympy.Symbol('t')
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit()
        circuit.append(gate(q0, q1))
        circuit.append(cirq.measure((q0), key='key'))
        resolver = cirq.ParamResolver({'t': 0.5})
        sim = cirq.ClassicalSimulator()
        with pytest.raises(
            ValueError, match="Can not simulate gates other than cirq.XGate or cirq.CNOT"
        ):
            sim._run(circuit=circuit, param_resolver=resolver, repetitions=1)
