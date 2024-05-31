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

from itertools import product
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


@pytest.mark.parametrize(['initial_state'], [(list(x),) for x in product([0, 1], repeat=4)])
def test_CCCX(initial_state):
    CCCX = cirq.CCNOT.controlled()
    qubits = cirq.LineQubit.range(4)

    circuit = cirq.Circuit()
    circuit.append(CCCX(*qubits))
    circuit.append(cirq.measure(qubits, key='key'))

    final_state = initial_state.copy()
    final_state[-1] ^= all(final_state[:-1])

    sim = cirq.ClassicalStateSimulator()
    results = sim.simulate(circuit, initial_state=initial_state).measurements['key']
    np.testing.assert_equal(results, final_state)


@pytest.mark.parametrize(['initial_state'], [(list(x),) for x in product([0, 1], repeat=3)])
def test_CSWAP(initial_state):
    CSWAP = cirq.SWAP.controlled()
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit()

    circuit = cirq.Circuit()
    circuit.append(CSWAP(*qubits))
    circuit.append(cirq.measure(qubits, key='key'))

    a, b, c = initial_state
    if a:
        b, c = c, b
    final_state = [a, b, c]

    sim = cirq.ClassicalStateSimulator()
    results = sim.simulate(circuit, initial_state=initial_state).measurements['key']
    np.testing.assert_equal(results, final_state)


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


def test_simulate_sweeps_param_resolver():
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.ClassicalStateSimulator()
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit(
                (cirq.X ** sympy.Symbol('b0'))(q0), (cirq.X ** sympy.Symbol('b1'))(q1)
            )
            params = [
                cirq.ParamResolver({'b0': b0, 'b1': b1}),
                cirq.ParamResolver({'b0': b1, 'b1': b0}),
            ]
            results = simulator.simulate_sweep(circuit, params=params)

            assert results[0].params == params[0]
            assert results[1].params == params[1]


def test_create_partial_simulation_state_from_int_with_no_qubits():
    sim = cirq.ClassicalStateSimulator()
    initial_state = 5
    qs = None
    classical_data = cirq.value.ClassicalDataDictionaryStore()
    with pytest.raises(ValueError):
        sim._create_partial_simulation_state(
            initial_state=initial_state, qubits=qs, classical_data=classical_data
        )


def test_create_partial_simulation_state_from_invalid_state():
    sim = cirq.ClassicalStateSimulator()
    initial_state = None
    qs = cirq.LineQubit.range(2)
    classical_data = cirq.value.ClassicalDataDictionaryStore()
    with pytest.raises(ValueError):
        sim._create_partial_simulation_state(
            initial_state=initial_state, qubits=qs, classical_data=classical_data
        )


def test_create_partial_simulation_state_from_int():
    sim = cirq.ClassicalStateSimulator()
    initial_state = 15
    qs = cirq.LineQubit.range(4)
    classical_data = cirq.value.ClassicalDataDictionaryStore()
    expected_result = [1, 1, 1, 1]
    result = sim._create_partial_simulation_state(
        initial_state=initial_state, qubits=qs, classical_data=classical_data
    )._state.basis
    assert result == expected_result


def test_create_valid_partial_simulation_state_from_list():
    sim = cirq.ClassicalStateSimulator()
    initial_state = [1, 1, 1, 1]
    qs = cirq.LineQubit.range(4)
    classical_data = cirq.value.ClassicalDataDictionaryStore()
    expected_result = [1, 1, 1, 1]
    result = sim._create_partial_simulation_state(
        initial_state=initial_state, qubits=qs, classical_data=classical_data
    )._state.basis
    assert result == expected_result


def test_create_valid_partial_simulation_state_from_np():
    sim = cirq.ClassicalStateSimulator()
    initial_state = np.array([1, 1])
    qs = cirq.LineQubit.range(2)
    classical_data = cirq.value.ClassicalDataDictionaryStore()
    sim_state = sim._create_partial_simulation_state(
        initial_state=initial_state, qubits=qs, classical_data=classical_data
    )
    sim_state._act_on_fallback_(action=cirq.CX, qubits=qs)
    result = sim_state._state.basis
    expected_result = np.array([1, 0])
    np.testing.assert_equal(result, expected_result)


def test_create_invalid_partial_simulation_state_from_np():
    initial_state = np.array([[1, 1], [1, 1]])
    qs = cirq.LineQubit.range(2)
    classical_data = cirq.value.ClassicalDataDictionaryStore()
    sim = cirq.ClassicalStateSimulator()
    sim_state = sim._create_partial_simulation_state(
        initial_state=initial_state, qubits=qs, classical_data=classical_data
    )
    with pytest.raises(ValueError):
        sim_state._act_on_fallback_(action=cirq.CX, qubits=qs)


def test_noise_model():
    noise_model = cirq.NoiseModel.from_noise_model_like(cirq.depolarize(p=0.01))
    with pytest.raises(ValueError):
        cirq.ClassicalStateSimulator(noise=noise_model)
