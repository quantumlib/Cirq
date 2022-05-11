# Copyright 2020 The Cirq Developers
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

import pytest

import numpy as np

import cirq_ionq as ionq
import cirq.testing


def test_qpu_result_fields():
    result = ionq.QPUResult({0: 10, 1: 10}, num_qubits=1, measurement_dict={'a': [0]})
    assert result.counts() == {0: 10, 1: 10}
    assert result.repetitions() == 20
    assert result.num_qubits() == 1
    assert result.measurement_dict() == {'a': [0]}


def test_qpu_result_str():
    result = ionq.QPUResult({0: 10, 1: 10}, num_qubits=2, measurement_dict={})
    assert str(result) == '00: 10\n01: 10'


def test_qpu_result_eq():
    equals_tester = cirq.testing.EqualsTester()
    equals_tester.add_equality_group(
        ionq.QPUResult({0: 10, 1: 10}, num_qubits=1, measurement_dict={'a': [0]}),
        ionq.QPUResult({0: 10, 1: 10}, num_qubits=1, measurement_dict={'a': [0]}),
    )
    equals_tester.add_equality_group(
        ionq.QPUResult({0: 10, 1: 20}, num_qubits=1, measurement_dict={'a': [0]})
    )
    equals_tester.add_equality_group(
        ionq.QPUResult({0: 15, 1: 15}, num_qubits=1, measurement_dict={'a': [0]})
    )
    equals_tester.add_equality_group(
        ionq.QPUResult({0: 10, 1: 10}, num_qubits=2, measurement_dict={'a': [0]})
    )
    equals_tester.add_equality_group(
        ionq.QPUResult({0: 10, 1: 10}, num_qubits=1, measurement_dict={'b': [0]})
    )
    equals_tester.add_equality_group(
        ionq.QPUResult({0: 10, 1: 10}, num_qubits=1, measurement_dict={'a': [1]})
    )
    equals_tester.add_equality_group(
        ionq.QPUResult({0: 10, 1: 10}, num_qubits=1, measurement_dict={'a': [0, 1]})
    )


def test_qpu_result_measurement_key():
    result = ionq.QPUResult({0b00: 10, 0b01: 20}, num_qubits=2, measurement_dict={'a': [0]})
    assert result.counts() == {0b00: 10, 0b01: 20}
    result = ionq.QPUResult({0b00: 10, 0b01: 20}, num_qubits=2, measurement_dict={'a': [0]})
    assert result.counts('a') == {0b0: 30}
    result = ionq.QPUResult({0b00: 10, 0b01: 20}, num_qubits=2, measurement_dict={'a': [1]})
    assert result.counts('a') == {0b0: 10, 0b1: 20}
    result = ionq.QPUResult({0b00: 10, 0b01: 20}, num_qubits=2, measurement_dict={'a': [0, 1]})
    assert result.counts('a') == {0b00: 10, 0b01: 20}
    result = ionq.QPUResult({0b00: 10, 0b01: 20}, num_qubits=2, measurement_dict={'a': [1, 0]})
    assert result.counts('a') == {0b00: 10, 0b10: 20}
    result = ionq.QPUResult({0b000: 10, 0b111: 20}, num_qubits=3, measurement_dict={'a': [2]})
    assert result.counts('a') == {0b0: 10, 0b1: 20}
    result = ionq.QPUResult({0b000: 10, 0b100: 20}, num_qubits=3, measurement_dict={'a': [1, 2]})
    assert result.counts('a') == {0b00: 30}


def test_qpu_result_measurement_multiple_key():
    result = ionq.QPUResult(
        {0b00: 10, 0b01: 20}, num_qubits=2, measurement_dict={'a': [0], 'b': [1]}
    )
    assert result.counts('a') == {0b0: 30}
    assert result.counts('b') == {0b0: 10, 0b1: 20}
    result = ionq.QPUResult(
        {0b00: 10, 0b01: 20}, num_qubits=2, measurement_dict={'a': [1], 'b': [0]}
    )
    assert result.counts('a') == {0b0: 10, 0b1: 20}
    assert result.counts('b') == {0b0: 30}


def test_qpu_result_bad_measurement_key():
    result = ionq.QPUResult(
        {0b00: 10, 0b01: 20}, num_qubits=2, measurement_dict={'a': [0], 'b': [1]}
    )
    with pytest.raises(ValueError, match='bad'):
        result.counts('bad')


def test_qpu_result_to_cirq_result():
    result = ionq.QPUResult({0b00: 1, 0b01: 2}, num_qubits=2, measurement_dict={'x': [0, 1]})
    assert result.to_cirq_result() == cirq.ResultDict(
        params=cirq.ParamResolver({}), measurements={'x': np.array([[0, 0], [0, 1], [0, 1]])}
    )
    params = cirq.ParamResolver({'a': 0.1})
    assert result.to_cirq_result(params) == cirq.ResultDict(
        params=params, measurements={'x': np.array([[0, 0], [0, 1], [0, 1]])}
    )
    result = ionq.QPUResult({0b00: 1, 0b01: 2}, num_qubits=2, measurement_dict={'x': [0]})
    assert result.to_cirq_result() == cirq.ResultDict(
        params=cirq.ParamResolver({}), measurements={'x': np.array([[0], [0], [0]])}
    )
    result = ionq.QPUResult({0b00: 1, 0b01: 2}, num_qubits=2, measurement_dict={'x': [1]})
    assert result.to_cirq_result() == cirq.ResultDict(
        params=cirq.ParamResolver({}), measurements={'x': np.array([[0], [1], [1]])}
    )
    # cirq.Result only compares pandas data frame, so possible to have supplied an list of
    # list instead of a numpy multidimensional array. Check this here.
    assert type(result.to_cirq_result().measurements['x']) == np.ndarray
    # Results bitstreams need to be consistent betwween measurement keys
    # Ordering is by bitvector, so 0b01 0b01 0b10 should be the ordering for all measurement dicts.
    result = ionq.QPUResult(
        {0b10: 1, 0b01: 2}, num_qubits=2, measurement_dict={'x': [0, 1], 'y': [0], 'z': [1]}
    )
    assert result.to_cirq_result() == cirq.ResultDict(
        params=cirq.ParamResolver({}),
        measurements={
            'x': np.array([[0, 1], [0, 1], [1, 0]]),
            'y': np.array([[0], [0], [1]]),
            'z': np.array([[1], [1], [0]]),
        },
    )


def test_qpu_result_to_cirq_result_multiple_keys():
    result = ionq.QPUResult(
        {0b000: 2, 0b101: 3}, num_qubits=3, measurement_dict={'x': [1], 'y': [2, 0]}
    )
    assert result.to_cirq_result() == cirq.ResultDict(
        params=cirq.ParamResolver({}),
        measurements={
            'x': np.array([[0], [0], [0], [0], [0]]),
            'y': np.array([[0, 0], [0, 0], [1, 1], [1, 1], [1, 1]]),
        },
    )


def test_qpu_result_to_cirq_result_no_keys():
    result = ionq.QPUResult({0b00: 1, 0b01: 2}, num_qubits=2, measurement_dict={})
    with pytest.raises(ValueError, match='cirq results'):
        _ = result.to_cirq_result()


def test_ordered_results_invalid_key():
    result = ionq.QPUResult({0b00: 1, 0b01: 2}, num_qubits=2, measurement_dict={'x': [1]})
    with pytest.raises(ValueError, match='is not a key for'):
        _ = result.ordered_results('y')


def test_simulator_result_fields():
    result = ionq.SimulatorResult(
        {0: 0.4, 1: 0.6}, num_qubits=1, measurement_dict={'a': [0]}, repetitions=100
    )
    assert result.probabilities() == {0: 0.4, 1: 0.6}
    assert result.num_qubits() == 1
    assert result.measurement_dict() == {'a': [0]}
    assert result.repetitions() == 100


def test_simulator_result_str():
    result = ionq.SimulatorResult(
        {0: 0.4, 1: 0.6}, num_qubits=2, measurement_dict={'a': [0]}, repetitions=100
    )
    assert str(result) == '00: 0.4\n01: 0.6'


def test_simulator_result_eq():
    equals_tester = cirq.testing.EqualsTester()
    equals_tester.add_equality_group(
        ionq.SimulatorResult(
            {0: 0.5, 1: 0.5}, num_qubits=1, measurement_dict={'a': [0]}, repetitions=100
        ),
        ionq.SimulatorResult(
            {0: 0.5, 1: 0.5}, num_qubits=1, measurement_dict={'a': [0]}, repetitions=100
        ),
    )
    equals_tester.add_equality_group(
        ionq.SimulatorResult(
            {0: 0.4, 1: 0.6}, num_qubits=1, measurement_dict={'a': [0]}, repetitions=100
        )
    )
    equals_tester.add_equality_group(
        ionq.SimulatorResult(
            {0: 0.5, 1: 0.5}, num_qubits=2, measurement_dict={'a': [0]}, repetitions=100
        )
    )
    equals_tester.add_equality_group(
        ionq.SimulatorResult(
            {0: 0.5, 1: 0.5}, num_qubits=1, measurement_dict={'b': [0]}, repetitions=100
        )
    )
    equals_tester.add_equality_group(
        ionq.SimulatorResult(
            {0: 0.5, 1: 0.5}, num_qubits=1, measurement_dict={'a': [1]}, repetitions=100
        )
    )
    equals_tester.add_equality_group(
        ionq.SimulatorResult(
            {0: 0.5, 1: 0.5}, num_qubits=1, measurement_dict={'a': [0, 1]}, repetitions=100
        )
    )
    equals_tester.add_equality_group(
        ionq.SimulatorResult(
            {0: 0.5, 1: 0.5}, num_qubits=1, measurement_dict={'a': [0, 1]}, repetitions=10
        )
    )


def test_simulator_result_measurement_key():
    result = ionq.SimulatorResult(
        {0b00: 0.2, 0b01: 0.8}, num_qubits=2, measurement_dict={'a': [0]}, repetitions=100
    )
    assert result.probabilities() == {0b00: 0.2, 0b01: 0.8}
    result = ionq.SimulatorResult(
        {0b00: 0.2, 0b01: 0.8}, num_qubits=2, measurement_dict={'a': [0]}, repetitions=100
    )
    assert result.probabilities('a') == {0b0: 1.0}

    result = ionq.SimulatorResult(
        {0b00: 0.2, 0b01: 0.8}, num_qubits=2, measurement_dict={'a': [1]}, repetitions=100
    )
    assert result.probabilities('a') == {0b0: 0.2, 0b1: 0.8}
    result = ionq.SimulatorResult(
        {0b00: 0.2, 0b01: 0.8}, num_qubits=2, measurement_dict={'a': [0, 1]}, repetitions=100
    )
    assert result.probabilities('a') == {0b00: 0.2, 0b01: 0.8}
    result = ionq.SimulatorResult(
        {0b00: 0.2, 0b01: 0.8}, num_qubits=2, measurement_dict={'a': [1, 0]}, repetitions=100
    )
    assert result.probabilities('a') == {0b00: 0.2, 0b10: 0.8}
    result = ionq.SimulatorResult(
        {0b000: 0.2, 0b111: 0.8}, num_qubits=3, measurement_dict={'a': [2]}, repetitions=100
    )
    assert result.probabilities('a') == {0b0: 0.2, 0b1: 0.8}
    result = ionq.SimulatorResult(
        {0b000: 0.2, 0b100: 0.8}, num_qubits=3, measurement_dict={'a': [1, 2]}, repetitions=100
    )
    assert result.probabilities('a') == {0b00: 1.0}


def test_simulator_result_measurement_multiple_key():
    result = ionq.SimulatorResult(
        {0b00: 0.2, 0b01: 0.8}, num_qubits=2, measurement_dict={'a': [0], 'b': [1]}, repetitions=100
    )
    assert result.probabilities('a') == {0b0: 1.0}
    assert result.probabilities('b') == {0b0: 0.2, 0b1: 0.8}
    result = ionq.SimulatorResult(
        {0b00: 0.2, 0b01: 0.8}, num_qubits=2, measurement_dict={'a': [1], 'b': [0]}, repetitions=100
    )
    assert result.probabilities('a') == {0b0: 0.2, 0b1: 0.8}
    assert result.probabilities('b') == {0b0: 1.0}


def test_simulator_result_bad_measurement_key():
    result = ionq.SimulatorResult(
        {0b00: 0.2, 0b01: 0.8}, num_qubits=2, measurement_dict={'a': [0], 'b': [1]}, repetitions=100
    )
    with pytest.raises(ValueError, match='bad'):
        result.probabilities('bad')


def test_simulator_result_to_cirq_result():
    result = ionq.SimulatorResult(
        {0b00: 0.25, 0b01: 0.75}, num_qubits=2, measurement_dict={'x': [0, 1]}, repetitions=3
    )
    assert result.to_cirq_result(seed=2) == cirq.ResultDict(
        params=cirq.ParamResolver({}), measurements={'x': np.array([[0, 1], [0, 0], [0, 1]])}
    )
    assert result.to_cirq_result(seed=3) == cirq.ResultDict(
        params=cirq.ParamResolver({}), measurements={'x': np.array([[0, 1], [0, 1], [0, 1]])}
    )
    params = cirq.ParamResolver({'a': 0.1})
    assert result.to_cirq_result(seed=3, params=params) == cirq.ResultDict(
        params=params, measurements={'x': np.array([[0, 1], [0, 1], [0, 1]])}
    )
    assert result.to_cirq_result(seed=2, override_repetitions=2) == cirq.ResultDict(
        params=cirq.ParamResolver({}), measurements={'x': np.array([[0, 1], [0, 0]])}
    )

    result = ionq.SimulatorResult(
        {0b00: 0.25, 0b01: 0.75}, num_qubits=2, measurement_dict={'x': [0]}, repetitions=3
    )
    assert result.to_cirq_result(seed=2) == cirq.ResultDict(
        params=cirq.ParamResolver({}), measurements={'x': np.array([[0], [0], [0]])}
    )
    result = ionq.SimulatorResult(
        {0b00: 0.25, 0b01: 0.75}, num_qubits=2, measurement_dict={'x': [1]}, repetitions=3
    )
    assert result.to_cirq_result(seed=2) == cirq.ResultDict(
        params=cirq.ParamResolver({}), measurements={'x': np.array([[1], [0], [1]])}
    )
    # cirq.Result only compares pandas data frame, so possible to have supplied an list of
    # list instead of a numpy multidimensional array. Check this here.
    assert type(result.to_cirq_result().measurements['x']) == np.ndarray


def test_simulator_result_to_cirq_result_multiple_keys():
    result = ionq.SimulatorResult(
        {0b000: 0.25, 0b011: 0.75},
        num_qubits=3,
        measurement_dict={'x': [1], 'y': [2, 0]},
        repetitions=3,
    )
    assert result.to_cirq_result(seed=2) == cirq.ResultDict(
        params=cirq.ParamResolver({}),
        measurements={'x': np.array([[1], [0], [1]]), 'y': np.array([[1, 0], [0, 0], [1, 0]])},
    )


def test_simulator_result_to_cirq_result_no_keys():
    result = ionq.SimulatorResult(
        {0b00: 1, 0b01: 2}, num_qubits=2, measurement_dict={}, repetitions=3
    )
    with pytest.raises(ValueError, match='cirq results'):
        _ = result.to_cirq_result()
