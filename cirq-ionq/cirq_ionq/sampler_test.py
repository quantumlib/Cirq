# Copyright 2021 The Cirq Developers
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

import pandas as pd
import sympy as sp

import cirq
import cirq_ionq as ionq


def test_sampler_qpu():
    mock_service = mock.MagicMock()
    job_dict = {
        'id': '1',
        'status': 'completed',
        'qubits': '1',
        'target': 'qpu',
        'metadata': {'shots': 4, 'measurement0': f'a{chr(31)}0'},
    }

    job = ionq.Job(client=mock_service, job_dict=job_dict)
    mock_service.create_job.return_value = job
    mock_service.get_results.return_value = {'0': '0.25', '1': '0.75'}

    sampler = ionq.Sampler(service=mock_service, target='qpu')
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q0), cirq.measure(q0, key='a'))
    results = sampler.sample(program=circuit, repetitions=4)
    pd.testing.assert_frame_equal(
        results, pd.DataFrame(columns=['a'], index=[0, 1, 2, 3], data=[[0], [1], [1], [1]])
    )
    mock_service.create_job.assert_called_once_with(circuit=circuit, repetitions=4, target='qpu')


def test_sampler_simulator():
    mock_service = mock.MagicMock()
    job_dict = {
        'id': '1',
        'status': 'completed',
        'qubits': '1',
        'target': 'simulator',
        'metadata': {'shots': 4, 'measurement0': f'a{chr(31)}0'},
    }

    job = ionq.Job(client=mock_service, job_dict=job_dict)
    mock_service.create_job.return_value = job
    mock_service.get_results.return_value = {'0': '0.25', '1': '0.75'}

    sampler = ionq.Sampler(service=mock_service, target='simulator', seed=10)
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q0), cirq.measure(q0, key='a'))
    results = sampler.sample(program=circuit, repetitions=4)
    pd.testing.assert_frame_equal(
        results, pd.DataFrame(columns=['a'], index=[0, 1, 2, 3], data=[[1], [0], [1], [1]])
    )
    mock_service.create_job.assert_called_once_with(
        circuit=circuit, repetitions=4, target='simulator'
    )


def test_sampler_multiple_jobs():
    mock_service = mock.MagicMock()
    job_dict0 = {
        'id': '1',
        'status': 'completed',
        'qubits': '1',
        'target': 'qpu',
        'metadata': {'shots': 4, 'measurement0': f'a{chr(31)}0'},
    }
    job_dict1 = {
        'id': '1',
        'status': 'completed',
        'qubits': '1',
        'target': 'qpu',
        'metadata': {'shots': 4, 'measurement0': f'a{chr(31)}0'},
    }

    job0 = ionq.Job(client=mock_service, job_dict=job_dict0)
    job1 = ionq.Job(client=mock_service, job_dict=job_dict1)
    mock_service.create_job.side_effect = [job0, job1]
    mock_service.get_results.side_effect = [{'0': '0.25', '1': '0.75'}, {'0': '0.5', '1': '0.5'}]

    sampler = ionq.Sampler(service=mock_service, timeout_seconds=10, target='qpu')
    q0 = cirq.LineQubit(0)
    x = sp.Symbol('x')
    circuit = cirq.Circuit(cirq.X(q0) ** x, cirq.measure(q0, key='a'))
    results = sampler.sample(
        program=circuit,
        repetitions=4,
        params=[cirq.ParamResolver({x: 0.5}), cirq.ParamResolver({x: 0.6})],
    )
    pd.testing.assert_frame_equal(
        results,
        pd.DataFrame(
            columns=['x', 'a'],
            index=[0, 1, 2, 3] * 2,
            data=[[0.5, 0], [0.5, 1], [0.5, 1], [0.5, 1], [0.6, 0], [0.6, 0], [0.6, 1], [0.6, 1]],
        ),
    )
    circuit0 = cirq.Circuit(cirq.X(q0) ** 0.5, cirq.measure(q0, key='a'))
    circuit1 = cirq.Circuit(cirq.X(q0) ** 0.6, cirq.measure(q0, key='a'))
    mock_service.create_job.assert_has_calls(
        [
            mock.call(circuit=circuit0, repetitions=4, target='qpu'),
            mock.call(circuit=circuit1, repetitions=4, target='qpu'),
        ]
    )
    assert mock_service.create_job.call_count == 2


def test_sampler_run_sweep():
    mock_service = mock.MagicMock()
    job_dict = {
        'id': '1',
        'status': 'completed',
        'qubits': '1',
        'target': 'qpu',
        'metadata': {'shots': 4, 'measurement0': f'a{chr(31)}0'},
    }

    job = ionq.Job(client=mock_service, job_dict=job_dict)
    mock_service.create_job.return_value = job
    mock_service.get_results.return_value = {'0': '0.25', '1': '0.75'}

    sampler = ionq.Sampler(service=mock_service, target='qpu')
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q0) ** sp.Symbol('theta'), cirq.measure(q0, key='a'))

    sweep = cirq.Linspace(key='theta', start=0.0, stop=2.0, length=5)

    results = sampler.run_sweep(program=circuit, params=sweep, repetitions=4)
    assert len(results) == 5  # Confirm that 5 result objects were created
    assert all([result.repetitions == 4 for result in results])  # Confirm all repetitions were 4

    # Assert that create_job was called 5 times
    assert mock_service.create_job.call_count == 5
    # For each call, assert that it was called with the correct repetitions and target
    for call in mock_service.create_job.call_args_list:
        _, kwargs = call
        assert kwargs["repetitions"] == 4
        assert kwargs["target"] == 'qpu'
