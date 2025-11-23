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

from __future__ import annotations

import datetime
import json
import os
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import sympy

import cirq
import cirq_ionq as ionq


@pytest.mark.parametrize(
    'target,expected_results', [('qpu', [[0], [1], [1], [1]]), ('simulator', [[1], [0], [1], [1]])]
)
def test_service_run(target, expected_results):
    service = ionq.Service(remote_host='http://example.com', api_key='key')
    mock_client = mock.MagicMock()
    mock_client.create_job.return_value = {'id': 'job_id', 'status': 'ready'}
    mock_client.get_job.return_value = {
        'id': 'job_id',
        'status': 'completed',
        'backend': target,
        'metadata': {'shots': '4', 'measurement0': f'a{chr(31)}0'},
        'stats': {'qubits': '1'},
    }
    mock_client.get_results.return_value = {'0': '0.25', '1': '0.75'}
    service._client = mock_client

    a = sympy.Symbol('a')
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit((cirq.X**a)(q), cirq.measure(q, key='a'))
    params = cirq.ParamResolver({'a': 0.5})
    result = service.run(
        circuit=circuit, repetitions=4, target=target, name='bacon', param_resolver=params, seed=2
    )
    assert result == cirq.ResultDict(params=params, measurements={'a': np.array(expected_results)})

    create_job_kwargs = mock_client.create_job.call_args[1]
    # Serialization induces a float, so we don't validate full circuit.
    assert create_job_kwargs['serialized_program'].input['qubits'] == 1
    assert create_job_kwargs['serialized_program'].metadata == {'measurement0': f'a{chr(31)}0'}
    assert create_job_kwargs['repetitions'] == 4
    assert create_job_kwargs['target'] == target
    assert create_job_kwargs['name'] == 'bacon'


@pytest.mark.parametrize(
    'target,expected_results1,expected_results2',
    [
        ('qpu', [[0], [1], [1], [1]], [[1], [1], [1], [1]]),
        ('simulator', [[1], [0], [1], [1]], [[1], [1], [1], [1]]),
    ],
)
def test_service_run_batch(target, expected_results1, expected_results2):
    service = ionq.Service(remote_host='http://example.com', api_key='key')
    mock_client = mock.MagicMock()
    mock_client.create_job.return_value = {'id': 'job_id', 'status': 'ready'}
    mock_client.get_job.return_value = {
        'id': 'job_id',
        'status': 'completed',
        'backend': target,
        'metadata': {
            'shots': '4',
            'measurements': (
                "[{\"measurement0\": \"a\\u001f0\"}, {\"measurement0\": \"b\\u001f0\"}]"
            ),
            'qubit_numbers': '[1, 1]',
        },
        'stats': {'qubits': '1'},
    }
    mock_client.get_results.return_value = {
        "xxx": {'0': '0.25', '1': '0.75'},
        "yyy": {'0': '0', '1': '1'},
    }
    service._client = mock_client

    a = sympy.Symbol('a')
    b = sympy.Symbol('b')
    q = cirq.LineQubit(0)
    circuits = [
        cirq.Circuit((cirq.X**a)(q), cirq.measure(q, key='a')),
        cirq.Circuit((cirq.X**b)(q), cirq.measure(q, key='b')),
    ]
    params = cirq.ParamResolver({'a': 0.5, 'b': 1})
    result = service.run_batch(
        circuits=circuits, repetitions=4, target=target, name='bacon', param_resolver=params, seed=2
    )
    assert result[0] == cirq.ResultDict(
        params=params, measurements={'a': np.array(expected_results1)}
    )
    assert result[1] == cirq.ResultDict(
        params=params, measurements={'b': np.array(expected_results2)}
    )

    create_job_kwargs = mock_client.create_job.call_args[1]
    # Serialization induces a float, so we don't validate full circuit.
    assert create_job_kwargs['serialized_program'].input['qubits'] == 1
    assert create_job_kwargs['serialized_program'].metadata == {
        'measurements': "[{\"measurement0\": \"a\\u001f0\"}, {\"measurement0\": \"b\\u001f0\"}]",
        'qubit_numbers': '[1, 1]',
    }
    assert create_job_kwargs['repetitions'] == 4
    assert create_job_kwargs['target'] == target
    assert create_job_kwargs['name'] == 'bacon'


def test_sampler():
    service = ionq.Service(remote_host='http://example.com', api_key='key')
    mock_client = mock.MagicMock()
    service._client = mock_client
    job_dict = {
        'id': '1',
        'status': 'completed',
        'stats': {'qubits': '1'},
        'backend': 'qpu',
        'metadata': {'shots': 4, 'measurement0': f'a{chr(31)}0'},
    }
    mock_client.create_job.return_value = job_dict
    mock_client.get_job.return_value = job_dict
    mock_client.get_results.return_value = {'0': '0.25', '1': '0.75'}

    sampler = service.sampler(target='qpu', seed=10)

    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q0), cirq.measure(q0, key='a'))
    results = sampler.sample(program=circuit, repetitions=4)
    pd.testing.assert_frame_equal(
        results, pd.DataFrame(columns=['a'], index=[0, 1, 2, 3], data=[[0], [1], [1], [1]])
    )
    mock_client.create_job.assert_called_once()


def test_service_get_job():
    service = ionq.Service(remote_host='http://example.com', api_key='key')
    mock_client = mock.MagicMock()
    job_dict = {'id': 'job_id', 'status': 'ready'}
    mock_client.get_job.return_value = job_dict
    service._client = mock_client

    job = service.get_job('job_id')
    assert job.job_id() == 'job_id'
    mock_client.get_job.assert_called_with(job_id='job_id')


def test_service_create_job():
    service = ionq.Service(remote_host='http://example.com', api_key='key')
    mock_client = mock.MagicMock()
    mock_client.create_job.return_value = {'id': 'job_id', 'status': 'ready'}
    mock_client.get_job.return_value = {'id': 'job_id', 'status': 'completed'}
    service._client = mock_client

    circuit = cirq.Circuit(cirq.X(cirq.LineQubit(0)))
    job = service.create_job(circuit=circuit, repetitions=100, target='qpu', name='bacon')
    assert job.status() == 'completed'
    create_job_kwargs = mock_client.create_job.call_args[1]
    # Serialization induces a float, so we don't validate full circuit.
    assert create_job_kwargs['serialized_program'].input['qubits'] == 1
    assert create_job_kwargs['repetitions'] == 100
    assert create_job_kwargs['target'] == 'qpu'
    assert create_job_kwargs['name'] == 'bacon'


def test_service_list_jobs():
    service = ionq.Service(remote_host='http://example.com', api_key='key')
    mock_client = mock.MagicMock()
    jobs = [{'id': '1'}, {'id': '2'}]
    mock_client.list_jobs.return_value = jobs
    service._client = mock_client

    listed_jobs = service.list_jobs(status='completed', limit=10, batch_size=2)
    assert listed_jobs[0].job_id() == '1'
    assert listed_jobs[1].job_id() == '2'
    mock_client.list_jobs.assert_called_with(status='completed', limit=10, batch_size=2)


def test_service_get_current_calibration():
    service = ionq.Service(remote_host='http://example.com', api_key='key')
    mock_client = mock.MagicMock()
    calibration_dict = {'qubits': 11}
    mock_client.get_current_calibration.return_value = calibration_dict
    service._client = mock_client

    cal = service.get_current_calibration()
    assert cal.num_qubits() == 11
    mock_client.get_current_calibration.assert_called_once()


def test_service_list_calibrations():
    service = ionq.Service(remote_host='http://example.com', api_key='key')
    mock_client = mock.MagicMock()
    calibrations = [{'id': '1', 'qubits': '1'}, {'id': '2', 'qubits': 2}]
    mock_client.list_calibrations.return_value = calibrations
    service._client = mock_client
    start = datetime.datetime.fromtimestamp(1284286794, datetime.UTC)
    end = datetime.datetime.fromtimestamp(1284286795, datetime.UTC)

    listed_calibrations = service.list_calibrations(start=start, end=end, limit=10, batch_size=2)
    assert listed_calibrations[0].num_qubits() == 1
    assert listed_calibrations[1].num_qubits() == 2
    mock_client.list_calibrations.assert_called_with(start=start, end=end, limit=10, batch_size=2)


@mock.patch.dict(os.environ, {'IONQ_API_KEY': 'tomyheart'})
def test_service_api_key_via_env():
    service = ionq.Service(remote_host='http://example.com')
    assert service.api_key == 'tomyheart'


@mock.patch.dict(os.environ, {'IONQ_REMOTE_HOST': 'http://example.com'})
def test_service_remote_host_via_env():
    service = ionq.Service(api_key='tomyheart')
    assert service.remote_host == 'http://example.com'


@mock.patch.dict(os.environ, {}, clear=True)
def test_service_no_param_or_env_variable():
    with pytest.raises(EnvironmentError):
        _ = ionq.Service(remote_host='http://example.com')


@mock.patch.dict(os.environ, {'IONQ_API_KEY': 'not_this_key'})
def test_service_api_key_passed_directly():
    service = ionq.Service(remote_host='http://example.com', api_key='tomyheart')
    assert service.api_key == 'tomyheart'


@mock.patch.dict(os.environ, {'CIRQ_IONQ_API_KEY': 'tomyheart'})
def test_service_api_key_from_env_var_cirq_ionq():
    service = ionq.Service(remote_host='http://example.com')
    assert service.api_key == 'tomyheart'


@mock.patch.dict(os.environ, {'IONQ_API_KEY': 'tomyheart'})
def test_service_api_key_from_env_var_ionq():
    service = ionq.Service(remote_host='http://example.com')
    assert service.api_key == 'tomyheart'


@mock.patch.dict(os.environ, {}, clear=True)
def test_service_api_key_not_found_raises_error():
    with pytest.raises(EnvironmentError):
        _ = ionq.Service(remote_host='http://example.com')


@mock.patch.dict(os.environ, {'CIRQ_IONQ_API_KEY': 'tomyheart', 'IONQ_API_KEY': 'not_this_key'})
def test_service_api_key_from_env_var_cirq_ionq_precedence():
    service = ionq.Service(remote_host='http://example.com')
    assert service.api_key == 'tomyheart'


@mock.patch.dict(os.environ, {'CIRQ_IONQ_REMOTE_HOST': 'not_this_host'})
def test_service_remote_host_passed_directly():
    service = ionq.Service(remote_host='http://example.com', api_key='tomyheart')
    assert service.remote_host == 'http://example.com'


@mock.patch.dict(os.environ, {'CIRQ_IONQ_REMOTE_HOST': 'http://example.com'})
def test_service_remote_host_from_env_var_cirq_ionq():
    service = ionq.Service(api_key='tomyheart')
    assert service.remote_host == 'http://example.com'


@mock.patch.dict(os.environ, {'IONQ_REMOTE_HOST': 'http://example.com'})
def test_service_remote_host_from_env_var_ionq():
    service = ionq.Service(api_key='tomyheart')
    assert service.remote_host == 'http://example.com'


@mock.patch.dict(os.environ, {}, clear=True)
def test_service_remote_host_default():
    service = ionq.Service(api_key='tomyheart', api_version='v0.4')
    assert service.remote_host == 'https://api.ionq.co/v0.4'


@mock.patch.dict(
    os.environ, {'CIRQ_IONQ_REMOTE_HOST': 'http://example.com', 'IONQ_REMOTE_HOST': 'not_this_host'}
)
def test_service_remote_host_from_env_var_cirq_ionq_precedence():
    service = ionq.Service(api_key='tomyheart')
    assert service.remote_host == 'http://example.com'


def test_service_run_unwraps_single_result_list():
    """`Service.run` should unwrap `[result]` to `result`."""
    # set up a real Service object (we'll monkey-patch its create_job)
    service = ionq.Service(remote_host="http://example.com", api_key="key")

    # simple 1-qubit circuit
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q), cirq.measure(q, key="m"))

    # fabricate a QPUResult and wrap it in a list to mimic an erroneous behavior
    qpu_result = ionq.QPUResult(counts={1: 1}, num_qubits=1, measurement_dict={"m": [0]})
    mock_job = mock.MagicMock()
    mock_job.results.return_value = [qpu_result]  # <- list of length-1

    # monkey-patch create_job so Service.run sees our mock_job
    with mock.patch.object(service, "create_job", return_value=mock_job):
        out = service.run(circuit=circuit, repetitions=1, target="qpu")

    # expected Cirq result after unwrapping and conversion
    expected = qpu_result.to_cirq_result(params=cirq.ParamResolver({}))

    assert out == expected
    mock_job.results.assert_called_once()


@pytest.mark.parametrize("target", ["qpu", "simulator"])
def test_run_batch_preserves_order(target):
    """``Service.run_batch`` must return results in the same order as the
    input ``circuits`` list, regardless of how the IonQ API happens to order
    its per-circuit results.
    """

    # Service with a fully mocked HTTP client.
    service = ionq.Service(remote_host="http://example.com", api_key="key")
    client = mock.MagicMock()
    service._client = client

    # Three trivial 1-qubit circuits, each measuring under a unique key.
    keys = ["a", "b", "c"]
    q = cirq.LineQubit(0)
    circuits = [cirq.Circuit(cirq.measure(q, key=k)) for k in keys]

    client.create_job.return_value = {"id": "job_id", "status": "ready"}

    client.get_job.return_value = {
        "id": "job_id",
        "status": "completed",
        "backend": target,
        "qubits": "1",
        "metadata": {
            "shots": "1",
            "measurements": json.dumps([{"measurement0": f"{k}\u001f0"} for k in keys]),
            "qubit_numbers": json.dumps([1, 1, 1]),
        },
    }

    # Intentionally scramble the order returned by the API: b, a, c.
    client.get_results.return_value = {
        "res_b": {"0": "1"},
        "res_a": {"0": "1"},
        "res_c": {"0": "1"},
    }

    results = service.run_batch(circuits, repetitions=1, target=target)

    # The order of measurement keys in the results should match the input
    # circuit order exactly (a, b, c).
    assert [next(iter(r.measurements)) for r in results] == keys

    # Smoke-test on the mocked client usage.
    client.create_job.assert_called_once()
    client.get_results.assert_called_once()


def test_service_run_seed():
    """Test create_job in another way, for more complete coverage."""
    # Set up a real Service object (we'll monkey-patch its create_job).
    service = ionq.Service(remote_host="http://example.com", api_key="key")

    # Setup the job to return a SimulatorResult.
    mock_job = mock.MagicMock()
    mock_simulator_result = mock.MagicMock(spec=ionq.SimulatorResult)
    mock_job.results.return_value = mock_simulator_result

    # We need to mock create_job on the service because service.run calls self.create_job.
    with mock.patch.object(service, 'create_job', return_value=mock_job):
        circuit = cirq.Circuit(cirq.X(cirq.LineQubit(0)))
        service.run(circuit, repetitions=1, target='simulator', seed=123)

        mock_simulator_result.to_cirq_result.assert_called_once()
        kwargs = mock_simulator_result.to_cirq_result.call_args[1]
        assert kwargs['seed'] == 123


def test_service_run_returns_list_defensive():
    """Cover the case where job.results() returns a list for a single run."""
    service = ionq.Service(remote_host="http://example.com", api_key="key")
    mock_job = mock.MagicMock()
    # Return a list containing one QPUResult
    qpu_result = ionq.QPUResult(counts={1: 1}, num_qubits=1, measurement_dict={"m": [0]})
    mock_job.results.return_value = [qpu_result]

    with mock.patch.object(service, 'create_job', return_value=mock_job):
        circuit = cirq.Circuit(cirq.X(cirq.LineQubit(0)))
        # This should handle the list and extract the first element
        result = service.run(circuit, repetitions=1, target='qpu')

    assert result == qpu_result.to_cirq_result(params=cirq.ParamResolver({}))


def test_service_run_batch_returns_single_defensive():
    """Cover the case where job.results() returns a single item for a batch run."""
    service = ionq.Service(remote_host="http://example.com", api_key="key")
    mock_job = mock.MagicMock()
    # Return a single QPUResult (not a list)
    qpu_result = ionq.QPUResult(counts={1: 1}, num_qubits=1, measurement_dict={"m": [0]})
    mock_job.results.return_value = qpu_result

    with mock.patch.object(service, 'create_batch_job', return_value=mock_job):
        circuit = cirq.Circuit(cirq.X(cirq.LineQubit(0)))
        # run_batch expects a list of circuits
        results = service.run_batch([circuit], repetitions=1, target='qpu')

    # Should wrap the single result in a list
    assert len(results) == 1
    assert results[0] == qpu_result.to_cirq_result(params=cirq.ParamResolver({}))
