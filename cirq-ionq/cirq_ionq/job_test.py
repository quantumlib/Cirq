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

from unittest import mock

import json
import warnings
import pytest

import cirq_ionq as ionq


def test_job_fields():
    job_dict = {
        'id': 'my_id',
        'target': 'qpu',
        'name': 'bacon',
        'qubits': '5',
        'status': 'completed',
        'metadata': {'shots': 1000, 'measurement0': f'a{chr(31)}0,1'},
    }
    job = ionq.Job(None, job_dict)
    assert job.job_id() == 'my_id'
    assert job.target() == 'qpu'
    assert job.name() == 'bacon'
    assert job.num_qubits() == 5
    assert job.repetitions() == 1000
    assert job.measurement_dict() == {'a': [0, 1]}


def test_job_fields_multiple_circuits():
    job_dict = {
        'id': 'my_id',
        'target': 'qpu',
        'name': 'bacon',
        'qubits': '5',
        'status': 'completed',
        'metadata': {
            'shots': 1000,
            'measurements': json.dumps([{'measurement0': f'a{chr(31)}0,1'}]),
        },
    }
    job = ionq.Job(None, job_dict)
    assert job.job_id() == 'my_id'
    assert job.target() == 'qpu'
    assert job.name() == 'bacon'
    assert job.num_qubits() == 5
    assert job.repetitions() == 1000
    assert job.measurement_dict() == {'a': [0, 1]}


def test_job_status_refresh():
    for status in ionq.Job.NON_TERMINAL_STATES:
        mock_client = mock.MagicMock()
        mock_client.get_job.return_value = {'id': 'my_id', 'status': 'completed'}
        job = ionq.Job(mock_client, {'id': 'my_id', 'status': status})
        assert job.status() == 'completed'
        mock_client.get_job.assert_called_with('my_id')
    for status in ionq.Job.TERMINAL_STATES:
        mock_client = mock.MagicMock()
        job = ionq.Job(mock_client, {'id': 'my_id', 'status': status})
        assert job.status() == status
        mock_client.get_job.assert_not_called()


def test_job_str():
    job = ionq.Job(None, {'id': 'my_id'})
    assert str(job) == 'cirq_ionq.Job(job_id=my_id)'


def test_job_results_qpu():
    mock_client = mock.MagicMock()
    mock_client.get_results.return_value = {'0': '0.6', '2': '0.4'}
    job_dict = {
        'id': 'my_id',
        'status': 'completed',
        'qubits': '2',
        'target': 'qpu',
        'metadata': {'shots': 1000, 'measurement0': f'a{chr(31)}0,1'},
        'warning': {'messages': ['foo', 'bar']},
    }
    job = ionq.Job(mock_client, job_dict)
    with warnings.catch_warnings(record=True) as w:
        results = job.results()
        assert len(w) == 2
        assert "foo" in str(w[0].message)
        assert "bar" in str(w[1].message)
    expected = ionq.QPUResult({0: 600, 1: 400}, 2, {'a': [0, 1]})
    assert results == expected


def test_batch_job_results_qpu():
    mock_client = mock.MagicMock()
    mock_client.get_results.return_value = {
        '0190070f-9691-7000-a1f6-306623179a83': {'0': '0.6', '2': '0.4'},
        '0190070f-991c-7000-8700-c4b56b30715d': {'1': 1.0},
    }
    job_dict = {
        'id': 'my_id',
        'status': 'completed',
        'qubits': '2',
        'target': 'qpu',
        'metadata': {
            'shots': 1000,
            'measurements': json.dumps(
                [{'measurement0': f'a{chr(31)}0,1'}, {'measurement0': f'a{chr(31)}0'}]
            ),
            'qubit_numbers': json.dumps([2, 1]),
        },
        'warning': {'messages': ['foo', 'bar']},
    }
    job = ionq.Job(mock_client, job_dict)
    with warnings.catch_warnings(record=True) as w:
        results = job.results()
        assert len(w) == 2
        assert "foo" in str(w[0].message)
        assert "bar" in str(w[1].message)
    expected_0 = ionq.QPUResult({0: 600, 1: 400}, 2, {'a': [0, 1]})
    expected_1 = ionq.QPUResult({1: 1000}, 1, {'a': [0]})
    assert results[0] == expected_0
    assert results[1] == expected_1


def test_job_results_rounding_qpu():
    mock_client = mock.MagicMock()
    mock_client.get_results.return_value = {'0': '0.0006', '2': '0.9994'}
    job_dict = {
        'id': 'my_id',
        'status': 'completed',
        'qubits': '2',
        'target': 'qpu',
        'metadata': {'shots': 5000, 'measurement0': f'a{chr(31)}0,1'},
    }
    # 5000*0.0006 ~ 2.9999 but should be interpreted as 3
    job = ionq.Job(mock_client, job_dict)
    expected = ionq.QPUResult({0: 3, 1: 4997}, 2, {'a': [0, 1]})
    results = job.results()
    assert results == expected


def test_job_results_failed():
    job_dict = {'id': 'my_id', 'status': 'failed', 'failure': {'error': 'too many qubits'}}
    job = ionq.Job(None, job_dict)
    with pytest.raises(RuntimeError, match='too many qubits'):
        _ = job.results()
    assert job.status() == 'failed'


def test_job_results_failed_no_error_message():
    job_dict = {'id': 'my_id', 'status': 'failed', 'failure': {}}
    job = ionq.Job(None, job_dict)
    with pytest.raises(RuntimeError, match='failed'):
        _ = job.results()
    assert job.status() == 'failed'


def test_job_results_qpu_endianness():
    mock_client = mock.MagicMock()
    mock_client.get_results.return_value = {'0': '0.6', '1': '0.4'}
    job_dict = {
        'id': 'my_id',
        'status': 'completed',
        'qubits': '2',
        'target': 'qpu',
        'metadata': {'shots': 1000},
    }
    job = ionq.Job(mock_client, job_dict)
    results = job.results()
    assert results == ionq.QPUResult({0: 600, 2: 400}, 2, measurement_dict={})


def test_batch_job_results_qpu_endianness():
    mock_client = mock.MagicMock()
    mock_client.get_results.return_value = {
        '0190070f-9691-7000-a1f6-306623179a83': {'0': '0.6', '1': '0.4'}
    }
    job_dict = {
        'id': 'my_id',
        'status': 'completed',
        'qubits': '2',
        'target': 'qpu',
        'metadata': {
            'shots': 1000,
            'measurements': json.dumps([{'measurement0': f'a{chr(31)}0,1'}]),
            'qubit_numbers': json.dumps([2]),
        },
    }
    job = ionq.Job(mock_client, job_dict)
    results = job.results()
    assert results[0] == ionq.QPUResult({0: 600, 2: 400}, 2, measurement_dict={'a': [0, 1]})


def test_job_results_qpu_target_endianness():
    mock_client = mock.MagicMock()
    mock_client.get_results.return_value = {'0': '0.6', '1': '0.4'}
    job_dict = {
        'id': 'my_id',
        'status': 'completed',
        'qubits': '2',
        'target': 'qpu.target',
        'metadata': {'shots': 1000},
        'data': {'histogram': {'0': '0.6', '1': '0.4'}},
    }
    job = ionq.Job(mock_client, job_dict)
    results = job.results()
    assert results == ionq.QPUResult({0: 600, 2: 400}, 2, measurement_dict={})


def test_batch_job_results_qpu_target_endianness():
    mock_client = mock.MagicMock()
    mock_client.get_results.return_value = {
        '0190070f-9691-7000-a1f6-306623179a83': {'0': '0.6', '1': '0.4'}
    }
    job_dict = {
        'id': 'my_id',
        'status': 'completed',
        'qubits': '2',
        'target': 'qpu.target',
        'metadata': {
            'shots': 1000,
            'measurements': json.dumps([{'measurement0': f'a{chr(31)}0,1'}]),
            'qubit_numbers': json.dumps([2]),
        },
        'data': {'histogram': {'0': '0.6', '1': '0.4'}},
    }
    job = ionq.Job(mock_client, job_dict)
    results = job.results()
    assert results[0] == ionq.QPUResult({0: 600, 2: 400}, 2, measurement_dict={'a': [0, 1]})


@mock.patch('time.sleep', return_value=None)
def test_job_results_poll(mock_sleep):
    ready_job = {'id': 'my_id', 'status': 'ready'}
    completed_job = {
        'id': 'my_id',
        'status': 'completed',
        'qubits': '1',
        'target': 'qpu',
        'metadata': {'shots': 1000},
    }
    mock_client = mock.MagicMock()
    mock_client.get_job.side_effect = [ready_job, completed_job]
    mock_client.get_results.return_value = {'0': '0.6', '1': '0.4'}
    job = ionq.Job(mock_client, ready_job)
    results = job.results(polling_seconds=0)
    assert results == ionq.QPUResult({0: 600, 1: 400}, 1, measurement_dict={})
    mock_sleep.assert_called_once()


@mock.patch('time.sleep', return_value=None)
def test_job_results_poll_timeout(mock_sleep):
    ready_job = {'id': 'my_id', 'status': 'ready'}
    mock_client = mock.MagicMock()
    mock_client.get_job.return_value = ready_job
    job = ionq.Job(mock_client, ready_job)
    with pytest.raises(TimeoutError, match='seconds'):
        _ = job.results(timeout_seconds=1, polling_seconds=0.1)
    assert mock_sleep.call_count == 11


@mock.patch('time.sleep', return_value=None)
def test_job_results_poll_timeout_with_error_message(mock_sleep):
    ready_job = {'id': 'my_id', 'status': 'failure', 'failure': {'error': 'too many qubits'}}
    mock_client = mock.MagicMock()
    mock_client.get_job.return_value = ready_job
    job = ionq.Job(mock_client, ready_job)
    with pytest.raises(RuntimeError, match='too many qubits'):
        _ = job.results(timeout_seconds=1, polling_seconds=0.1)
    assert mock_sleep.call_count == 11


def test_job_results_simulator():
    mock_client = mock.MagicMock()
    mock_client.get_results.return_value = {'0': '0.6', '1': '0.4'}
    job_dict = {
        'id': 'my_id',
        'status': 'completed',
        'qubits': '1',
        'target': 'simulator',
        'metadata': {'shots': '100'},
    }
    job = ionq.Job(mock_client, job_dict)
    results = job.results()
    assert results == ionq.SimulatorResult({0: 0.6, 1: 0.4}, 1, {}, 100)


def test_batch_job_results_simulator():
    mock_client = mock.MagicMock()
    mock_client.get_results.return_value = {
        '0190070f-9691-7000-a1f6-306623179a83': {'0': '0.6', '2': '0.4'},
        '0190070f-991c-7000-8700-c4b56b30715d': {'1': 1.0},
    }
    job_dict = {
        'id': 'my_id',
        'status': 'completed',
        'qubits': '2',
        'target': 'simulator',
        'metadata': {
            'shots': 1000,
            'measurements': json.dumps(
                [{'measurement0': f'a{chr(31)}0,1'}, {'measurement0': f'a{chr(31)}0'}]
            ),
            'qubit_numbers': json.dumps([2, 1]),
        },
    }
    job = ionq.Job(mock_client, job_dict)
    results = job.results()
    expected_0 = ionq.SimulatorResult({0: 0.6, 1: 0.4}, 2, {'a': [0, 1]}, repetitions=1000)
    expected_1 = ionq.SimulatorResult({1: 1}, 1, {'a': [0]}, repetitions=1000)
    assert results[0] == expected_0
    assert results[1] == expected_1


def test_job_results_simulator_endianness():
    mock_client = mock.MagicMock()
    mock_client.get_results.return_value = {'0': '0.6', '1': '0.4'}
    job_dict = {
        'id': 'my_id',
        'status': 'completed',
        'qubits': '2',
        'target': 'simulator',
        'metadata': {'shots': '100'},
    }
    job = ionq.Job(mock_client, job_dict)
    results = job.results()
    assert results == ionq.SimulatorResult({0: 0.6, 2: 0.4}, 2, {}, 100)


def test_batch_job_results_simulator_endianness():
    mock_client = mock.MagicMock()
    mock_client.get_results.return_value = {
        '0190070f-9691-7000-a1f6-306623179a83': {'0': '0.6', '1': '0.4'}
    }
    job_dict = {
        'id': 'my_id',
        'status': 'completed',
        'qubits': '2',
        'target': 'simulator',
        'metadata': {
            'shots': 1000,
            'measurements': json.dumps([{'measurement0': f'a{chr(31)}0,1'}]),
            'qubit_numbers': json.dumps([2]),
        },
    }
    job = ionq.Job(mock_client, job_dict)
    results = job.results()
    assert results[0] == ionq.SimulatorResult({0: 0.6, 2: 0.4}, 2, {'a': [0, 1]}, 1000)


def test_job_sharpen_results():
    mock_client = mock.MagicMock()
    mock_client.get_results.return_value = {'0': '60', '1': '40'}
    job_dict = {
        'id': 'my_id',
        'status': 'completed',
        'qubits': '1',
        'target': 'simulator',
        'metadata': {'shots': '100'},
    }
    job = ionq.Job(mock_client, job_dict)
    results = job.results(sharpen=False)
    assert results == ionq.SimulatorResult({0: 60, 1: 40}, 1, {}, 100)


def test_job_cancel():
    ready_job = {'id': 'my_id', 'status': 'ready'}
    canceled_job = {'id': 'my_id', 'status': 'canceled'}
    mock_client = mock.MagicMock()
    mock_client.cancel_job.return_value = canceled_job
    job = ionq.Job(mock_client, ready_job)
    job.cancel()
    mock_client.cancel_job.assert_called_with(job_id='my_id')
    assert job.status() == 'canceled'


def test_job_delete():
    ready_job = {'id': 'my_id', 'status': 'ready'}
    deleted_job = {'id': 'my_id', 'status': 'deleted'}
    mock_client = mock.MagicMock()
    mock_client.delete_job.return_value = deleted_job
    job = ionq.Job(mock_client, ready_job)
    job.delete()
    mock_client.delete_job.assert_called_with(job_id='my_id')
    assert job.status() == 'deleted'


def test_job_fields_unsuccessful():
    job_dict = {
        'id': 'my_id',
        'target': 'qpu',
        'name': 'bacon',
        'qubits': '5',
        'status': 'deleted',
        'metadata': {'shots': 1000},
    }
    job = ionq.Job(None, job_dict)
    with pytest.raises(ionq.IonQUnsuccessfulJobException, match='deleted'):
        _ = job.target()
    with pytest.raises(ionq.IonQUnsuccessfulJobException, match='deleted'):
        _ = job.name()
    with pytest.raises(ionq.IonQUnsuccessfulJobException, match='deleted'):
        _ = job.num_qubits()
    with pytest.raises(ionq.IonQUnsuccessfulJobException, match='deleted'):
        _ = job.repetitions()


def test_job_fields_cannot_get_status():
    job_dict = {
        'id': 'my_id',
        'target': 'qpu',
        'name': 'bacon',
        'qubits': '5',
        'status': 'running',
        'metadata': {'shots': 1000},
    }
    mock_client = mock.MagicMock()
    mock_client.get_job.side_effect = ionq.IonQException('bad')
    job = ionq.Job(mock_client, job_dict)
    with pytest.raises(ionq.IonQException, match='bad'):
        _ = job.target()
    with pytest.raises(ionq.IonQException, match='bad'):
        _ = job.name()
    with pytest.raises(ionq.IonQException, match='bad'):
        _ = job.num_qubits()
    with pytest.raises(ionq.IonQException, match='bad'):
        _ = job.repetitions()


def test_job_fields_update_status():
    job_dict = {
        'id': 'my_id',
        'target': 'qpu',
        'name': 'bacon',
        'qubits': '5',
        'status': 'running',
        'metadata': {'shots': 1000},
    }
    mock_client = mock.MagicMock()
    mock_client.get_job.return_value = job_dict
    job = ionq.Job(mock_client, job_dict)
    assert job.job_id() == 'my_id'
    assert job.target() == 'qpu'
    assert job.name() == 'bacon'
    assert job.num_qubits() == 5
    assert job.repetitions() == 1000
