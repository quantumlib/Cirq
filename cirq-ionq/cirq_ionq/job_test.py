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
    job_dict = {
        'id': 'my_id',
        'status': 'completed',
        'qubits': '2',
        'target': 'qpu',
        'metadata': {'shots': 1000, 'measurement0': f'a{chr(31)}0,1'},
        'data': {'histogram': {'0': '0.6', '2': '0.4'}},
    }
    job = ionq.Job(None, job_dict)
    results = job.results()
    expected = ionq.QPUResult({0: 600, 1: 400}, 2, {'a': [0, 1]})
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
    job_dict = {
        'id': 'my_id',
        'status': 'completed',
        'qubits': '2',
        'target': 'qpu',
        'metadata': {'shots': 1000},
        'data': {'histogram': {'0': '0.6', '1': '0.4'}},
    }
    job = ionq.Job(None, job_dict)
    results = job.results()
    assert results == ionq.QPUResult({0: 600, 2: 400}, 2, measurement_dict={})


def test_job_results_qpu_target_endianness():
    job_dict = {
        'id': 'my_id',
        'status': 'completed',
        'qubits': '2',
        'target': 'qpu.target',
        'metadata': {'shots': 1000},
        'data': {'histogram': {'0': '0.6', '1': '0.4'}},
    }
    job = ionq.Job(None, job_dict)
    results = job.results()
    assert results == ionq.QPUResult({0: 600, 2: 400}, 2, measurement_dict={})


@mock.patch('time.sleep', return_value=None)
def test_job_results_poll(mock_sleep):
    ready_job = {
        'id': 'my_id',
        'status': 'ready',
    }
    completed_job = {
        'id': 'my_id',
        'status': 'completed',
        'qubits': '1',
        'target': 'qpu',
        'metadata': {'shots': 1000},
        'data': {'histogram': {'0': '0.6', '1': '0.4'}},
    }
    mock_client = mock.MagicMock()
    mock_client.get_job.side_effect = [ready_job, completed_job]
    job = ionq.Job(mock_client, ready_job)
    results = job.results(polling_seconds=0)
    assert results == ionq.QPUResult({0: 600, 1: 400}, 1, measurement_dict={})
    mock_sleep.assert_called_once()


@mock.patch('time.sleep', return_value=None)
def test_job_results_poll_timeout(mock_sleep):
    ready_job = {
        'id': 'my_id',
        'status': 'ready',
    }
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
    job_dict = {
        'id': 'my_id',
        'status': 'completed',
        'qubits': '1',
        'target': 'simulator',
        'data': {'histogram': {'0': '0.6', '1': '0.4'}},
        'metadata': {'shots': '100'},
    }
    job = ionq.Job(None, job_dict)
    results = job.results()
    assert results == ionq.SimulatorResult({0: 0.6, 1: 0.4}, 1, {}, 100)


def test_job_results_simulator_endianness():
    job_dict = {
        'id': 'my_id',
        'status': 'completed',
        'qubits': '2',
        'target': 'simulator',
        'data': {'histogram': {'0': '0.6', '1': '0.4'}},
        'metadata': {'shots': '100'},
    }
    job = ionq.Job(None, job_dict)
    results = job.results()
    assert results == ionq.SimulatorResult({0: 0.6, 2: 0.4}, 2, {}, 100)


def test_job_cancel():
    ready_job = {
        'id': 'my_id',
        'status': 'ready',
    }
    canceled_job = {'id': 'my_id', 'status': 'canceled'}
    mock_client = mock.MagicMock()
    mock_client.cancel_job.return_value = canceled_job
    job = ionq.Job(mock_client, ready_job)
    job.cancel()
    mock_client.cancel_job.assert_called_with(job_id='my_id')
    assert job.status() == 'canceled'


def test_job_delete():
    ready_job = {
        'id': 'my_id',
        'status': 'ready',
    }
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
