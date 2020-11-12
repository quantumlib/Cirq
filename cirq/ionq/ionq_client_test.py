# Copyright 2020 The Cirq Developers
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

import requests
import pytest
from unittest import mock

import cirq.ionq as ionq


def test_ionq_exception_str():
    ex = ionq.IonQException('err', status_code=501)
    assert (str(ex) == 'Status code: 501, Message: \'err\'')


def test_ionq_not_found_exception_str():
    ex = ionq.IonQNotFoundException('err')
    assert (str(ex) == 'Status code: 404, Message: \'err\'')


def test_ionq_client_invalid_remote_host():
    for invalid_url in ('', 'myurl', 'http://', 'ftp://', 'http://'):
        with pytest.raises(AssertionError, match='not a valid url'):
            _ = ionq.ionq_client._IonQClient(remote_host=invalid_url,
                                             api_key='a')
        with pytest.raises(AssertionError, match=invalid_url):
            _ = ionq.ionq_client._IonQClient(remote_host=invalid_url,
                                             api_key='a')


def test_ionq_client_invalid_api_version():
    with pytest.raises(AssertionError, match='is accepted'):
        _ = ionq.ionq_client._IonQClient(remote_host='http://example.com',
                                         api_key='a',
                                         api_version='v0.0')
    with pytest.raises(AssertionError, match='0.0'):
        _ = ionq.ionq_client._IonQClient(remote_host='http://example.com',
                                         api_key='a',
                                         api_version='v0.0')


def test_ionq_client_invalid_target():
    with pytest.raises(AssertionError, match='the store'):
        _ = ionq.ionq_client._IonQClient(remote_host='http://example.com',
                                         api_key='a',
                                         default_target='the store')
    with pytest.raises(AssertionError, match='Target'):
        _ = ionq.ionq_client._IonQClient(remote_host='http://example.com',
                                         api_key='a',
                                         default_target='the store')


def test_ionq_client_time_travel():
    with pytest.raises(AssertionError, match='time machine'):
        _ = ionq.ionq_client._IonQClient(remote_host='http://example.com',
                                         api_key='a',
                                         max_retry_seconds=-1)


def test_ionq_client_attributes():
    client = ionq.ionq_client._IonQClient(remote_host='http://example.com',
                                          api_key='tomyheart',
                                          default_target='qpu',
                                          max_retry_seconds=10,
                                          verbose=True)
    assert client.url == 'http://example.com/v0.1'
    assert client.headers == {
        'Authorization': 'apiKey tomyheart',
        'Content-Type': 'application/json'
    }
    assert client.default_target == 'qpu'
    assert client.max_retry_seconds == 10
    assert client.verbose == True


@mock.patch('requests.post')
def test_ionq_client_create_job(mock_post):
    mock_post.return_value.status_code.return_value = requests.codes.ok
    mock_post.return_value.json.return_value = {'foo': 'bar'}

    client = ionq.ionq_client._IonQClient(remote_host='http://example.com',
                                          api_key='tomyheart')
    job_dict = {'job': 'mine'}
    response = client.create_job(job_dict=job_dict,
                                 repetitions=200,
                                 target='qpu',
                                 name='bacon')
    assert response == {'foo': 'bar'}

    expected_json = {
        'target': 'qpu',
        'body': {
            'job': 'mine'
        },
        'lang': 'json',
        'name': 'bacon',
        'metadata': {
            'shots': '200'
        }
    }
    expected_headers = {
        'Authorization': 'apiKey tomyheart',
        'Content-Type': 'application/json'
    }
    mock_post.assert_called_with('http://example.com/v0.1/jobs',
                                 json=expected_json,
                                 headers=expected_headers)


@mock.patch('requests.post')
def test_ionq_client_create_job_default_target(mock_post):
    mock_post.return_value.status_code.return_value = requests.codes.ok
    mock_post.return_value.json.return_value = {'foo'}

    client = ionq.ionq_client._IonQClient(remote_host='http://example.com',
                                          api_key='tomyheart',
                                          default_target='simulator')
    _ = client.create_job(job_dict={'job': 'mine'})
    assert mock_post.call_args.kwargs['json']['target'] == 'simulator'


@mock.patch('requests.post')
def test_ionq_client_create_job_target_overrides_default_target(mock_post):
    mock_post.return_value.status_code.return_value = requests.codes.ok
    mock_post.return_value.json.return_value = {'foo'}

    client = ionq.ionq_client._IonQClient(remote_host='http://example.com',
                                          api_key='tomyheart',
                                          default_target='simulator')
    _ = client.create_job(job_dict={'job': 'mine'}, target='qpu', repetitions=1)
    assert mock_post.call_args.kwargs['json']['target'] == 'qpu'


def test_ionq_client_create_job_no_targets():
    client = ionq.ionq_client._IonQClient(remote_host='http://example.com',
                                          api_key='tomyheart')
    with pytest.raises(AssertionError, match='neither were set'):
        _ = client.create_job(job_dict={'job': 'mine'})


def test_ionq_client_create_job_qpu_but_no_repetitions():
    client = ionq.ionq_client._IonQClient(remote_host='http://example.com',
                                          api_key='tomyheart')
    with pytest.raises(AssertionError, match='qpu'):
        _ = client.create_job(job_dict={'job': 'mine'}, target='qpu')


def test_ionq_client_create_job_simulator_but_repetitions():
    client = ionq.ionq_client._IonQClient(remote_host='http://example.com',
                                          api_key='tomyheart')
    with pytest.raises(AssertionError, match='simulator'):
        _ = client.create_job(job_dict={'job': 'mine'},
                              target='simulator',
                              repetitions=10)


@mock.patch('requests.post')
def test_ionq_client_create_job_unauthorized(mock_post):
    mock_post.return_value.ok = False
    mock_post.return_value.status_code = requests.codes.unauthorized

    client = ionq.ionq_client._IonQClient(remote_host='http://example.com',
                                          api_key='tomyheart',
                                          default_target='simulator')
    with pytest.raises(ionq.IonQException, match='Not authorized'):
        _ = client.create_job(job_dict={'job': 'mine'})


@mock.patch('requests.post')
def test_ionq_client_create_job_not_found(mock_post):
    (mock_post.return_value).ok = False
    (mock_post.return_value).status_code = requests.codes.not_found

    client = ionq.ionq_client._IonQClient(remote_host='http://example.com',
                                          api_key='tomyheart',
                                          default_target='simulator')
    with pytest.raises(ionq.IonQNotFoundException, match='not find'):
        _ = client.create_job(job_dict={'job': 'mine'})


@mock.patch('requests.post')
def test_ionq_client_create_job_not_retriable(mock_post):
    mock_post.return_value.ok = False
    mock_post.return_value.status_code = requests.codes.not_implemented

    client = ionq.ionq_client._IonQClient(remote_host='http://example.com',
                                          api_key='tomyheart',
                                          default_target='simulator')
    with pytest.raises(ionq.IonQException, match='Status: 501'):
        _ = client.create_job(job_dict={'job': 'mine'})


@mock.patch('requests.post')
def test_ionq_client_create_job_retry(mock_post):
    response1 = mock.MagicMock()
    response2 = mock.MagicMock()
    mock_post.side_effect = [response1, response2]
    response1.ok = False
    response1.status_code = requests.codes.service_unavailable
    response2.ok = True
    client = ionq.ionq_client._IonQClient(remote_host='http://example.com',
                                          api_key='tomyheart',
                                          default_target='simulator')
    _ = client.create_job(job_dict={'job': 'mine'})
    assert mock_post.call_count == 2


@mock.patch('requests.post')
def test_ionq_client_create_job_retry(mock_post):
    response1 = mock.MagicMock()
    response2 = mock.MagicMock()
    mock_post.side_effect = [
        requests.RequestException(response=mock.MagicMock()), response2
    ]
    response1.ok = False
    response1.status_code = requests.codes.service_unavailable
    response2.ok = True
    client = ionq.ionq_client._IonQClient(remote_host='http://example.com',
                                          api_key='tomyheart',
                                          default_target='simulator')
    _ = client.create_job(job_dict={'job': 'mine'})
    assert mock_post.call_count == 2


@mock.patch('requests.get')
def test_ionq_client_get_job(mock_get):
    mock_get.return_value.ok = True
    mock_get.return_value.json.return_value = {'foo': 'bar'}
    client = ionq.ionq_client._IonQClient(remote_host='http://example.com',
                                          api_key='tomyheart')
    response = client.get_job(job_id='jobid')
    assert response == {'foo': 'bar'}

    expected_headers = {
        'Authorization': 'apiKey tomyheart',
        'Content-Type': 'application/json'
    }
    mock_get.assert_called_with('http://example.com/v0.1/jobs/jobid',
                                headers=expected_headers)


@mock.patch('requests.get')
def test_ionq_client_get_job_unauthorized(mock_get):
    mock_get.return_value.ok = False
    mock_get.return_value.status_code = requests.codes.unauthorized

    client = ionq.ionq_client._IonQClient(remote_host='http://example.com',
                                          api_key='tomyheart',
                                          default_target='simulator')
    with pytest.raises(ionq.IonQException, match='Not authorized'):
        _ = client.get_job('jobid')


@mock.patch('requests.get')
def test_ionq_client_get_job_not_found(mock_get):
    (mock_get.return_value).ok = False
    (mock_get.return_value).status_code = requests.codes.not_found

    client = ionq.ionq_client._IonQClient(remote_host='http://example.com',
                                          api_key='tomyheart',
                                          default_target='simulator')
    with pytest.raises(ionq.IonQNotFoundException, match='not find'):
        _ = client.get_job('jobid')


@mock.patch('requests.get')
def test_ionq_client_get_job_not_retriable(mock_get):
    mock_get.return_value.ok = False
    mock_get.return_value.status_code = requests.codes.not_implemented

    client = ionq.ionq_client._IonQClient(remote_host='http://example.com',
                                          api_key='tomyheart',
                                          default_target='simulator')
    with pytest.raises(ionq.IonQException, match='Status: 501'):
        _ = client.get_job('jobid')


@mock.patch('requests.get')
def test_ionq_client_get_job_retry(mock_get):
    response1 = mock.MagicMock()
    response2 = mock.MagicMock()
    mock_get.side_effect = [response1, response2]
    response1.ok = False
    response1.status_code = requests.codes.service_unavailable
    response2.ok = True
    client = ionq.ionq_client._IonQClient(remote_host='http://example.com',
                                          api_key='tomyheart',
                                          default_target='simulator')
    _ = client.get_job('jobid')
    assert mock_get.call_count == 2
