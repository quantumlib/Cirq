# Copyright 2021 The Cirq Developers
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
import contextlib
import io
import json
from unittest import mock

import cirq
import pytest
import qubovert as qv
import requests

import cirq_superstaq

API_VERSION = cirq_superstaq.API_VERSION
expected_headers = {
    "Authorization": "to_my_heart",
    "Content-Type": "application/json",
    "X-Client-Version": API_VERSION,
    "X-Client-Name": "cirq-superstaq",
}


def test_cirq_superstaq_exception_str() -> None:
    ex = cirq_superstaq.SuperstaQException("err", status_code=501)
    assert str(ex) == "Status code: 501, Message: 'err'"


def test_cirq_superstaq_not_found_exception_str() -> None:
    ex = cirq_superstaq.SuperstaQNotFoundException("err")
    assert str(ex) == "Status code: 404, Message: 'err'"


def test_superstaq_client_invalid_remote_host() -> None:
    for invalid_url in ("", "url", "http://", "ftp://", "http://"):
        with pytest.raises(AssertionError, match="not a valid url"):
            _ = cirq_superstaq.superstaq_client._SuperstaQClient(
                remote_host=invalid_url, api_key="a"
            )
        with pytest.raises(AssertionError, match=invalid_url):
            _ = cirq_superstaq.superstaq_client._SuperstaQClient(
                remote_host=invalid_url, api_key="a"
            )


def test_superstaq_client_invalid_api_version() -> None:
    with pytest.raises(AssertionError, match="are accepted"):
        _ = cirq_superstaq.superstaq_client._SuperstaQClient(
            remote_host="http://example.com", api_key="a", api_version="v0.0"
        )
    with pytest.raises(AssertionError, match="0.0"):
        _ = cirq_superstaq.superstaq_client._SuperstaQClient(
            remote_host="http://example.com", api_key="a", api_version="v0.0"
        )


def test_superstaq_client_invalid_target() -> None:
    with pytest.raises(AssertionError, match="the store"):
        _ = cirq_superstaq.superstaq_client._SuperstaQClient(
            remote_host="http://example.com", api_key="a", default_target="the store"
        )
    with pytest.raises(AssertionError, match="Target"):
        _ = cirq_superstaq.superstaq_client._SuperstaQClient(
            remote_host="http://example.com", api_key="a", default_target="the store"
        )


def test_superstaq_client_time_travel() -> None:
    with pytest.raises(AssertionError, match="time machine"):
        _ = cirq_superstaq.superstaq_client._SuperstaQClient(
            remote_host="http://example.com", api_key="a", max_retry_seconds=-1
        )


def test_superstaq_client_attributes() -> None:
    client = cirq_superstaq.superstaq_client._SuperstaQClient(
        remote_host="http://example.com",
        api_key="to_my_heart",
        default_target="qpu",
        max_retry_seconds=10,
        verbose=True,
    )
    assert client.url == f"http://example.com/{API_VERSION}"
    assert client.headers == expected_headers
    assert client.default_target == "qpu"
    assert client.max_retry_seconds == 10
    assert client.verbose


@mock.patch("requests.post")
def test_supertstaq_client_create_job(mock_post: mock.MagicMock) -> None:
    mock_post.return_value.status_code.return_value = requests.codes.ok
    mock_post.return_value.json.return_value = {"foo": "bar"}

    client = cirq_superstaq.superstaq_client._SuperstaQClient(
        remote_host="http://example.com", api_key="to_my_heart"
    )
    # program = cirq_superstaq.SerializedProgram(body={"job": "mine"}, metadata={"a": "0,1"})
    response = client.create_job(
        serialized_program=json.dumps({"job": "mine"}), repetitions=200, target="qpu", name="bacon"
    )
    assert response == {"foo": "bar"}

    expected_json = {
        "circuit": {"job": "mine"},
        "backend": "qpu",
        "shots": 200,
        "ibmq_token": None,
        "ibmq_group": None,
        "ibmq_project": None,
        "ibmq_hub": None,
        "ibmq_pulse": True,
    }
    mock_post.assert_called_with(
        f"http://example.com/{API_VERSION}/job",
        json=expected_json,
        headers=expected_headers,
        verify=False,
    )


@mock.patch("requests.post")
def test_superstaq_client_create_job_default_target(mock_post: mock.MagicMock) -> None:
    mock_post.return_value.status_code.return_value = requests.codes.ok
    mock_post.return_value.json.return_value = {"foo"}

    client = cirq_superstaq.superstaq_client._SuperstaQClient(
        remote_host="http://example.com", api_key="to_my_heart", default_target="simulator"
    )
    _ = client.create_job(json.dumps({"job": "mine"}))
    assert mock_post.call_args[1]["json"]["backend"] == "simulator"


@mock.patch("requests.post")
def test_superstaq_client_create_job_target_overrides_default_target(
    mock_post: mock.MagicMock,
) -> None:
    mock_post.return_value.status_code.return_value = requests.codes.ok
    mock_post.return_value.json.return_value = {"foo"}

    client = cirq_superstaq.superstaq_client._SuperstaQClient(
        remote_host="http://example.com", api_key="to_my_heart", default_target="simulator"
    )
    _ = client.create_job(
        serialized_program=json.dumps({"job": "mine"}),
        target="qpu",
        repetitions=1,
    )
    assert mock_post.call_args[1]["json"]["backend"] == "qpu"


def test_superstaq_client_create_job_no_targets() -> None:
    client = cirq_superstaq.superstaq_client._SuperstaQClient(
        remote_host="http://example.com", api_key="to_my_heart"
    )
    with pytest.raises(AssertionError, match="neither were set"):
        _ = client.create_job(serialized_program=json.dumps({"job": "mine"}))


@mock.patch("requests.post")
def test_superstaq_client_create_job_unauthorized(mock_post: mock.MagicMock) -> None:
    mock_post.return_value.ok = False
    mock_post.return_value.status_code = requests.codes.unauthorized

    client = cirq_superstaq.superstaq_client._SuperstaQClient(
        remote_host="http://example.com", api_key="to_my_heart", default_target="simulator"
    )
    with pytest.raises(cirq_superstaq.SuperstaQException, match="Not authorized"):
        _ = client.create_job(serialized_program=json.dumps({"job": "mine"}))


@mock.patch("requests.post")
def test_superstaq_client_create_job_not_found(mock_post: mock.MagicMock) -> None:
    mock_post.return_value.ok = False
    mock_post.return_value.status_code = requests.codes.not_found

    client = cirq_superstaq.superstaq_client._SuperstaQClient(
        remote_host="http://example.com", api_key="to_my_heart", default_target="simulator"
    )
    with pytest.raises(cirq_superstaq.SuperstaQNotFoundException, match="not find"):
        _ = client.create_job(serialized_program=json.dumps({"job": "mine"}))


@mock.patch("requests.post")
def test_superstaq_client_create_job_not_retriable(mock_post: mock.MagicMock) -> None:
    mock_post.return_value.ok = False
    mock_post.return_value.status_code = requests.codes.not_implemented

    client = cirq_superstaq.superstaq_client._SuperstaQClient(
        remote_host="http://example.com", api_key="to_my_heart", default_target="simulator"
    )
    with pytest.raises(cirq_superstaq.SuperstaQException, match="Status: 501"):
        _ = client.create_job(serialized_program=json.dumps({"job": "mine"}))


@mock.patch("requests.post")
def test_superstaq_client_create_job_retry(mock_post: mock.MagicMock) -> None:
    response1 = mock.MagicMock()
    response2 = mock.MagicMock()
    mock_post.side_effect = [response1, response2]
    response1.ok = False
    response1.status_code = requests.codes.service_unavailable
    response2.ok = True
    client = cirq_superstaq.superstaq_client._SuperstaQClient(
        remote_host="http://example.com",
        api_key="to_my_heart",
        default_target="simulator",
        verbose=True,
    )
    test_stdout = io.StringIO()
    with contextlib.redirect_stdout(test_stdout):
        _ = client.create_job(serialized_program=json.dumps({"job": "mine"}))
    assert test_stdout.getvalue().strip() == "Waiting 0.1 seconds before retrying."
    assert mock_post.call_count == 2


@mock.patch("requests.post")
def test_superstaq_client_create_job_retry_request_error(mock_post: mock.MagicMock) -> None:
    response2 = mock.MagicMock()
    mock_post.side_effect = [requests.exceptions.ConnectionError(), response2]
    response2.ok = True
    client = cirq_superstaq.superstaq_client._SuperstaQClient(
        remote_host="http://example.com", api_key="to_my_heart", default_target="simulator"
    )
    _ = client.create_job(serialized_program=json.dumps({"job": "mine"}))
    assert mock_post.call_count == 2


@mock.patch("requests.post")
def test_superstaq_client_create_job_timeout(mock_post: mock.MagicMock) -> None:
    mock_post.return_value.ok = False
    mock_post.return_value.status_code = requests.codes.service_unavailable

    client = cirq_superstaq.superstaq_client._SuperstaQClient(
        remote_host="http://example.com",
        api_key="to_my_heart",
        default_target="simulator",
        max_retry_seconds=0.2,
    )
    with pytest.raises(TimeoutError):
        _ = client.create_job(serialized_program=json.dumps({"job": "mine"}))


@mock.patch("requests.get")
def test_superstaq_client_get_job(mock_get: mock.MagicMock) -> None:
    mock_get.return_value.ok = True
    mock_get.return_value.json.return_value = {"foo": "bar"}
    client = cirq_superstaq.superstaq_client._SuperstaQClient(
        remote_host="http://example.com", api_key="to_my_heart"
    )
    response = client.get_job(job_id="job_id")
    assert response == {"foo": "bar"}

    mock_get.assert_called_with(
        f"http://example.com/{API_VERSION}/job/job_id", headers=expected_headers, verify=False
    )


@mock.patch("requests.get")
def test_superstaq_client_get_balance(mock_get: mock.MagicMock) -> None:
    mock_get.return_value.ok = True
    mock_get.return_value.json.return_value = {"balance": 123.4567}
    client = cirq_superstaq.superstaq_client._SuperstaQClient(
        remote_host="http://example.com", api_key="to_my_heart"
    )
    response = client.get_balance()
    assert response == {"balance": 123.4567}

    mock_get.assert_called_with(
        f"http://example.com/{API_VERSION}/balance", headers=expected_headers, verify=False
    )


@mock.patch("requests.get")
def test_superstaq_client_get_job_unauthorized(mock_get: mock.MagicMock) -> None:
    mock_get.return_value.ok = False
    mock_get.return_value.status_code = requests.codes.unauthorized

    client = cirq_superstaq.superstaq_client._SuperstaQClient(
        remote_host="http://example.com", api_key="to_my_heart", default_target="simulator"
    )
    with pytest.raises(cirq_superstaq.SuperstaQException, match="Not authorized"):
        _ = client.get_job("job_id")


@mock.patch("requests.get")
def test_superstaq_client_get_job_not_found(mock_get: mock.MagicMock) -> None:
    (mock_get.return_value).ok = False
    (mock_get.return_value).status_code = requests.codes.not_found

    client = cirq_superstaq.superstaq_client._SuperstaQClient(
        remote_host="http://example.com", api_key="to_my_heart", default_target="simulator"
    )
    with pytest.raises(cirq_superstaq.SuperstaQNotFoundException, match="not find"):
        _ = client.get_job("job_id")


@mock.patch("requests.get")
def test_superstaq_client_get_job_not_retriable(mock_get: mock.MagicMock) -> None:
    mock_get.return_value.ok = False
    mock_get.return_value.status_code = requests.codes.bad_request

    client = cirq_superstaq.superstaq_client._SuperstaQClient(
        remote_host="http://example.com", api_key="to_my_heart", default_target="simulator"
    )
    with pytest.raises(cirq_superstaq.SuperstaQException, match="Status: 400"):
        _ = client.get_job("job_id")


@mock.patch("requests.get")
def test_superstaq_client_get_job_retry(mock_get: mock.MagicMock) -> None:
    response1 = mock.MagicMock()
    response2 = mock.MagicMock()
    mock_get.side_effect = [response1, response2]
    response1.ok = False
    response1.status_code = requests.codes.service_unavailable
    response2.ok = True
    client = cirq_superstaq.superstaq_client._SuperstaQClient(
        remote_host="http://example.com", api_key="to_my_heart", default_target="simulator"
    )
    _ = client.get_job("job_id")
    assert mock_get.call_count == 2


@mock.patch("requests.post")
def test_superstaq_client_aqt_compile(mock_post: mock.MagicMock) -> None:
    client = cirq_superstaq.superstaq_client._SuperstaQClient(
        remote_host="http://example.com", api_key="to_my_heart", default_target="simulator"
    )
    circuit = cirq.Circuit(cirq.H(cirq.LineQubit(5)))
    client.aqt_compile(cirq_superstaq.serialization.serialize_circuits([circuit, circuit]))

    mock_post.assert_called_once()
    assert mock_post.call_args[0][0] == f"http://example.com/{API_VERSION}/aqt_compile"


@mock.patch("requests.post")
def test_superstaq_client_ibmq_compile(mock_post: mock.MagicMock) -> None:
    client = cirq_superstaq.superstaq_client._SuperstaQClient(
        remote_host="http://example.com", api_key="to_my_heart", default_target="simulator"
    )
    circuit = cirq.Circuit()
    client.ibmq_compile(cirq_superstaq.serialization.serialize_circuits([circuit]))

    mock_post.assert_called_once()
    assert mock_post.call_args[0][0] == f"http://example.com/{API_VERSION}/ibmq_compile"


@mock.patch("requests.post")
def test_superstaq_client_submit_qubo(mock_post: mock.MagicMock) -> None:
    client = cirq_superstaq.superstaq_client._SuperstaQClient(
        remote_host="http://example.com", api_key="to_my_heart", default_target="simulator"
    )

    example_qubo = qv.QUBO({(0,): 1.0, (1,): 1.0, (0, 1): -2.0})
    target = "example_target"
    repetitions = 10
    client.submit_qubo(example_qubo, target, repetitions=repetitions)

    expected_json = {
        "qubo": [
            {"keys": ["0"], "value": 1.0},
            {"keys": ["1"], "value": 1.0},
            {"keys": ["0", "1"], "value": -2.0},
        ],
        "backend": target,
        "shots": repetitions,
    }

    mock_post.assert_called_with(
        f"http://example.com/{API_VERSION}/qubo",
        headers=expected_headers,
        json=expected_json,
        verify=False,
    )


@mock.patch("requests.post")
def test_superstaq_client_find_min_vol_portfolio(mock_post: mock.MagicMock) -> None:
    client = cirq_superstaq.superstaq_client._SuperstaQClient(
        remote_host="http://example.com", api_key="to_my_heart", default_target="simulator"
    )
    client.find_min_vol_portfolio(
        {"stock_symbols": ["AAPL", "GOOG", "IEF", "MMM"], "desired_return": 8}
    )

    expected_json = {"stock_symbols": ["AAPL", "GOOG", "IEF", "MMM"], "desired_return": 8}
    mock_post.assert_called_with(
        f"http://example.com/{API_VERSION}/minvol",
        headers=expected_headers,
        json=expected_json,
        verify=False,
    )


@mock.patch("requests.post")
def test_superstaq_client_find_max_pseudo_sharpe_ratio(mock_post: mock.MagicMock) -> None:
    client = cirq_superstaq.superstaq_client._SuperstaQClient(
        remote_host="http://example.com", api_key="to_my_heart", default_target="simulator"
    )
    client.find_max_pseudo_sharpe_ratio({"stock_symbols": ["AAPL", "GOOG", "IEF", "MMM"], "k": 0.5})

    expected_json = {"stock_symbols": ["AAPL", "GOOG", "IEF", "MMM"], "k": 0.5}
    mock_post.assert_called_with(
        f"http://example.com/{API_VERSION}/maxsharpe",
        headers=expected_headers,
        json=expected_json,
        verify=False,
    )


@mock.patch("requests.post")
def test_superstaq_client_tsp(mock_post: mock.MagicMock) -> None:
    client = cirq_superstaq.superstaq_client._SuperstaQClient(
        remote_host="http://example.com", api_key="to_my_heart", default_target="simulator"
    )
    client.tsp({"locs": ["Chicago", "St Louis", "St Paul"]})

    expected_json = {"locs": ["Chicago", "St Louis", "St Paul"]}
    mock_post.assert_called_with(
        f"http://example.com/{API_VERSION}/tsp",
        headers=expected_headers,
        json=expected_json,
        verify=False,
    )


@mock.patch("requests.post")
def test_superstaq_client_warehouse(mock_post: mock.MagicMock) -> None:
    client = cirq_superstaq.superstaq_client._SuperstaQClient(
        remote_host="http://example.com", api_key="to_my_heart", default_target="simulator"
    )
    client.warehouse(
        {
            "k": 1,
            "possible_warehouses": ["Chicago", "San Francisco"],
            "customers": ["Rockford", "Aurora"],
        }
    )

    expected_json = {
        "k": 1,
        "possible_warehouses": ["Chicago", "San Francisco"],
        "customers": ["Rockford", "Aurora"],
    }
    mock_post.assert_called_with(
        f"http://example.com/{API_VERSION}/warehouse",
        headers=expected_headers,
        json=expected_json,
    )


@mock.patch("requests.post")
def test_superstaq_client_aqt_upload_configs(mock_post: mock.MagicMock) -> None:
    client = cirq_superstaq.superstaq_client._SuperstaQClient(
        remote_host="http://example.com", api_key="to_my_heart", default_target="simulator"
    )

    client.aqt_upload_configs({"pulses": "Hello", "variables": "World"})

    expected_json = {"pulses": "Hello", "variables": "World"}
    mock_post.assert_called_with(
        f"http://example.com/{API_VERSION}/aqt_configs",
        headers=expected_headers,
        json=expected_json,
        verify=False,
    )
