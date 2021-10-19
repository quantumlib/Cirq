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

import collections
import os
from unittest import mock

import applications_superstaq
import cirq
import numpy as np
import pytest
import qubovert as qv
import sympy

import cirq_superstaq


def test_counts_to_results() -> None:
    qubits = cirq.LineQubit.range(3)

    circuit = cirq.Circuit(
        cirq.H(qubits[1]),
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.measure(qubits[0]),
        cirq.measure(qubits[1]),
    )
    result = cirq_superstaq.service.counts_to_results(
        collections.Counter({"01": 1, "11": 2}), circuit, cirq.ParamResolver({})
    )
    assert result.histogram(key="01") == collections.Counter({3: 2, 1: 1})

    circuit = cirq.Circuit(
        cirq.H(qubits[0]),
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.measure(qubits[0]),
        cirq.measure(qubits[1]),
    )
    result = cirq_superstaq.service.counts_to_results(
        collections.Counter({"00": 50, "11": 50}), circuit, cirq.ParamResolver({})
    )
    assert result.histogram(key="01") == collections.Counter({0: 50, 3: 50})


def test_service_run_and_get_counts() -> None:
    service = cirq_superstaq.Service(remote_host="http://example.com", api_key="key")
    mock_client = mock.MagicMock()
    mock_client.create_job.return_value = {
        "job_id": "job_id",
        "status": "ready",
    }
    mock_client.get_job.return_value = {
        "data": {"histogram": {"11": 1}},
        "job_id": "my_id",
        "samples": {"11": 1},
        "shots": [
            {
                "data": {"counts": {"0x3": 1}},
                "meas_level": 2,
                "seed_simulator": 775709958,
                "shots": 1,
                "status": "DONE",
            }
        ],
        "status": "Done",
        "target": "simulator",
    }

    service._client = mock_client

    a = sympy.Symbol("a")
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit((cirq.X ** a)(q), cirq.measure(q, key="a"))
    params = cirq.ParamResolver({"a": 0.5})
    counts = service.get_counts(
        circuit=circuit,
        repetitions=4,
        target="ibmq_qasm_simulator",
        name="bacon",
        param_resolver=params,
    )
    assert counts == {"11": 1}

    result = service.run(
        circuit=circuit,
        repetitions=4,
        target="ibmq_qasm_simulator",
        name="bacon",
        param_resolver=params,
    )
    assert result.histogram(key="a") == collections.Counter({3: 1})


def test_service_get_job() -> None:
    service = cirq_superstaq.Service(remote_host="http://example.com", api_key="key")
    mock_client = mock.MagicMock()
    job_dict = {"job_id": "job_id", "status": "ready"}
    mock_client.get_job.return_value = job_dict
    service._client = mock_client

    job = service.get_job("job_id")
    assert job.job_id() == "job_id"
    mock_client.get_job.assert_called_with(job_id="job_id")


def test_service_create_job() -> None:
    service = cirq_superstaq.Service(remote_host="http://example.com", api_key="key")
    mock_client = mock.MagicMock()
    mock_client.create_job.return_value = {"job_id": "job_id", "status": "ready"}
    mock_client.get_job.return_value = {"job_id": "job_id", "status": "completed"}
    service._client = mock_client

    circuit = cirq.Circuit(cirq.X(cirq.LineQubit(0)))
    job = service.create_job(circuit=circuit, repetitions=100, target="qpu")
    assert job.status() == "completed"
    create_job_kwargs = mock_client.create_job.call_args[1]
    # Serialization induces a float, so we don't validate full circuit.
    assert create_job_kwargs["repetitions"] == 100
    assert create_job_kwargs["target"] == "qpu"


def test_service_get_balance() -> None:
    service = cirq_superstaq.Service(remote_host="http://example.com", api_key="key")
    mock_client = mock.MagicMock()
    mock_client.get_balance.return_value = {"balance": 12345.6789}
    service._client = mock_client

    assert service.get_balance() == "$12,345.68"
    assert service.get_balance(pretty_output=False) == 12345.6789


@mock.patch(
    "cirq_superstaq.superstaq_client._SuperstaQClient.aqt_compile",
    return_value={
        "cirq_circuits": [cirq.to_json(cirq.Circuit())],
        "state_jp": applications_superstaq.converters.serialize({}),
        "pulse_lists_jp": applications_superstaq.converters.serialize([[[]]]),
    },
)
def test_service_aqt_compile_single(mock_aqt_compile: mock.MagicMock) -> None:
    service = cirq_superstaq.Service(remote_host="http://example.com", api_key="key")
    out = service.aqt_compile(cirq.Circuit())
    assert out.circuit == cirq.Circuit()
    assert not hasattr(out, "circuits") and not hasattr(out, "pulse_lists")


@mock.patch(
    "cirq_superstaq.superstaq_client._SuperstaQClient.aqt_compile",
    return_value={
        "cirq_circuits": [cirq.to_json(cirq.Circuit()), cirq.to_json(cirq.Circuit())],
        "state_jp": applications_superstaq.converters.serialize({}),
        "pulse_lists_jp": applications_superstaq.converters.serialize([[[]], [[]]]),
    },
)
def test_service_aqt_compile_multiple(mock_aqt_compile: mock.MagicMock) -> None:
    service = cirq_superstaq.Service(remote_host="http://example.com", api_key="key")
    out = service.aqt_compile([cirq.Circuit(), cirq.Circuit()])
    assert out.circuits == [cirq.Circuit(), cirq.Circuit()]
    assert not hasattr(out, "circuit") and not hasattr(out, "pulse_list")


@mock.patch(
    "cirq_superstaq.superstaq_client._SuperstaQClient.ibmq_compile",
    return_value={"pulses": applications_superstaq.converters.serialize([mock.DEFAULT])},
)
def test_service_ibmq_compile(mock_ibmq_compile: mock.MagicMock) -> None:
    service = cirq_superstaq.Service(remote_host="http://example.com", api_key="key")
    assert service.ibmq_compile(cirq.Circuit()) == mock.DEFAULT

    with mock.patch.dict("sys.modules", {"unittest": None}), pytest.raises(
        cirq_superstaq.SuperstaQModuleNotFoundException,
        match="'ibmq_compile' requires module 'unittest'",
    ):
        _ = service.ibmq_compile(cirq.Circuit())


@mock.patch(
    "cirq_superstaq.superstaq_client._SuperstaQClient.submit_qubo",
    return_value={
        "solution": applications_superstaq.converters.serialize(
            np.rec.array(
                [({0: 0, 1: 1, 3: 1}, -1, 6), ({0: 1, 1: 1, 3: 1}, -1, 4)],
                dtype=[("solution", "O"), ("energy", "<f8"), ("num_occurrences", "<i8")],
            )
        )
    },
)
def test_service_submit_qubo(mock_submit_qubo: mock.MagicMock) -> None:
    service = cirq_superstaq.Service(remote_host="http://example.com", api_key="key")
    expected = np.rec.array(
        [({0: 0, 1: 1, 3: 1}, -1, 6), ({0: 1, 1: 1, 3: 1}, -1, 4)],
        dtype=[("solution", "O"), ("energy", "<f8"), ("num_occurrences", "<i8")],
    )
    assert repr(service.submit_qubo(qv.QUBO(), "target", repetitions=10)) == repr(expected)


@mock.patch(
    "cirq_superstaq.superstaq_client._SuperstaQClient.find_min_vol_portfolio",
    return_value={
        "best_portfolio": ["AAPL", "GOOG"],
        "best_ret": 8.1,
        "best_std_dev": 10.5,
        "qubo": [{"keys": ["0"], "value": 123}],
    },
)
def test_service_find_min_vol_portfolio(mock_find_min_vol_portfolio: mock.MagicMock) -> None:
    service = cirq_superstaq.Service(remote_host="http://example.com", api_key="key")
    qubo = {("0",): 123}
    expected = applications_superstaq.finance.MinVolOutput(["AAPL", "GOOG"], 8.1, 10.5, qubo)
    assert service.find_min_vol_portfolio(["AAPL", "GOOG", "IEF", "MMM"], 8) == expected


@mock.patch(
    "cirq_superstaq.superstaq_client._SuperstaQClient.find_max_pseudo_sharpe_ratio",
    return_value={
        "best_portfolio": ["AAPL", "GOOG"],
        "best_ret": 8.1,
        "best_std_dev": 10.5,
        "best_sharpe_ratio": 0.771,
        "qubo": [{"keys": ["0"], "value": 123}],
    },
)
def test_service_find_max_pseudo_sharpe_ratio(
    mock_find_max_pseudo_sharpe_ratio: mock.MagicMock,
) -> None:
    service = cirq_superstaq.Service(remote_host="http://example.com", api_key="key")
    qubo = {("0",): 123}
    expected = applications_superstaq.finance.MaxSharpeOutput(
        ["AAPL", "GOOG"], 8.1, 10.5, 0.771, qubo
    )
    assert service.find_max_pseudo_sharpe_ratio(["AAPL", "GOOG", "IEF", "MMM"], k=0.5) == expected


@mock.patch(
    "cirq_superstaq.superstaq_client._SuperstaQClient.tsp",
    return_value={
        "route": ["Chicago", "St Louis", "St Paul", "Chicago"],
        "route_list_numbers": [0, 1, 2, 0],
        "total_distance": 100.0,
        "map_link": ["maps.google.com"],
        "qubo": [{"keys": ["0"], "value": 123}],
    },
)
def test_service_tsp(mock_tsp: mock.MagicMock) -> None:
    service = cirq_superstaq.Service(remote_host="http://example.com", api_key="key")
    qubo = {("0",): 123}
    expected = applications_superstaq.logistics.TSPOutput(
        ["Chicago", "St Louis", "St Paul", "Chicago"],
        [0, 1, 2, 0],
        100.0,
        ["maps.google.com"],
        qubo,
    )
    assert service.tsp(["Chicago", "St Louis", "St Paul"]) == expected


@mock.patch(
    "cirq_superstaq.superstaq_client._SuperstaQClient.warehouse",
    return_value={
        "warehouse_to_destination": [("Chicago", "Rockford"), ("Chicago", "Aurora")],
        "total_distance": 100.0,
        "map_link": "map.html",
        "open_warehouses": ["Chicago"],
        "qubo": [{"keys": ["0"], "value": 123}],
    },
)
def test_service_warehouse(mock_warehouse: mock.MagicMock) -> None:
    service = cirq_superstaq.Service(remote_host="http://example.com", api_key="key")
    qubo = {("0",): 123}
    expected = applications_superstaq.logistics.WarehouseOutput(
        [("Chicago", "Rockford"), ("Chicago", "Aurora")], 100.0, "map.html", ["Chicago"], qubo
    )
    assert service.warehouse(1, ["Chicago", "San Francisco"], ["Rockford", "Aurora"]) == expected


@mock.patch(
    "cirq_superstaq.superstaq_client._SuperstaQClient.aqt_upload_configs",
    return_value={"status": "Your AQT configuration has been updated"},
)
def test_service_aqt_upload_configs(mock_aqt_compile: mock.MagicMock) -> None:
    service = cirq_superstaq.Service(remote_host="http://example.com", api_key="key")

    with open("/tmp/Pulses.yaml", "w") as pulses_file:
        pulses_file.write("Hello")

    with open("/tmp/Variables.yaml", "w") as variables_file:
        variables_file.write("World")

    assert service.aqt_upload_configs("/tmp/Pulses.yaml", "/tmp/Variables.yaml") == {
        "status": "Your AQT configuration has been updated"
    }


def test_service_api_key_via_env() -> None:
    os.environ["SUPERSTAQ_API_KEY"] = "tomyheart"
    service = cirq_superstaq.Service(remote_host="http://example.com")
    assert service.api_key == "tomyheart"
    del os.environ["SUPERSTAQ_API_KEY"]


def test_service_remote_host_via_env() -> None:
    os.environ["SUPERSTAQ_REMOTE_HOST"] = "http://example.com"
    service = cirq_superstaq.Service(api_key="tomyheart")
    assert service.remote_host == "http://example.com"
    del os.environ["SUPERSTAQ_REMOTE_HOST"]


def test_service_no_param_or_env_variable() -> None:
    with pytest.raises(EnvironmentError):
        _ = cirq_superstaq.Service(remote_host="http://example.com")


def test_service_no_url_default() -> None:
    service = cirq_superstaq.Service(api_key="tomyheart")
    assert service.remote_host == cirq_superstaq.API_URL
