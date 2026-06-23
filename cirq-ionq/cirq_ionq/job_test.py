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

from __future__ import annotations

import json
import warnings
from unittest import mock

import pytest

import cirq_ionq as ionq


def test_job_fields():
    job_dict = {
        "id": "my_id",
        "backend": "qpu",
        "name": "bacon",
        "stats": {"qubits": "5"},
        "status": "completed",
        "metadata": {"shots": 1000, "measurement0": f"a{chr(31)}0,1"},
    }
    job = ionq.Job(None, job_dict)
    assert job.job_id() == "my_id"
    assert job.target() == "qpu"
    assert job.name() == "bacon"
    assert job.num_qubits() == 5
    assert job.repetitions() == 1000
    assert job.measurement_dict() == {"a": [0, 1]}


def test_job_fields_multiple_circuits():
    job_dict = {
        "id": "my_id",
        "backend": "qpu",
        "name": "bacon",
        "stats": {"qubits": "5"},
        "status": "completed",
        "metadata": {
            "shots": 1000,
            "measurements": json.dumps([{"measurement0": f"a{chr(31)}0,1"}]),
        },
    }
    job = ionq.Job(None, job_dict)
    assert job.job_id() == "my_id"
    assert job.target() == "qpu"
    assert job.name() == "bacon"
    assert job.num_qubits() == 5
    assert job.repetitions() == 1000
    assert job.measurement_dict() == {"a": [0, 1]}


def test_job_status_refresh():
    for status in ionq.Job.NON_TERMINAL_STATES:
        mock_client = mock.MagicMock()
        mock_client.get_job.return_value = {"id": "my_id", "status": "completed"}
        job = ionq.Job(mock_client, {"id": "my_id", "status": status})
        assert job.status() == "completed"
        mock_client.get_job.assert_called_with("my_id")
    for status in ionq.Job.TERMINAL_STATES:
        mock_client = mock.MagicMock()
        job = ionq.Job(mock_client, {"id": "my_id", "status": status})
        assert job.status() == status
        mock_client.get_job.assert_not_called()


def test_job_str():
    job = ionq.Job(None, {"id": "my_id"})
    assert str(job) == "cirq_ionq.Job(job_id=my_id)"


def test_job_results_qpu():
    mock_client = mock.MagicMock()
    mock_client.get_results.return_value = {"0": "0.6", "2": "0.4"}
    job_dict = {
        "id": "my_id",
        "status": "completed",
        "stats": {"qubits": "2"},
        "backend": "qpu",
        "metadata": {"shots": 1000, "measurement0": f"a{chr(31)}0,1"},
        "warning": {"messages": ["foo", "bar"]},
    }
    job = ionq.Job(mock_client, job_dict)
    with warnings.catch_warnings(record=True) as w:
        results = job.results()
        assert len(w) == 2
        assert "foo" in str(w[0].message)
        assert "bar" in str(w[1].message)
    expected = ionq.QPUResult({0: 600, 1: 400}, 2, {"a": [0, 1]})
    assert results == expected


def test_batch_job_results_qpu():
    mock_client = mock.MagicMock()
    mock_client.get_results.return_value = {
        "0190070f-9691-7000-a1f6-306623179a83": {"0": "0.6", "2": "0.4"},
        "0190070f-991c-7000-8700-c4b56b30715d": {"1": 1.0},
    }
    job_dict = {
        "id": "my_id",
        "status": "completed",
        "stats": {"qubits": "2"},
        "backend": "qpu",
        "metadata": {
            "shots": 1000,
            "measurements": json.dumps(
                [{"measurement0": f"a{chr(31)}0,1"}, {"measurement0": f"a{chr(31)}0"}]
            ),
            "qubit_numbers": json.dumps([2, 1]),
        },
        "warning": {"messages": ["foo", "bar"]},
    }
    job = ionq.Job(mock_client, job_dict)
    with warnings.catch_warnings(record=True) as w:
        results = job.results()
        assert len(w) == 2
        assert "foo" in str(w[0].message)
        assert "bar" in str(w[1].message)
    expected_0 = ionq.QPUResult({0: 600, 1: 400}, 2, {"a": [0, 1]})
    expected_1 = ionq.QPUResult({1: 1000}, 1, {"a": [0]})
    assert results[0] == expected_0
    assert results[1] == expected_1


def test_job_results_rounding_qpu():
    mock_client = mock.MagicMock()
    mock_client.get_results.return_value = {"0": "0.0006", "2": "0.9994"}
    job_dict = {
        "id": "my_id",
        "status": "completed",
        "stats": {"qubits": "2"},
        "backend": "qpu",
        "metadata": {"shots": 5000, "measurement0": f"a{chr(31)}0,1"},
    }
    # 5000*0.0006 ~ 2.9999 but should be interpreted as 3
    job = ionq.Job(mock_client, job_dict)
    expected = ionq.QPUResult({0: 3, 1: 4997}, 2, {"a": [0, 1]})
    results = job.results()
    assert results == expected


def test_job_results_failed():
    job_dict = {"id": "my_id", "status": "failed", "failure": {"error": "too many qubits"}}
    job = ionq.Job(None, job_dict)
    with pytest.raises(RuntimeError, match="too many qubits"):
        _ = job.results()
    assert job.status() == "failed"


def test_job_results_failed_no_error_message():
    job_dict = {"id": "my_id", "status": "failed", "failure": {}}
    job = ionq.Job(None, job_dict)
    with pytest.raises(RuntimeError, match="failed"):
        _ = job.results()
    assert job.status() == "failed"


def test_job_results_qpu_endianness():
    mock_client = mock.MagicMock()
    mock_client.get_results.return_value = {"0": "0.6", "1": "0.4"}
    job_dict = {
        "id": "my_id",
        "status": "completed",
        "stats": {"qubits": "2"},
        "backend": "qpu",
        "metadata": {"shots": 1000},
    }
    job = ionq.Job(mock_client, job_dict)
    results = job.results()
    assert results == ionq.QPUResult({0: 600, 2: 400}, 2, measurement_dict={})


def test_batch_job_results_qpu_endianness():
    mock_client = mock.MagicMock()
    mock_client.get_results.return_value = {
        "0190070f-9691-7000-a1f6-306623179a83": {"0": "0.6", "1": "0.4"}
    }
    job_dict = {
        "id": "my_id",
        "status": "completed",
        "stats": {"qubits": "2"},
        "backend": "qpu",
        "metadata": {
            "shots": 1000,
            "measurements": json.dumps([{"measurement0": f"a{chr(31)}0,1"}]),
            "qubit_numbers": json.dumps([2]),
        },
    }
    job = ionq.Job(mock_client, job_dict)
    results = job.results()
    assert results == ionq.QPUResult({0: 600, 2: 400}, 2, measurement_dict={"a": [0, 1]})


def test_job_results_qpu_target_endianness():
    mock_client = mock.MagicMock()
    mock_client.get_results.return_value = {"0": "0.6", "1": "0.4"}
    job_dict = {
        "id": "my_id",
        "status": "completed",
        "stats": {"qubits": "2"},
        "backend": "qpu.target",
        "metadata": {"shots": 1000},
        "data": {"histogram": {"0": "0.6", "1": "0.4"}},
    }
    job = ionq.Job(mock_client, job_dict)
    results = job.results()
    assert results == ionq.QPUResult({0: 600, 2: 400}, 2, measurement_dict={})


def test_batch_job_results_qpu_target_endianness():
    mock_client = mock.MagicMock()
    mock_client.get_results.return_value = {
        "0190070f-9691-7000-a1f6-306623179a83": {"0": "0.6", "1": "0.4"}
    }
    job_dict = {
        "id": "my_id",
        "status": "completed",
        "stats": {"qubits": "2"},
        "backend": "qpu.target",
        "metadata": {
            "shots": 1000,
            "measurements": json.dumps([{"measurement0": f"a{chr(31)}0,1"}]),
            "qubit_numbers": json.dumps([2]),
        },
        "data": {"histogram": {"0": "0.6", "1": "0.4"}},
    }
    job = ionq.Job(mock_client, job_dict)
    results = job.results()
    assert results == ionq.QPUResult({0: 600, 2: 400}, 2, measurement_dict={"a": [0, 1]})


@mock.patch("time.sleep", return_value=None)
def test_job_results_poll(mock_sleep):
    ready_job = {"id": "my_id", "status": "ready"}
    completed_job = {
        "id": "my_id",
        "status": "completed",
        "stats": {"qubits": "1"},
        "backend": "qpu",
        "metadata": {"shots": 1000},
    }
    mock_client = mock.MagicMock()
    mock_client.get_job.side_effect = [ready_job, completed_job]
    mock_client.get_results.return_value = {"0": "0.6", "1": "0.4"}
    job = ionq.Job(mock_client, ready_job)
    results = job.results(polling_seconds=0)
    assert results == ionq.QPUResult({0: 600, 1: 400}, 1, measurement_dict={})
    mock_sleep.assert_called_once()


@mock.patch("time.sleep", return_value=None)
def test_job_results_poll_timeout(mock_sleep):
    ready_job = {"id": "my_id", "status": "ready"}
    mock_client = mock.MagicMock()
    mock_client.get_job.return_value = ready_job
    job = ionq.Job(mock_client, ready_job)
    with pytest.raises(TimeoutError, match="seconds"):
        _ = job.results(timeout_seconds=1, polling_seconds=0.1)
    assert mock_sleep.call_count == 11


@mock.patch("time.sleep", return_value=None)
def test_job_results_poll_timeout_with_error_message(mock_sleep):
    ready_job = {"id": "my_id", "status": "failure", "failure": {"error": "too many qubits"}}
    mock_client = mock.MagicMock()
    mock_client.get_job.return_value = ready_job
    job = ionq.Job(mock_client, ready_job)
    with pytest.raises(RuntimeError, match="too many qubits"):
        _ = job.results(timeout_seconds=1, polling_seconds=0.1)
    assert mock_sleep.call_count == 11


def test_job_results_simulator():
    mock_client = mock.MagicMock()
    mock_client.get_results.return_value = {"0": "0.6", "1": "0.4"}
    job_dict = {
        "id": "my_id",
        "status": "completed",
        "stats": {"qubits": "1"},
        "backend": "simulator",
        "metadata": {"shots": "100"},
    }
    job = ionq.Job(mock_client, job_dict)
    results = job.results()
    assert results == ionq.SimulatorResult({0: 0.6, 1: 0.4}, 1, {}, 100)


def test_batch_job_results_simulator():
    mock_client = mock.MagicMock()
    mock_client.get_results.return_value = {
        "0190070f-9691-7000-a1f6-306623179a83": {"0": "0.6", "2": "0.4"},
        "0190070f-991c-7000-8700-c4b56b30715d": {"1": 1.0},
    }
    job_dict = {
        "id": "my_id",
        "status": "completed",
        "stats": {"qubits": "2"},
        "backend": "simulator",
        "metadata": {
            "shots": 1000,
            "measurements": json.dumps(
                [{"measurement0": f"a{chr(31)}0,1"}, {"measurement0": f"a{chr(31)}0"}]
            ),
            "qubit_numbers": json.dumps([2, 1]),
        },
    }
    job = ionq.Job(mock_client, job_dict)
    results = job.results()
    expected_0 = ionq.SimulatorResult({0: 0.6, 1: 0.4}, 2, {"a": [0, 1]}, repetitions=1000)
    expected_1 = ionq.SimulatorResult({1: 1}, 1, {"a": [0]}, repetitions=1000)
    assert results[0] == expected_0
    assert results[1] == expected_1


def test_job_results_simulator_endianness():
    mock_client = mock.MagicMock()
    mock_client.get_results.return_value = {"0": "0.6", "1": "0.4"}
    job_dict = {
        "id": "my_id",
        "status": "completed",
        "stats": {"qubits": "2"},
        "backend": "simulator",
        "metadata": {"shots": "100"},
    }
    job = ionq.Job(mock_client, job_dict)
    results = job.results()
    assert results == ionq.SimulatorResult({0: 0.6, 2: 0.4}, 2, {}, 100)


def test_batch_job_results_simulator_endianness():
    mock_client = mock.MagicMock()
    mock_client.get_results.return_value = {
        "0190070f-9691-7000-a1f6-306623179a83": {"0": "0.6", "1": "0.4"}
    }
    job_dict = {
        "id": "my_id",
        "status": "completed",
        "stats": {"qubits": "2"},
        "backend": "simulator",
        "metadata": {
            "shots": 1000,
            "measurements": json.dumps([{"measurement0": f"a{chr(31)}0,1"}]),
            "qubit_numbers": json.dumps([2]),
        },
    }
    job = ionq.Job(mock_client, job_dict)
    results = job.results()
    assert results == ionq.SimulatorResult({0: 0.6, 2: 0.4}, 2, {"a": [0, 1]}, 1000)


def test_job_sharpen_results():
    mock_client = mock.MagicMock()
    mock_client.get_results.return_value = {"0": "60", "1": "40"}
    job_dict = {
        "id": "my_id",
        "status": "completed",
        "stats": {"qubits": "1"},
        "backend": "simulator",
        "metadata": {"shots": "100"},
    }
    job = ionq.Job(mock_client, job_dict)
    results = job.results(sharpen=False)
    assert results == ionq.SimulatorResult({0: 60, 1: 40}, 1, {}, 100)


def test_job_cancel():
    ready_job = {"id": "my_id", "status": "ready"}
    canceled_job = {"id": "my_id", "status": "canceled"}
    mock_client = mock.MagicMock()
    mock_client.cancel_job.return_value = canceled_job
    job = ionq.Job(mock_client, ready_job)
    job.cancel()
    mock_client.cancel_job.assert_called_with(job_id="my_id")
    assert job.status() == "canceled"


def test_job_delete():
    ready_job = {"id": "my_id", "status": "ready"}
    deleted_job = {"id": "my_id", "status": "deleted"}
    mock_client = mock.MagicMock()
    mock_client.delete_job.return_value = deleted_job
    job = ionq.Job(mock_client, ready_job)
    job.delete()
    mock_client.delete_job.assert_called_with(job_id="my_id")
    assert job.status() == "deleted"


def test_job_fields_unsuccessful():
    job_dict = {
        "id": "my_id",
        "backend": "qpu",
        "name": "bacon",
        "stats": {"qubits": "5"},
        "status": "deleted",
        "metadata": {"shots": 1000},
    }
    job = ionq.Job(None, job_dict)
    with pytest.raises(ionq.IonQUnsuccessfulJobException, match="deleted"):
        _ = job.target()
    with pytest.raises(ionq.IonQUnsuccessfulJobException, match="deleted"):
        _ = job.name()
    with pytest.raises(ionq.IonQUnsuccessfulJobException, match="deleted"):
        _ = job.num_qubits()
    with pytest.raises(ionq.IonQUnsuccessfulJobException, match="deleted"):
        _ = job.repetitions()


def test_job_fields_cannot_get_status():
    job_dict = {
        "id": "my_id",
        "backend": "qpu",
        "name": "bacon",
        "stats": {"qubits": "5"},
        "status": "running",
        "metadata": {"shots": 1000},
    }
    mock_client = mock.MagicMock()
    mock_client.get_job.side_effect = ionq.IonQException("bad")
    job = ionq.Job(mock_client, job_dict)
    with pytest.raises(ionq.IonQException, match="bad"):
        _ = job.target()
    with pytest.raises(ionq.IonQException, match="bad"):
        _ = job.name()
    with pytest.raises(ionq.IonQException, match="bad"):
        _ = job.num_qubits()
    with pytest.raises(ionq.IonQException, match="bad"):
        _ = job.repetitions()


def test_job_fields_update_status():
    job_dict = {
        "id": "my_id",
        "backend": "qpu",
        "name": "bacon",
        "stats": {"qubits": "5"},
        "status": "running",
        "metadata": {"shots": 1000},
    }
    mock_client = mock.MagicMock()
    mock_client.get_job.return_value = job_dict
    job = ionq.Job(mock_client, job_dict)
    assert job.job_id() == "my_id"
    assert job.target() == "qpu"
    assert job.name() == "bacon"
    assert job.num_qubits() == 5
    assert job.repetitions() == 1000


@pytest.mark.parametrize("memory", [True, False])
def test_memory_job_results_ideal_simulator(memory):
    mock_client = mock.MagicMock()
    mock_client.get_shots.return_value = [2, 1, 3, 1, 0]
    mock_client.get_results.return_value = {"0": "1"}
    job_dict = {
        "id": "my_id",
        "status": "completed",
        "stats": {"qubits": "2"},
        "backend": "simulator",
        "metadata": {
            "shots": "5",
            "measurements": json.dumps([{"measurement0": f"results{chr(31)}0,1"}]),
        },
        "results": {"shots": {"url": "/shots"}},
        "noise": {"model": "ideal"},
    }
    job = ionq.Job(mock_client, job_dict, memory=memory)
    result = job.results()
    cirq_result = result.to_cirq_result()
    assert cirq_result.measurements["results"].tolist() == [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    mock_client.get_shots.assert_not_called()


@pytest.mark.parametrize("memory", [True, False])
def test_memory_job_results_noisy_simulator(memory):
    mock_client = mock.MagicMock()
    mock_client.get_results.return_value = {"0": "0.6", "1": "0.4"}
    mock_client.get_shots.return_value = [2, 1, 3, 1, 0]
    job_dict = {
        "id": "my_id",
        "status": "completed",
        "stats": {"qubits": "2"},
        "backend": "simulator",
        "metadata": {
            "shots": "5",
            "measurements": json.dumps([{"measurement0": f"results{chr(31)}0,1"}]),
        },
        "results": {"shots": {"url": "/shots"}},
        "noise": {"model": "aria-1"},
    }
    job = ionq.Job(mock_client, job_dict, memory=memory)
    result = job.results()
    if memory:
        cirq_result = result.to_cirq_result()
        expected = [[0, 1], [1, 0], [1, 1], [1, 0], [0, 0]]
        mock_client.get_shots.assert_called_once_with("/shots")
    else:
        fake_random_state = mock.Mock()
        fake_random_state.choice.return_value = [1, 0, 1, 1, 0]
        with mock.patch(
            "cirq_ionq.results.cirq.value.parse_random_state", return_value=fake_random_state
        ) as parse_random_state:
            cirq_result = result.to_cirq_result()

        expected = [[1, 0], [0, 0], [1, 0], [1, 0], [0, 0]]
        parse_random_state.assert_called_once_with(None)
        fake_random_state.choice.assert_called_once_with(range(2), p=(0.6, 0.4), size=5)
        mock_client.get_shots.assert_not_called()

    assert cirq_result.measurements["results"].tolist() == expected


@pytest.mark.parametrize("memory", [True, False])
def test_memory_job_results_qpu(memory):
    mock_client = mock.MagicMock()
    mock_client.get_results.return_value = {"0": "0.6", "3": "0.4"}
    mock_client.get_shots.return_value = [2, 1, 3, 1, 0]
    job_dict = {
        "id": "my_id",
        "status": "completed",
        "stats": {"qubits": "2"},
        "backend": "qpu",
        "metadata": {
            "shots": 5,
            "measurements": json.dumps([{"measurement0": f"results{chr(31)}0,1"}]),
        },
        "results": {"shots": {"url": "/shots"}},
    }
    job = ionq.Job(mock_client, job_dict, memory=memory)
    result = job.results()
    cirq_result = result.to_cirq_result()
    expected = (
        [[0, 1], [1, 0], [1, 1], [1, 0], [0, 0]]
        if memory
        else [[0, 0], [0, 0], [0, 0], [1, 1], [1, 1]]
    )
    assert cirq_result.measurements["results"].tolist() == expected
    if memory:
        mock_client.get_shots.assert_called_once_with("/shots")
    else:
        mock_client.get_shots.assert_not_called()


def test_retrieve_job_shots_key_error():
    """Test _retrieve_job_shots handles KeyError when shots url is missing."""
    mock_client = mock.MagicMock()
    job_dict = {
        "id": "my_id",
        "status": "completed",
        "stats": {"qubits": "2"},
        "backend": "qpu",
        "metadata": {"shots": 5},
        "results": {},  # Missing 'shots' key
    }
    job = ionq.Job(mock_client, job_dict, memory=True)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = job._retrieve_job_shots()

        # Verify warning was raised
        assert len(w) == 1
        assert "memory argument" in str(w[0].message)
        assert "url for shots result was not found" in str(w[0].message)

    # Verify result is [None]
    assert result == [None]
    # Verify get_shots was not called
    mock_client.get_shots.assert_not_called()


@pytest.mark.parametrize(
    "exception,error_msg",
    [
        (ionq.IonQException("API error"), "API error"),
        (ionq.IonQNotFoundException("Job not found"), "Job not found"),
        (TimeoutError("Request timed out"), "Request timed out"),
    ],
)
def test_retrieve_job_shots_api_errors(exception, error_msg):
    """Test _retrieve_job_shots handles various API errors."""
    mock_client = mock.MagicMock()
    mock_client.get_shots.side_effect = exception

    job_dict = {
        "id": "my_id",
        "status": "completed",
        "stats": {"qubits": "2"},
        "backend": "qpu",
        "metadata": {"shots": 5},
        "results": {"shots": {"url": "/shots"}},
    }
    job = ionq.Job(mock_client, job_dict, memory=True)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = job._retrieve_job_shots()

        # Verify warning was raised
        assert len(w) == 1
        assert "memory argument" in str(w[0].message)
        assert error_msg in str(w[0].message)

    # Verify result is [None]
    assert result == [None]
    # Verify get_shots was called
    mock_client.get_shots.assert_called_once_with("/shots")


def test_retrieve_job_shots_success():
    """Test _retrieve_job_shots successfully retrieves shots."""
    mock_client = mock.MagicMock()
    mock_shots = [2, 1, 3, 1, 0]
    mock_client.get_shots.return_value = mock_shots

    job_dict = {
        "id": "my_id",
        "status": "completed",
        "stats": {"qubits": "2"},
        "backend": "qpu",
        "metadata": {"shots": 5},
        "results": {"shots": {"url": "/shots"}},
    }
    job = ionq.Job(mock_client, job_dict, memory=True)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = job._retrieve_job_shots()

        # Verify no warning was raised
        assert len(w) == 0

    # Verify result contains shots
    assert result == [mock_shots]
    # Verify get_shots was called
    mock_client.get_shots.assert_called_once_with("/shots")


def test_retrieve_child_job_shots_key_error():
    """Test _retrieve_child_job_shots handles KeyError when shots url is missing."""
    mock_client = mock.MagicMock()
    # First child job has missing shots URL
    mock_client.get_job.return_value = {
        "id": "child_1",
        "status": "completed",
        "results": {},  # Missing 'shots' key
    }

    job_dict = {
        "id": "batch_job",
        "status": "completed",
        "stats": {"qubits": "2"},
        "backend": "qpu",
        "metadata": {"shots": 5},
    }
    job = ionq.Job(mock_client, job_dict, memory=True)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = job._retrieve_child_job_shots(["child_1"])

        # Verify warning was raised
        assert len(w) == 1
        assert "memory argument" in str(w[0].message)
        assert "url for shots result was not found" in str(w[0].message)
        assert "child_1" in str(w[0].message)

    # Verify result is [None] for failed child
    assert result == [None]
    # Verify get_shots was not called
    mock_client.get_shots.assert_not_called()


@pytest.mark.parametrize(
    "exception,error_msg",
    [
        (ionq.IonQException("API error"), "API error"),
        (ionq.IonQNotFoundException("Job not found"), "Job not found"),
        (TimeoutError("Request timed out"), "Request timed out"),
    ],
)
def test_retrieve_child_job_shots_api_errors(exception, error_msg):
    """Test _retrieve_child_job_shots handles various API errors for child jobs."""
    mock_client = mock.MagicMock()
    mock_client.get_job.return_value = {
        "id": "child_1",
        "status": "completed",
        "results": {"shots": {"url": "/shots"}},
    }
    mock_client.get_shots.side_effect = exception

    job_dict = {
        "id": "batch_job",
        "status": "completed",
        "stats": {"qubits": "2"},
        "backend": "qpu",
        "metadata": {"shots": 5},
    }
    job = ionq.Job(mock_client, job_dict, memory=True)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = job._retrieve_child_job_shots(["child_1"])

        # Verify warning was raised
        assert len(w) == 1
        assert "memory argument" in str(w[0].message)
        assert error_msg in str(w[0].message)
        assert "child_1" in str(w[0].message)

    # Verify result is [None] for failed child
    assert result == [None]
    # Verify get_shots was called
    mock_client.get_shots.assert_called_once_with("/shots")


def test_retrieve_child_job_shots_success():
    """Test _retrieve_child_job_shots successfully retrieves shots for multiple child jobs."""
    mock_client = mock.MagicMock()
    mock_shots_1 = [2, 1, 3, 1, 0]
    mock_shots_2 = [0, 1, 0, 1, 1]

    def get_job_side_effect(job_id):
        if job_id == "child_1":
            return {
                "id": "child_1",
                "status": "completed",
                "results": {"shots": {"url": "/shots/child_1"}},
            }
        else:
            return {
                "id": "child_2",
                "status": "completed",
                "results": {"shots": {"url": "/shots/child_2"}},
            }

    mock_client.get_job.side_effect = get_job_side_effect
    mock_client.get_shots.side_effect = [mock_shots_1, mock_shots_2]

    job_dict = {
        "id": "batch_job",
        "status": "completed",
        "stats": {"qubits": "2"},
        "backend": "qpu",
        "metadata": {"shots": 5},
    }
    job = ionq.Job(mock_client, job_dict, memory=True)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = job._retrieve_child_job_shots(["child_1", "child_2"])

        # Verify no warning was raised
        assert len(w) == 0

    # Verify results contain shots for both children
    assert result == [mock_shots_1, mock_shots_2]
    # Verify get_shots was called twice
    assert mock_client.get_shots.call_count == 2


def test_retrieve_child_job_shots_partial_failure():
    """Test _retrieve_child_job_shots handles partial failures across child jobs."""
    mock_client = mock.MagicMock()
    mock_shots_1 = [2, 1, 3, 1, 0]

    def get_job_side_effect(job_id):
        if job_id == "child_1":
            return {
                "id": "child_1",
                "status": "completed",
                "results": {"shots": {"url": "/shots/child_1"}},
            }
        else:
            # child_2 is missing shots URL
            return {
                "id": "child_2",
                "status": "completed",
                "results": {},
            }

    mock_client.get_job.side_effect = get_job_side_effect
    mock_client.get_shots.return_value = mock_shots_1

    job_dict = {
        "id": "batch_job",
        "status": "completed",
        "stats": {"qubits": "2"},
        "backend": "qpu",
        "metadata": {"shots": 5},
    }
    job = ionq.Job(mock_client, job_dict, memory=True)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = job._retrieve_child_job_shots(["child_1", "child_2"])

        # Verify one warning was raised for child_2
        assert len(w) == 1
        assert "memory argument" in str(w[0].message)
        assert "child_2" in str(w[0].message)

    # Verify results: child_1 succeeds, child_2 fails
    assert result == [mock_shots_1, None]
    # Verify get_shots was called once (for child_1 only)
    mock_client.get_shots.assert_called_once_with("/shots/child_1")
