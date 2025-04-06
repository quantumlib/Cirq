# Copyright 2019 The Cirq Developers
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

import json
from unittest import mock

import numpy as np
import pytest
import sympy

import cirq
from cirq_aqt import AQTSampler, AQTSamplerLocalSimulator
from cirq_aqt.aqt_device import get_aqt_device, get_op_string


class GetResultReturn:
    """A put mock class for testing the REST interface"""

    def __init__(self):
        self.test_dict = {'job': {'job_id': '2131da'}, 'response': {'status': 'queued'}}
        self.counter = 0

    def json(self):
        self.counter += 1
        return self.test_dict

    def update(self, *args, **kwargs):
        return self


class GetResultError(GetResultReturn):
    """A put mock class for testing error responses"""

    def __init__(self):
        self.test_dict = {'response': {}}
        self.test_dict['response']['status'] = 'error'
        self.test_dict['response']['message'] = "Error message"
        self.counter = 0


class GetResultNoStatus(GetResultReturn):
    """A put mock class for testing error responses
    This will not return a status in the second call"""

    def update(self, *args, **kwargs):
        del self.test_dict['response']['status']
        return self


class GetResultErrorSecond(GetResultReturn):
    """A put mock class for testing error responses
    This will return an error on the second put call"""

    def update(self, *args, **kwargs):
        if self.counter >= 1:
            self.test_dict['response']['status'] = 'error'
        return self


class SubmitGoodResponse:
    def json(self):
        return {"job": {"job_id": "test_job"}, "response": {"status": "queued"}}


class SubmitResultNoID:
    """A put mock class for testing error responses
    This will not return an id at the first call"""

    def json(self):
        return {"job": {}, "response": {"status": "queued"}}


class SubmitResultNoStatus:
    """A put mock class for testing error responses
    This will not return an id at the first call"""

    def json(self):
        return {"job": {"job_id": "test_job"}, "response": {}}


class SubmitResultWithError:
    """A put mock class for testing error responses
    This will not return an id at the first call"""

    def json(self):
        return {"job": {"job_id": "test_job"}, "response": {"status": "error"}}


def test_aqt_sampler_submit_job_error_handling():
    for e_return in [SubmitResultNoID(), SubmitResultNoStatus(), SubmitResultWithError()]:
        with (
            mock.patch('cirq_aqt.aqt_sampler.post', return_value=e_return),
            mock.patch('cirq_aqt.aqt_sampler.get', return_value=GetResultReturn()),
        ):
            theta = sympy.Symbol('theta')
            num_points = 1
            max_angle = np.pi
            repetitions = 10
            sampler = AQTSampler(access_token='testkey', workspace="default", resource="test")
            _, qubits = get_aqt_device(1)
            circuit = cirq.Circuit(
                cirq.PhasedXPowGate(exponent=theta, phase_exponent=0.0).on(qubits[0])
            )
            sweep = cirq.Linspace(key='theta', start=0.1, stop=max_angle / np.pi, length=num_points)
            with pytest.raises(RuntimeError):
                _results = sampler.run_sweep(circuit, params=sweep, repetitions=repetitions)


def test_aqt_sampler_get_result_error_handling():
    for e_return in [GetResultError(), GetResultErrorSecond(), GetResultNoStatus()]:
        with (
            mock.patch('cirq_aqt.aqt_sampler.post', return_value=SubmitGoodResponse()),
            mock.patch(
                'cirq_aqt.aqt_sampler.get', return_value=e_return, side_effect=e_return.update
            ),
        ):
            theta = sympy.Symbol('theta')
            num_points = 1
            max_angle = np.pi
            repetitions = 10
            sampler = AQTSampler(access_token='testkey', workspace="default", resource="test")
            _, qubits = get_aqt_device(1)
            circuit = cirq.Circuit(
                cirq.PhasedXPowGate(exponent=theta, phase_exponent=0.0).on(qubits[0])
            )
            sweep = cirq.Linspace(key='theta', start=0.1, stop=max_angle / np.pi, length=num_points)
            with pytest.raises(RuntimeError):
                _results = sampler.run_sweep(circuit, params=sweep, repetitions=repetitions)


def test_aqt_sampler_empty_circuit():
    num_points = 10
    max_angle = np.pi
    repetitions = 1000
    num_qubits = 4
    _, _qubits = get_aqt_device(num_qubits)
    sampler = AQTSamplerLocalSimulator()
    sampler.simulate_ideal = True
    circuit = cirq.Circuit()
    sweep = cirq.Linspace(key='theta', start=0.1, stop=max_angle / np.pi, length=num_points)
    with pytest.raises(RuntimeError):
        _results = sampler.run_sweep(circuit, params=sweep, repetitions=repetitions)


def test_aqt_sampler():
    class ResultReturn:
        def __init__(self):
            self.request_counter = 0
            self.status = "queued"

        def json(self):
            return {"response": {"status": self.status, "result": {"0": [[1, 1], [0, 0]]}}}

        def on_request(self, *args, **kwargs):
            self.request_counter += 1
            if self.request_counter >= 3:
                self.status = "finished"
            return self

    result_return = ResultReturn()

    with (
        mock.patch('cirq_aqt.aqt_sampler.post', return_value=SubmitGoodResponse()) as submit_method,
        mock.patch(
            'cirq_aqt.aqt_sampler.get',
            return_value=result_return,
            side_effect=result_return.on_request,
        ) as result_method,
    ):
        theta = sympy.Symbol('theta')
        num_points = 1
        max_angle = np.pi
        repetitions = 10
        sampler = AQTSampler(access_token='testkey', workspace="default", resource="test")
        _, qubits = get_aqt_device(1)
        circuit = cirq.Circuit(
            cirq.PhasedXPowGate(exponent=theta, phase_exponent=0.0).on(qubits[0])
        )
        sweep = cirq.Linspace(key='theta', start=0.1, stop=max_angle / np.pi, length=num_points)
        results = sampler.run_sweep(circuit, params=sweep, repetitions=repetitions)
        excited_state_probs = np.zeros(num_points)

        for i in range(num_points):
            excited_state_probs[i] = np.mean(results[i].measurements['m'])

    assert submit_method.call_count == 1
    assert result_method.call_count == 3


def test_aqt_sampler_sim():
    theta = sympy.Symbol('theta')
    num_points = 10
    max_angle = np.pi
    repetitions = 1000
    num_qubits = 4
    _, qubits = get_aqt_device(num_qubits)
    sampler = AQTSamplerLocalSimulator()
    sampler.simulate_ideal = True
    circuit = cirq.Circuit(
        cirq.PhasedXPowGate(phase_exponent=0.0, exponent=theta).on(qubits[3]),
        cirq.PhasedXPowGate(phase_exponent=0.0, exponent=1.0).on(qubits[0]),
        cirq.PhasedXPowGate(phase_exponent=0.0, exponent=1.0).on(qubits[0]),
        cirq.PhasedXPowGate(phase_exponent=0.0, exponent=1.0).on(qubits[1]),
        cirq.PhasedXPowGate(phase_exponent=0.0, exponent=1.0).on(qubits[1]),
        cirq.PhasedXPowGate(phase_exponent=0.0, exponent=1.0).on(qubits[2]),
        cirq.PhasedXPowGate(phase_exponent=0.0, exponent=1.0).on(qubits[2]),
    )
    circuit.append(cirq.PhasedXPowGate(phase_exponent=0.5, exponent=-0.5).on(qubits[0]))
    circuit.append(cirq.PhasedXPowGate(phase_exponent=0.5, exponent=0.5).on(qubits[0]))
    sweep = cirq.Linspace(key='theta', start=0.1, stop=max_angle / np.pi, length=num_points)
    results = sampler.run_sweep(circuit, params=sweep, repetitions=repetitions)
    excited_state_probs = np.zeros(num_points)
    for i in range(num_points):
        excited_state_probs[i] = np.mean(results[i].measurements['m'])
    assert excited_state_probs[-1] == 0.25


def test_aqt_sampler_sim_xtalk():
    num_points = 10
    max_angle = np.pi
    repetitions = 100
    num_qubits = 4
    _, qubits = get_aqt_device(num_qubits)
    sampler = AQTSamplerLocalSimulator()
    sampler.simulate_ideal = False
    circuit = cirq.Circuit(
        cirq.PhasedXPowGate(phase_exponent=0.0, exponent=1.0).on(qubits[0]),
        cirq.PhasedXPowGate(phase_exponent=0.0, exponent=1.0).on(qubits[1]),
        cirq.PhasedXPowGate(phase_exponent=0.0, exponent=1.0).on(qubits[1]),
        cirq.PhasedXPowGate(phase_exponent=0.0, exponent=1.0).on(qubits[3]),
        cirq.PhasedXPowGate(phase_exponent=0.0, exponent=1.0).on(qubits[2]),
        cirq.XX(qubits[0], qubits[1]) ** 0.5,
        cirq.Z.on_each(*qubits),
    )
    sweep = cirq.Linspace(key='theta', start=0.1, stop=max_angle / np.pi, length=num_points)
    _results = sampler.run_sweep(circuit, params=sweep, repetitions=repetitions)


def test_aqt_sampler_ms():
    repetitions = 1000
    num_qubits = 4
    _, qubits = get_aqt_device(num_qubits)
    sampler = AQTSamplerLocalSimulator()
    circuit = cirq.Circuit(cirq.Z.on_each(*qubits), cirq.Z.on_each(*qubits))
    for _ in range(9):
        circuit.append(cirq.XX(qubits[0], qubits[1]) ** 0.5)
    circuit.append(cirq.Z(qubits[0]) ** 0.5)
    results = sampler.run(circuit, repetitions=repetitions)
    hist = results.histogram(key='m')
    assert hist[12] > repetitions / 3
    assert hist[0] > repetitions / 3


def test_aqt_device_wrong_op_str():
    circuit = cirq.Circuit()
    q0, q1 = cirq.LineQubit.range(2)
    circuit.append(cirq.CNOT(q0, q1) ** 1.0)
    for op in circuit.all_operations():
        with pytest.raises(ValueError):
            _result = get_op_string(op)


def test_aqt_sampler_parses_legacy_json_correctly() -> None:
    legacy_json = json.dumps(
        [
            ["R", 1.0, 0.0, [0]],
            ["MS", 0.5, [0, 1]],
            ["Z", -0.5, [0]],
            ["R", 0.5, 1.0, [0]],
            ["R", 0.5, 1.0, [1]],
        ]
    )

    sampler = AQTSampler("default", "test", "testkey")
    quantum_circuit = sampler._parse_legacy_circuit_json(legacy_json)

    assert quantum_circuit == [
        {"operation": "R", "phi": 0.0, "theta": 1.0, "qubit": 0},
        {"operation": "RXX", "qubits": [0, 1], "theta": 0.5},
        {"operation": "RZ", "qubit": 0, "phi": -0.5},
        {"operation": "R", "qubit": 0, "theta": 0.5, "phi": 1.0},
        {"operation": "R", "qubit": 1, "theta": 0.5, "phi": 1.0},
        {"operation": "MEASURE"},
    ]


def test_aqt_sampler_submits_jobs_correctly() -> None:
    legacy_json = json.dumps(
        [
            ["R", 1.0, 0.0, [0]],
            ["MS", 0.5, [0, 1]],
            ["Z", -0.5, [0]],
            ["R", 0.5, 1.0, [0]],
            ["R", 0.5, 1.0, [1]],
        ]
    )

    result = [[1, 1], [0, 0]]

    class ResultReturn:
        def json(self):
            return {"response": {"status": "finished", "result": {"0": result}}}

    sampler = AQTSampler("default", "test", "testkey", "http://localhost:7777/api/v1/")

    with (
        mock.patch('cirq_aqt.aqt_sampler.post', return_value=SubmitGoodResponse()) as submit_method,
        mock.patch('cirq_aqt.aqt_sampler.get', return_value=ResultReturn()) as result_method,
    ):
        measurements = sampler._send_json(
            json_str=legacy_json, id_str="test", repetitions=2, num_qubits=2
        )

        assert submit_method.call_count == 1
        assert submit_method.call_args[0][0] == "http://localhost:7777/api/v1/submit/default/test"

        assert result_method.call_count == 1
        assert result_method.call_args[0][0] == "http://localhost:7777/api/v1/result/test_job"

        for i, rep in enumerate(measurements):
            for j, sample in enumerate(rep):
                assert sample == result[i][j]


def test_measurement_not_at_end_is_not_allowed() -> None:
    legacy_json = json.dumps([["R", 1.0, 0.0, [0]], ["Meas"], ["MS", 0.5, [0, 1]]])

    sampler = AQTSampler("default", "dummy_resource", "test")
    with pytest.raises(ValueError):
        sampler._send_json(json_str=legacy_json, id_str="test")


def test_multiple_measurements_are_not_allowed() -> None:
    legacy_json = json.dumps([["R", 1.0, 0.0, [0]], ["Meas"], ["Meas"]])

    sampler = AQTSampler("default", "dummy_resource", "test")
    with pytest.raises(ValueError):
        sampler._send_json(json_str=legacy_json, id_str="test")


def test_unknown_gate_in_json() -> None:
    legacy_json = json.dumps([["A", 1.0, 0.0, [0]], ["Meas"]])

    sampler = AQTSampler("default", "dummy_resource", "test")
    with pytest.raises(
        ValueError, match=r"Got unknown gate on operation: \['A', 1\.0, 0\.0, \[0\]\]\."
    ):
        sampler._send_json(json_str=legacy_json, id_str="test")


def test_aqt_sampler_raises_exception_on_bad_result_response() -> None:
    legacy_json = json.dumps([["R", 1.0, 0.0, [0]]])

    class ResultReturn:
        def json(self):
            return {"response": {"status": "finished"}}

    sampler = AQTSampler("default", "test", "testkey", "http://localhost:7777/api/v1/")

    with (
        mock.patch('cirq_aqt.aqt_sampler.post', return_value=SubmitGoodResponse()),
        mock.patch('cirq_aqt.aqt_sampler.get', return_value=ResultReturn()),
        pytest.raises(RuntimeError),
    ):
        sampler._send_json(json_str=legacy_json, id_str="test", repetitions=2, num_qubits=2)


def test_aqt_sampler_print_resources_shows_hint_if_no_workspaces() -> None:
    output = []

    def intercept(values):
        output.append(str(values))

    with mock.patch('cirq_aqt.aqt_sampler.AQTSampler.fetch_resources', return_value=[]):
        AQTSampler.print_resources(access_token="test", emit=intercept)

    assert output[0] == "No workspaces are accessible with this access token."


def test_aqt_sampler_print_resources_shows_hint_if_no_resources() -> None:
    output = []

    def intercept(values):
        output.append(str(values))

    empty_workspace_list = [{"id": "test_ws", "resources": []}]

    with mock.patch(
        'cirq_aqt.aqt_sampler.AQTSampler.fetch_resources', return_value=empty_workspace_list
    ):
        AQTSampler.print_resources("test", emit=intercept)

    assert output[0] == "No workspaces accessible with this access token contain resources."


def test_aqt_sampler_print_resources_includes_received_resources_in_table() -> None:
    output = []

    def intercept(values):
        output.append(str(values))

    workspace_list = [
        {"id": "test_ws", "resources": [{"id": "resource", "name": "Resource", "type": "device"}]}
    ]

    with mock.patch('cirq_aqt.aqt_sampler.AQTSampler.fetch_resources', return_value=workspace_list):
        AQTSampler.print_resources("test", emit=intercept)

    assert any("test_ws" in o and "resource" in o and "Resource" in o and "D" in o for o in output)


def test_aqt_sampler_fetch_resources_raises_exception_if_non_200_status_code() -> None:
    class ResourceResponse:
        def __init__(self):
            self.status_code = 403

        def json(self):
            return "error"

    sampler = AQTSampler("default", "test", "testkey", "http://localhost:7777/api/v1/")

    with (
        mock.patch('cirq_aqt.aqt_sampler.get', return_value=ResourceResponse()),
        pytest.raises(RuntimeError),
    ):
        sampler.fetch_resources("token")


def test_aqt_sampler_fetch_resources_returns_retrieved_resources() -> None:
    class ResourceResponse:
        def __init__(self):
            self.status_code = 200

        def json(self):
            return [
                {"id": "wid", "resources": [{"id": "rid", "name": "Resource", "type": "device"}]}
            ]

    sampler = AQTSampler("default", "test", "testkey", "http://localhost:7777/api/v1/")

    with mock.patch('cirq_aqt.aqt_sampler.get', return_value=ResourceResponse()):
        workspaces = sampler.fetch_resources("token")

    assert workspaces[0]["id"] == "wid"
    assert workspaces[0]["resources"][0]["id"] == "rid"
    assert workspaces[0]["resources"][0]["name"] == "Resource"
    assert workspaces[0]["resources"][0]["type"] == "device"
